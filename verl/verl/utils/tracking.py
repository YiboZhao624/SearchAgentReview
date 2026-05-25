# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A unified tracking interface that supports logging data to different backend
"""

import dataclasses
import json
import os
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any

import orjson


def build_token_clip_scatter(swanlab_module, data: dict, step: int, max_points: int = 2000):
    """Build and log a token clip scatter plot to SwanLab.

    Creates a scatter chart showing old policy probability (x-axis) vs probability ratio (y-axis),
    color-coded by advantage sign (green = positive, red = negative), with clip boundary dash lines.

    Args:
        swanlab_module: The swanlab module instance.
        data: Metrics dict containing _scatter_old_prob, _scatter_ratio, _scatter_advantages arrays.
        step: Current training step.
        max_points: Maximum number of scatter points to display (downsampled if exceeded).
    """
    import logging
    import numpy as np

    logger = logging.getLogger(__name__)

    old_prob = data.get("actor/_scatter_old_prob", None)
    ratio = data.get("actor/_scatter_ratio", None)
    advantages = data.get("actor/_scatter_advantages", None)

    if old_prob is None or ratio is None or advantages is None:
        return

    # Data may arrive as a list of arrays from multiple workers (e.g. 8 GPUs).
    # Concatenate them into a single flat array.
    def _to_flat_array(x):
        if isinstance(x, list):
            return np.concatenate([np.asarray(a, dtype=np.float32).ravel() for a in x])
        return np.asarray(x, dtype=np.float32).ravel()

    old_prob = _to_flat_array(old_prob)
    ratio = _to_flat_array(ratio)
    advantages = _to_flat_array(advantages)

    logger.info(
        f"Scatter data stats - ratio: min={ratio.min():.4f}, max={ratio.max():.4f}, "
        f"mean={ratio.mean():.4f}, count={len(ratio)}"
    )

    # Split by advantage sign
    pos_mask = advantages > 0
    neg_mask = advantages <= 0

    old_prob_pos, ratio_pos = old_prob[pos_mask], ratio[pos_mask]
    old_prob_neg, ratio_neg = old_prob[neg_mask], ratio[neg_mask]

    # Downsample if needed
    total = len(old_prob_pos) + len(old_prob_neg)
    if total > max_points and total > 0:
        frac = max_points / total
        n_pos = max(1, int(len(old_prob_pos) * frac)) if len(old_prob_pos) > 0 else 0
        n_neg = max(1, int(len(old_prob_neg) * frac)) if len(old_prob_neg) > 0 else 0
        if len(old_prob_pos) > n_pos:
            idx = np.random.choice(len(old_prob_pos), n_pos, replace=False)
            old_prob_pos, ratio_pos = old_prob_pos[idx], ratio_pos[idx]
        if len(old_prob_neg) > n_neg:
            idx = np.random.choice(len(old_prob_neg), n_neg, replace=False)
            old_prob_neg, ratio_neg = old_prob_neg[idx], ratio_neg[idx]

    # Read clip ratios from data (passed from trainer)
    clip_ratio_low = data.get("actor/_clip_ratio_low", 0.2)
    clip_ratio_high = data.get("actor/_clip_ratio_high", 0.2)
    clip_ratio_c = data.get("actor/_clip_ratio_c", 3.0)

    from pyecharts import options as opts
    from pyecharts.charts import Scatter as PyechartsScatter

    # Use pyecharts Scatter directly — SwanLab is fully compatible with pyecharts objects
    # Log positive and negative series as separate scatter charts under one key
    scatter = PyechartsScatter(init_opts=opts.InitOpts(width="900px", height="500px"))

    # Sort positive series by old_prob for rendering
    if len(old_prob_pos) > 0:
        pos_order = np.argsort(old_prob_pos)
        x_pos = [float(v) for v in old_prob_pos[pos_order]]
        y_pos = [float(v) for v in ratio_pos[pos_order]]
        scatter.add_xaxis(x_pos)
        scatter.add_yaxis(
            "Advantage > 0",
            y_pos,
            symbol_size=4,
            symbol="circle",
            label_opts=opts.LabelOpts(is_show=False),
            itemstyle_opts=opts.ItemStyleOpts(color="rgba(76, 175, 80, 0.6)"),
        )
    else:
        scatter.add_xaxis([0])
        scatter.add_yaxis("Advantage > 0", [None], label_opts=opts.LabelOpts(is_show=False))

    # For the negative series, use a second Scatter and overlap via Grid,
    # OR just log a second scatter. For simplicity, log negative as second chart.
    scatter2 = PyechartsScatter(init_opts=opts.InitOpts(width="900px", height="500px"))
    if len(old_prob_neg) > 0:
        neg_order = np.argsort(old_prob_neg)
        x_neg = [float(v) for v in old_prob_neg[neg_order]]
        y_neg = [float(v) for v in ratio_neg[neg_order]]
        scatter2.add_xaxis(x_neg)
        scatter2.add_yaxis(
            "Advantage < 0",
            y_neg,
            symbol_size=4,
            symbol="rect",
            label_opts=opts.LabelOpts(is_show=False),
            itemstyle_opts=opts.ItemStyleOpts(color="rgba(244, 67, 54, 0.6)"),
        )
    else:
        scatter2.add_xaxis([0])
        scatter2.add_yaxis("Advantage < 0", [None], label_opts=opts.LabelOpts(is_show=False))

    # Set axes and marklines for both charts
    markline_data = [
        opts.MarkLineItem(y=1 - clip_ratio_low, name=f"1-ε={1 - clip_ratio_low:.2f}"),
        opts.MarkLineItem(y=1 + clip_ratio_high, name=f"1+ε={1 + clip_ratio_high:.2f}"),
        opts.MarkLineItem(y=clip_ratio_c, name=f"c={clip_ratio_c:.1f}"),
    ]
    global_opts = dict(
        title_opts=opts.TitleOpts(title="Token Clip Scatter (Adv > 0)"),
        xaxis_opts=opts.AxisOpts(name="old policy prob", type_="value"),
        yaxis_opts=opts.AxisOpts(name="ratio (new/old)", type_="value"),
        tooltip_opts=opts.TooltipOpts(trigger="item"),
    )
    series_opts = dict(
        markline_opts=opts.MarkLineOpts(
            data=markline_data,
            linestyle_opts=opts.LineStyleOpts(type_="dashed", width=2),
        )
    )
    scatter.set_global_opts(**global_opts)
    scatter.set_series_opts(**series_opts)

    global_opts["title_opts"] = opts.TitleOpts(title="Token Clip Scatter (Adv < 0)")
    scatter2.set_global_opts(**global_opts)
    scatter2.set_series_opts(**series_opts)

    swanlab_module.log({
        "actor/token_clip_scatter_pos": scatter,
        "actor/token_clip_scatter_neg": scatter2,
    }, step=step)

def compute_group_step_outcome_metrics(
    outcome_scores: list[float], step_means: list[float], rollout_n: int
) -> dict[str, float]:
    """Compute group-level step/outcome correlation metrics.

    For each group (rollout_n consecutive samples sharing a prompt), splits samples into
    success (outcome_score == 1) and failure subsets, then reports the mean and std of
    rubric_step_mean within each subset.  These metrics reveal whether the step-level
    reward signal aligns with the outcome reward.

    Additionally reports:
    - all_success_group_ratio: fraction of groups where every rollout succeeded
    - all_failure_group_ratio: fraction where every rollout failed
    - mixed_group_ratio: fraction with both successes and failures (most useful for training)
    """
    import numpy as np

    n = len(outcome_scores)
    if n == 0 or rollout_n <= 0:
        return {}

    success_step_means: list[float] = []
    failure_step_means: list[float] = []
    all_success_groups = 0
    all_failure_groups = 0
    mixed_groups = 0

    for i in range(0, n, rollout_n):
        group_outcomes = outcome_scores[i : i + rollout_n]
        group_steps = step_means[i : i + rollout_n]

        n_success = sum(1 for o in group_outcomes if o >= 1.0)
        for o, s in zip(group_outcomes, group_steps):
            if o >= 1.0:
                success_step_means.append(s)
            else:
                failure_step_means.append(s)

        g = len(group_outcomes)
        if n_success == g:
            all_success_groups += 1
        elif n_success == 0:
            all_failure_groups += 1
        else:
            mixed_groups += 1

    total_groups = all_success_groups + all_failure_groups + mixed_groups
    metrics: dict[str, float] = {}

    if success_step_means:
        metrics["critic/rewards/success_step_mean"] = float(np.mean(success_step_means))
        metrics["critic/rewards/success_step_std"] = float(np.std(success_step_means))
    if failure_step_means:
        metrics["critic/rewards/failure_step_mean"] = float(np.mean(failure_step_means))
        metrics["critic/rewards/failure_step_std"] = float(np.std(failure_step_means))
    if success_step_means and failure_step_means:
        metrics["critic/rewards/step_discriminativeness_gap"] = (
            float(np.mean(success_step_means)) - float(np.mean(failure_step_means))
        )
    if total_groups > 0:
        metrics["critic/rewards/all_success_group_ratio"] = all_success_groups / total_groups
        metrics["critic/rewards/all_failure_group_ratio"] = all_failure_groups / total_groups
        metrics["critic/rewards/mixed_group_ratio"] = mixed_groups / total_groups

    return metrics


def compute_advantage_sign_flip_metrics(
    outcome_scores: list[float], final_scores: list[float], rollout_n: int
) -> dict[str, float]:
    """Fraction of trajectories where reward-based advantage sign disagrees with outcome.

    For each group (rollout_n consecutive samples), computes the group mean of final_scores.
    A "sign flip" occurs when:
      - outcome == 0 but final_score > group_mean  (failure gets positive advantage)
      - outcome > 0 but final_score < group_mean   (success gets negative advantage)

    Only trajectories in mixed groups (containing both success and failure) are counted.
    """
    n = len(outcome_scores)
    if n == 0 or rollout_n <= 0:
        return {}

    total_mixed = 0
    sign_flips = 0

    for i in range(0, n, rollout_n):
        group_outcomes = outcome_scores[i : i + rollout_n]
        group_finals = final_scores[i : i + rollout_n]

        n_success = sum(1 for o in group_outcomes if o > 0)
        # Only count mixed groups
        if n_success == 0 or n_success == len(group_outcomes):
            continue

        group_mean = sum(group_finals) / len(group_finals)
        for o, f in zip(group_outcomes, group_finals):
            total_mixed += 1
            if (o <= 0 and f > group_mean) or (o > 0 and f < group_mean):
                sign_flips += 1

    if total_mixed == 0:
        return {}

    return {"critic/rewards/advantage_sign_flip_ratio": sign_flips / total_mixed}


def log_ingroup_reward_variance_chart(
    swanlab_module, outcome_scores: list[float], rollout_n: int, step: int
) -> None:
    """Log a histogram of per-group reward variance to SwanLab.

    For each group (rollout_n samples), computes the variance of outcome_score within the
    group, then bins all group variances into a histogram and logs a bar chart.  A histogram
    skewed towards zero indicates most groups have collapsed to uniform reward — i.e., little
    gradient signal remains.
    """
    import numpy as np

    try:
        from pyecharts import options as opts
        from pyecharts.charts import Bar
    except ImportError:
        return

    n = len(outcome_scores)
    if n == 0 or rollout_n <= 1:
        return

    group_variances: list[float] = []
    for i in range(0, n, rollout_n):
        group = outcome_scores[i : i + rollout_n]
        if len(group) > 1:
            group_variances.append(float(np.var(group)))

    if not group_variances:
        return

    n_bins = 10
    max_var = max(group_variances)
    bin_edges = np.linspace(0.0, max(max_var, 1e-9), n_bins + 1)
    counts, _ = np.histogram(group_variances, bins=bin_edges)
    bin_labels = [f"{bin_edges[j]:.3f}\n-{bin_edges[j+1]:.3f}" for j in range(n_bins)]

    bar = Bar(init_opts=opts.InitOpts(width="800px", height="400px"))
    bar.add_xaxis(bin_labels)
    bar.add_yaxis(
        "# groups",
        counts.tolist(),
        itemstyle_opts=opts.ItemStyleOpts(color="rgba(66, 135, 245, 0.8)"),
        label_opts=opts.LabelOpts(is_show=False),
    )
    bar.set_global_opts(
        title_opts=opts.TitleOpts(title="In-Group Reward Variance Histogram"),
        xaxis_opts=opts.AxisOpts(name="variance", axislabel_opts=opts.LabelOpts(font_size=10)),
        yaxis_opts=opts.AxisOpts(name="# groups"),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
    )
    swanlab_module.log({"critic/ingroup_variance_hist": bar}, step=step)


def log_training_rollout_generations_to_swanlab(
    swanlab_module,
    questions: list[str],
    rollouts: list[str],
    outcome_scores: list[float],
    step_means: list[float],
    step_reasonings: list[str],
    rollout_n: int,
    step: int,
    num_questions: int = 2,
) -> None:
    """Log training rollout generations for a random sample of groups to SwanLab.

    Picks num_questions groups at random and logs every rollout within each group as a
    table row, including the question, truncated response, outcome score, step mean reward,
    and step-level judge reasoning.
    """
    import random

    n = len(questions)
    if n == 0 or rollout_n <= 0:
        return

    n_groups = n // rollout_n
    if n_groups == 0:
        return

    selected_groups = random.sample(range(n_groups), min(num_questions, n_groups))

    table = swanlab_module.echarts.Table()
    headers = ["step", "group", "rollout", "question", "response", "outcome_score", "step_mean", "step_reasonings"]
    rows = []

    for g_idx in selected_groups:
        start = g_idx * rollout_n
        for r in range(rollout_n):
            i = start + r
            if i >= n:
                break
            rows.append([
                step,
                g_idx,
                r,
                (questions[i] or "")[:300],
                (rollouts[i] or ""),
                round(float(outcome_scores[i]), 4) if i < len(outcome_scores) else 0.0,
                round(float(step_means[i]), 4) if i < len(step_means) else 0.0,
                (step_reasonings[i] or "") if i < len(step_reasonings) else "",
            ])

    if rows:
        table.add(headers=headers, rows=rows)
        swanlab_module.log({"train/rollout_generations": table}, step=step)


class Tracking:
    """A unified tracking interface for logging experiment data to multiple backends.

    This class provides a centralized way to log experiment metrics, parameters, and artifacts
    to various tracking backends including WandB, MLflow, SwanLab, TensorBoard, and console.

    Attributes:
        supported_backend: List of supported tracking backends.
        logger: Dictionary of initialized logger instances for each backend.
    """

    supported_backend = [
        "wandb",
        "mlflow",
        "swanlab",
        "vemlp_wandb",
        "tensorboard",
        "console",
        "clearml",
        "trackio",
        "file",
    ]

    def __init__(self, project_name, experiment_name, default_backend: str | list[str] = "console", config=None):
        if isinstance(default_backend, str):
            default_backend = [default_backend]
        for backend in default_backend:
            if backend == "tracking":
                import warnings

                warnings.warn("`tracking` logger is deprecated. use `wandb` instead.", DeprecationWarning, stacklevel=2)
            else:
                assert backend in self.supported_backend, f"{backend} is not supported"

        self.logger = {}

        if "tracking" in default_backend or "wandb" in default_backend:
            import os

            import wandb

            settings = None
            if config and config["trainer"].get("wandb_proxy", None):
                settings = wandb.Settings(https_proxy=config["trainer"]["wandb_proxy"])
            entity = os.environ.get("WANDB_ENTITY", None)
            wandb.init(project=project_name, name=experiment_name, entity=entity, config=config, settings=settings)
            self.logger["wandb"] = wandb

        if "trackio" in default_backend:
            import trackio

            trackio.init(project=project_name, name=experiment_name, config=config)
            self.logger["trackio"] = trackio

        if "mlflow" in default_backend:
            import os

            import mlflow

            MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:////tmp/mlruns.db")
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

            # Some cloud providers like Azure ML or Databricks automatically set MLFLOW_RUN_ID
            # If set, attach to the existing run instead of creating a new one
            run_id = os.environ.get("MLFLOW_RUN_ID")
            if run_id:
                mlflow.start_run(run_id=run_id)
            else:
                # Project_name is actually experiment_name in MLFlow
                # If experiment does not exist, will create a new experiment
                experiment = mlflow.set_experiment(project_name)
                mlflow.start_run(experiment_id=experiment.experiment_id, run_name=experiment_name)

            mlflow.log_params(_compute_mlflow_params_from_objects(config))
            self.logger["mlflow"] = _MlflowLoggingAdapter()

        if "swanlab" in default_backend:
            import os

            import swanlab

            SWANLAB_API_KEY = os.environ.get("SWANLAB_API_KEY", None)
            SWANLAB_LOG_DIR = os.environ.get("SWANLAB_LOG_DIR", "swanlog")
            SWANLAB_MODE = os.environ.get("SWANLAB_MODE", "cloud")
            if SWANLAB_API_KEY:
                swanlab.login(SWANLAB_API_KEY)  # NOTE: previous login information will be overwritten

            if config is None:
                config = {}  # make sure config is not None, otherwise **config will raise error
            swanlab.init(
                project=project_name,
                experiment_name=experiment_name,
                config={"FRAMEWORK": "verl", **config},
                logdir=SWANLAB_LOG_DIR,
                mode=SWANLAB_MODE,
            )
            self.logger["swanlab"] = swanlab

        if "vemlp_wandb" in default_backend:
            import os

            import volcengine_ml_platform
            from volcengine_ml_platform import wandb as vemlp_wandb

            volcengine_ml_platform.init(
                ak=os.environ["VOLC_ACCESS_KEY_ID"],
                sk=os.environ["VOLC_SECRET_ACCESS_KEY"],
                region=os.environ["MLP_TRACKING_REGION"],
            )

            vemlp_wandb.init(
                project=project_name,
                name=experiment_name,
                config=config,
                sync_tensorboard=True,
            )
            self.logger["vemlp_wandb"] = vemlp_wandb

        if "tensorboard" in default_backend:
            self.logger["tensorboard"] = _TensorboardAdapter(project_name, experiment_name)

        if "console" in default_backend:
            from verl.utils.logger import LocalLogger

            self.console_logger = LocalLogger(print_to_console=True)
            self.logger["console"] = self.console_logger

        if "clearml" in default_backend:
            self.logger["clearml"] = ClearMLLogger(project_name, experiment_name, config)

        if "file" in default_backend:
            self.logger["file"] = FileLogger(project_name, experiment_name)

    def log(self, data, step, backend=None):
        # Handle token clip scatter plot for SwanLab
        if "swanlab" in self.logger:
            has_scatter = any(k.startswith("actor/_scatter_") for k in data)
            if has_scatter:
                try:
                    build_token_clip_scatter(self.logger["swanlab"], data, step)
                except Exception as e:
                    import logging

                    logging.getLogger(__name__).warning(f"Failed to log token clip scatter: {e}")

        # Filter out scatter data keys from regular logging (they are numpy arrays, not scalars)
        filtered_data = {k: v for k, v in data.items() if not k.startswith("actor/_scatter_") and not k.startswith("actor/_clip_ratio")}

        for default_backend, logger_instance in self.logger.items():
            if backend is None or default_backend in backend:
                logger_instance.log(data=filtered_data, step=step)

    def __del__(self):
        if "wandb" in self.logger:
            self.logger["wandb"].finish(exit_code=0)
        if "swanlab" in self.logger:
            self.logger["swanlab"].finish()
        if "vemlp_wandb" in self.logger:
            self.logger["vemlp_wandb"].finish(exit_code=0)
        if "tensorboard" in self.logger:
            self.logger["tensorboard"].finish()
        if "clearml" in self.logger:
            self.logger["clearml"].finish()
        if "trackio" in self.logger:
            self.logger["trackio"].finish()
        if "file" in self.logger:
            self.logger["file"].finish()


class ClearMLLogger:
    def __init__(self, project_name: str, experiment_name: str, config):
        self.project_name = project_name
        self.experiment_name = experiment_name

        import clearml

        self._task: clearml.Task = clearml.Task.init(
            task_name=experiment_name,
            project_name=project_name,
            continue_last_task=True,
            output_uri=False,
        )

        self._task.connect_configuration(config, name="Hyperparameters")

    def _get_logger(self):
        return self._task.get_logger()

    def log(self, data, step):
        import numpy as np
        import pandas as pd

        # logs = self._rewrite_logs(data)
        logger = self._get_logger()
        for k, v in data.items():
            title, series = k.split("/", 1)

            if isinstance(v, int | float | np.floating | np.integer):
                logger.report_scalar(
                    title=title,
                    series=series,
                    value=v,
                    iteration=step,
                )
            elif isinstance(v, pd.DataFrame):
                logger.report_table(
                    title=title,
                    series=series,
                    table_plot=v,
                    iteration=step,
                )
            else:
                logger.warning(
                    f'Trainer is attempting to log a value of "{v}" of type {type(v)} for key "{k}". This '
                    f"invocation of ClearML logger's function is incorrect so this attribute was dropped. "
                )

    def finish(self):
        self._task.close()


class FileLogger:
    def __init__(self, project_name: str, experiment_name: str):
        self.project_name = project_name
        self.experiment_name = experiment_name

        self.filepath = os.getenv("VERL_FILE_LOGGER_PATH", None)
        if self.filepath is None:
            root_path = os.path.expanduser(os.getenv("VERL_FILE_LOGGER_ROOT", "."))
            directory = os.path.join(root_path, self.project_name)
            os.makedirs(directory, exist_ok=True)
            self.filepath = os.path.join(directory, f"{self.experiment_name}.jsonl")
            print(f"Creating file logger at {self.filepath}")
        self.fp = open(self.filepath, "wb", buffering=0)

    def log(self, data, step):
        data = {"step": step, "data": data}
        self.fp.write(orjson.dumps(data, option=orjson.OPT_SERIALIZE_NUMPY) + b"\n")

    def finish(self):
        self.fp.close()


class _TensorboardAdapter:
    def __init__(self, project_name, experiment_name):
        import os

        from torch.utils.tensorboard import SummaryWriter

        tensorboard_dir = os.environ.get("TENSORBOARD_DIR", f"tensorboard_log/{project_name}/{experiment_name}")
        os.makedirs(tensorboard_dir, exist_ok=True)
        print(f"Saving tensorboard log to {tensorboard_dir}.")
        self.writer = SummaryWriter(tensorboard_dir)

    def log(self, data, step):
        for key in data:
            self.writer.add_scalar(key, data[key], step)

    def finish(self):
        self.writer.close()


class _MlflowLoggingAdapter:
    def __init__(self):
        import logging
        import re

        self.logger = logging.getLogger(__name__)
        # MLflow metric key validation logic:
        # https://github.com/mlflow/mlflow/blob/master/mlflow/utils/validation.py#L157C12-L157C44
        # Only characters allowed: slashes, alphanumerics, underscores, periods, dashes, colons,
        # and spaces.
        self._invalid_chars_pattern = re.compile(
            r"[^/\w.\- :]"
        )  # Allowed: slashes, alphanumerics, underscores, periods, dashes, colons, and spaces.
        self._consecutive_slashes_pattern = re.compile(r"/+")

    def log(self, data, step):
        import mlflow

        def sanitize_key(key):
            # First replace @ with _at_ for backward compatibility
            sanitized = key.replace("@", "_at_")
            # Replace consecutive slashes with a single slash (MLflow treats them as file paths)
            sanitized = self._consecutive_slashes_pattern.sub("/", sanitized)
            # Then replace any other invalid characters with _
            sanitized = self._invalid_chars_pattern.sub("_", sanitized)
            if sanitized != key:
                self.logger.warning(
                    "[MLflow] Metric key '%s' sanitized to '%s' due to invalid characters.", key, sanitized
                )
            return sanitized

        results = {sanitize_key(k): v for k, v in data.items()}
        mlflow.log_metrics(metrics=results, step=step)


def _compute_mlflow_params_from_objects(params) -> dict[str, Any]:
    if params is None:
        return {}

    return _flatten_dict(_transform_params_to_json_serializable(params, convert_list_to_dict=True), sep="/")


def _transform_params_to_json_serializable(x, convert_list_to_dict: bool):
    _transform = partial(_transform_params_to_json_serializable, convert_list_to_dict=convert_list_to_dict)

    if dataclasses.is_dataclass(x):
        return _transform(dataclasses.asdict(x))
    if isinstance(x, dict):
        return {k: _transform(v) for k, v in x.items()}
    if isinstance(x, list):
        if convert_list_to_dict:
            return {"list_len": len(x)} | {f"{i}": _transform(v) for i, v in enumerate(x)}
        else:
            return [_transform(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, Enum):
        return x.value

    return x


def _flatten_dict(raw: dict[str, Any], *, sep: str) -> dict[str, Any]:
    import pandas as pd

    ans = pd.json_normalize(raw, sep=sep).to_dict(orient="records")[0]
    assert isinstance(ans, dict)
    return ans


@dataclasses.dataclass
class ValidationGenerationsLogger:
    project_name: str = None
    experiment_name: str = None

    def log(self, loggers, samples, step):
        if "wandb" in loggers:
            self.log_generations_to_wandb(samples, step)
        if "swanlab" in loggers:
            self.log_generations_to_swanlab(samples, step)
        if "mlflow" in loggers:
            self.log_generations_to_mlflow(samples, step)

        if "clearml" in loggers:
            self.log_generations_to_clearml(samples, step)
        if "tensorboard" in loggers:
            self.log_generations_to_tensorboard(samples, step)

        if "vemlp_wandb" in loggers:
            self.log_generations_to_vemlp_wandb(samples, step)

    def log_generations_to_vemlp_wandb(self, samples, step):
        from volcengine_ml_platform import wandb as vemlp_wandb

        self._log_generations_to_wandb(samples, step, vemlp_wandb)

    def log_generations_to_wandb(self, samples, step):
        import wandb

        self._log_generations_to_wandb(samples, step, wandb)

    def _log_generations_to_wandb(self, samples, step, wandb):
        """Log samples to wandb as a table"""

        # Create column names for all samples
        columns = ["step"] + sum(
            [[f"input_{i + 1}", f"output_{i + 1}", f"score_{i + 1}"] for i in range(len(samples))], []
        )

        if not hasattr(self, "validation_table"):
            # Initialize the table on first call
            self.validation_table = wandb.Table(columns=columns)

        # Create a new table with same columns and existing data
        # Workaround for https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
        new_table = wandb.Table(columns=columns, data=self.validation_table.data)

        # Add new row with all data
        row_data = []
        row_data.append(step)
        for sample in samples:
            row_data.extend(sample)

        new_table.add_data(*row_data)

        # Update reference and log
        if wandb.run is not None:
            wandb.log({"val/generations": new_table}, step=step)
        self.validation_table = new_table

    def log_generations_to_swanlab(self, samples, step):
        """Log samples to swanlab as text"""
        import swanlab

        swanlab_table = swanlab.echarts.Table()
        
        # Create column names
        headers = ["step", "input", "output", "score"]
        
        swanlab_row_list = [[step, *sample] for sample in samples]
        swanlab_table.add(headers=headers, rows=swanlab_row_list)
        
        # Log to swanlab
        swanlab.log({"val/generations": swanlab_table}, step=step)

    def log_generations_to_mlflow(self, samples, step):
        """Log validation generation to mlflow as artifacts"""
        # https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html?highlight=log_artifact#mlflow.log_artifact

        import tempfile

        import mlflow

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                validation_gen_step_file = Path(tmp_dir, f"val_step{step}.json")
                row_data = []
                for sample in samples:
                    data = {"input": sample[0], "output": sample[1], "score": sample[2]}
                    row_data.append(data)
                with open(validation_gen_step_file, "w") as file:
                    json.dump(row_data, file)
                mlflow.log_artifact(validation_gen_step_file)
        except Exception as e:
            print(f"WARNING: save validation generation file to mlflow failed with error {e}")

    def log_generations_to_clearml(self, samples, step):
        """Log validation generation to clearml as table"""

        import clearml
        import pandas as pd

        task: clearml.Task | None = clearml.Task.current_task()
        if task is None:
            return

        table = [
            {
                "step": step,
                "input": sample[0],
                "output": sample[1],
                "score": sample[2],
            }
            for sample in samples
        ]

        logger = task.get_logger()
        logger.report_table(
            series="Validation generations",
            title="Validation",
            table_plot=pd.DataFrame.from_records(table),
            iteration=step,
        )

    def log_generations_to_tensorboard(self, samples, step):
        """Log samples to tensorboard as text"""
        # Initialize tensorboard writer if not exists
        if not hasattr(self, "writer"):
            from torch.utils.tensorboard import SummaryWriter

            # Use the same directory structure as _TensorboardAdapter
            if self.project_name and self.experiment_name:
                default_dir = os.path.join("tensorboard_log", self.project_name, self.experiment_name)
            else:
                default_dir = "tensorboard_log"

            tensorboard_dir = os.environ.get("TENSORBOARD_DIR", default_dir)
            os.makedirs(tensorboard_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=tensorboard_dir)

        # Format the samples data into readable text
        text_content = f"**Generation Results - Step {step}**\n\n"

        for i, sample in enumerate(samples):
            text_content += f"### Sample {i + 1}\n"

            # Assuming sample contains [input, output, score]
            if len(sample) >= 3:
                input_text, output_text, score = sample[0], sample[1], sample[2]

                text_content += f"**Input:** {input_text}\n\n"
                text_content += f"**Output:** {output_text}\n\n"
                text_content += f"**Score:** {score}\n\n"
            else:
                # Handle cases where sample format might be different
                text_content += f"**Data:** {sample}\n\n"

            text_content += "---\n\n"

        # Log to tensorboard as text
        self.writer.add_text("val/generations", text_content, step)
        # Flush to ensure data is written
        self.writer.flush()
