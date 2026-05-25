"""Local search tool over a JSONL corpus produced by hotpot_local_prep."""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections import Counter
import logging
from pathlib import Path
import functools
from typing import Callable, TypeVar, ParamSpec, Any, Optional

import aiohttp

logger = logging.getLogger(__name__)

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import (
    OpenAIFunctionParametersSchema,
    OpenAIFunctionPropertySchema,
    OpenAIFunctionSchema,
    OpenAIFunctionToolSchema,
    ToolResponse,
)

class ClientError(Exception):
    """client Error, without retry."""
    pass

P = ParamSpec('P')
T = TypeVar('T')

def async_retry(
    max_retries: int = 3,
    retry_delay: float = 1.0,
    backoff_factor: float = 2.0,
    retry_on_exceptions: tuple = (
        asyncio.TimeoutError,
        aiohttp.ClientConnectionError,
        aiohttp.ClientResponseError,
        aiohttp.ClientError,
    ),
    retry_on_status_codes: set = {500, 502, 503, 504},
):
    """
    异步重试装饰器
    
    Args:
        max_retries: 最大重试次数
        retry_delay: 初始重试延迟（秒）
        backoff_factor: 指数退避因子
        retry_on_exceptions: 需要重试的异常类型
        retry_on_status_codes: 需要重试的HTTP状态码
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    result = await func(*args, **kwargs)
                    return result
                    
                except retry_on_exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        delay = retry_delay * (backoff_factor ** attempt)
                        await asyncio.sleep(delay)
                        continue
                    else:
                        raise
                        
                except Exception as e:
                    # 不在重试列表中的异常直接抛出
                    raise
            
            # 如果所有重试都失败，抛出最后一个异常
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator



def _tokenize(text: str) -> list[str]:
    return [t for t in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if t]


class LocalSearchTool(BaseTool):
    """Lightweight lexical search over preprocessed corpus."""

    def __init__(self, config: dict, tool_schema: Optional[OpenAIFunctionToolSchema] = None):
        self.corpus_path = Path(config.get("corpus_path", ""))
        self.topk_default = int(config.get("topk_default", 5))
        self.docs = []
        self.doc_tf = []
        self.idf = Counter()
        if self.corpus_path.is_file():
            self._load_corpus(self.corpus_path)
        schema = tool_schema or self._build_schema()
        super().__init__(config, schema)

    def _build_schema(self) -> OpenAIFunctionToolSchema:
        return OpenAIFunctionToolSchema(
            type="function",
            function=OpenAIFunctionSchema(
                name="local_search",
                description="Search a local HotpotQA-derived corpus and return top-k documents.",
                parameters=OpenAIFunctionParametersSchema(
                    type="object",
                    properties={
                        "query": OpenAIFunctionPropertySchema(type="string", description="User query text."),
                        "k": OpenAIFunctionPropertySchema(type="integer", description="Top-k documents to return."),
                    },
                    required=["query"],
                ),
            ),
        )

    def _load_corpus(self, path: Path):
        docs = []
        tf_list = []
        df = Counter()
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                text = row.get("text", "")
                tokens = _tokenize(text)
                tf = Counter(tokens)
                tf_list.append(tf)
                df.update(tf.keys())
                docs.append(
                    {
                        "doc_id": row.get("doc_id"),
                        "title": row.get("title"),
                        "text": text,
                    }
                )
        self.docs = docs
        self.doc_tf = tf_list
        n_docs = max(len(self.docs), 1)
        self.idf = Counter({t: math.log((n_docs + 1) / (df[t] + 1)) + 1.0 for t in df})

    def _score(self, query_tokens: list[str]) -> list[tuple[int, float]]:
        scores = []
        q_tf = Counter(query_tokens)
        for idx, tf in enumerate(self.doc_tf):
            s = 0.0
            for token, qfreq in q_tf.items():
                if token in tf:
                    s += qfreq * tf[token] * self.idf.get(token, 1.0)
            if s > 0:
                scores.append((idx, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        if not self.docs:
            return ToolResponse(text="local corpus empty"), 0.0, {"hits": 0}

        query = parameters.get("query", "")
        k = int(parameters.get("k") or self.topk_default)
        query_tokens = _tokenize(query)
        scored = self._score(query_tokens)[:k]
        hits = [
            {
                "doc_id": self.docs[i]["doc_id"],
                "title": self.docs[i]["title"],
                "text": self.docs[i]["text"],
                "score": score,
            }
            for i, score in scored
        ]
        return ToolResponse(text=json.dumps(hits, ensure_ascii=False)), 0.0, {"hits": len(hits)}


class LocalEmbeddingSearchTool(BaseTool):
    """Embedding-based search backed by a remote retrieval service."""

    _session: Optional[aiohttp.ClientSession] = None
    _profile_lock: Optional[asyncio.Lock] = None
    _profile_count: int = 0
    _profile_sums: dict[str, float] = {
        "normalize": 0.0,
        "request": 0.0,
        "parse": 0.0,
        "total": 0.0,
    }
    _profile_target: int = 10000
    _profile_output_path: Path = Path("local_embedding_search_profile.json")

    def __init__(self, config: dict, tool_schema: Optional[OpenAIFunctionToolSchema] = None):
        self.retrieval_service_url = config.get("retrieval_service_url", "")
        self.topk_default = int(config.get("topk_default", 3))
        self.timeout = int(config.get("timeout", 30))
        self.return_scores = bool(config.get("return_scores", True))
        if not self.retrieval_service_url:
            raise ValueError("retrieval_service_url is not set for LocalEmbeddingSearchTool")
        schema = tool_schema or self._build_schema()
        super().__init__(config, schema)

    @classmethod
    def _get_session(cls) -> aiohttp.ClientSession:
        if cls._session is None or cls._session.closed:
            connector = aiohttp.TCPConnector(limit=512, limit_per_host=512)
            timeout = aiohttp.ClientTimeout(total=None)
            cls._session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        return cls._session

    def _build_schema(self) -> OpenAIFunctionToolSchema:
        return OpenAIFunctionToolSchema(
            type="function",
            function=OpenAIFunctionSchema(
                name="local_search",
                description="Search a local corpus via embedding service and return top-k documents.",
                parameters=OpenAIFunctionParametersSchema(
                    type="object",
                    properties={
                        "query_list": OpenAIFunctionPropertySchema(
                            type="array",
                            description="A list of fully-formed semantic queries. The tool will return search results for each query.",
                        ),
                        "k": OpenAIFunctionPropertySchema(
                            type="integer",
                            description="Top-k documents to return; int or list[int] aligned to query_list. If k is an integer, the tool will return the top-k documents for all queries.",
                            default=3,
                        ),
                    },
                    required=["query_list"],
                ),
            ),
        )

    def _format_hits(self, raw_results: Any) -> list[dict[str, Any]]:
        hits = []
        if not raw_results:
            return hits
        if isinstance(raw_results, list) and raw_results and isinstance(raw_results[0], list):
            raw_results = raw_results[0]
        for item in raw_results or []:
            if not isinstance(item, dict):
                continue
            doc = item.get("document", {}) if isinstance(item.get("document"), dict) else {}
            content = (
                doc.get("contents")
                or item.get("contents")
                or doc.get("text")
                or item.get("text")
                or ""
            )
            title = doc.get("title") or item.get("title") or ""
            text = ""
            if content:
                parts = content.split("\n")
                if not title and parts:
                    title = parts[0]
                    text = "\n".join(parts[1:])
                else:
                    text = content
            if not text:
                text = doc.get("text") or item.get("text") or ""
            doc_id = doc.get("id") or item.get("doc_id") or item.get("id")
            hits.append(
                {
                    "doc_id": doc_id,
                    "title": title,
                    "text": text[:],
                }
            )
        return hits

    def _normalize_query_list(self, parameters: dict[str, Any]) -> list[str]:
        query_list = parameters.get("query_list")
        if not isinstance(query_list, list):
            return []
        return [q for q in query_list if isinstance(q, str) and q]

    def _normalize_k_list(self, k_value: Any, query_count: int) -> list[int]:
        try:
            if k_value is None:
                return [self.topk_default] * query_count
            if isinstance(k_value, list):
                if not k_value:
                    return [self.topk_default] * query_count
                if len(k_value) == 1:
                    k0 = int(k_value[0])
                    if k0 <= 0:
                        return []
                    return [k0] * query_count
                if len(k_value) != query_count:
                    return []
                normalized = [int(k) for k in k_value]
                return normalized if all(k > 0 for k in normalized) else []
            k_int = int(k_value)
            if k_int <= 0:
                return []
            return [k_int] * query_count
        except Exception:
            return []

    def _normalize_results(self, raw_results: Any, query_count: int) -> list[list[dict[str, Any]]]:
        if not raw_results:
            return [[] for _ in range(query_count)]
        normalized: list[list[dict[str, Any]]] = []
        if isinstance(raw_results, list) and raw_results and isinstance(raw_results[0], dict):
            normalized = [self._format_hits(raw_results)]
        elif isinstance(raw_results, list):
            normalized = [self._format_hits(item) for item in raw_results]
        if len(normalized) < query_count:
            normalized.extend([[] for _ in range(query_count - len(normalized))])
        elif len(normalized) > query_count:
            normalized = normalized[:query_count]
        return normalized

    @async_retry(max_retries=3, retry_delay=1.0, backoff_factor=2.0)
    async def _make_api_request(self, payload: dict) -> dict:
        """
        发起 API 请求（带重试装饰器）
        
        成功：返回 response_json
        失败：抛出异常
          - 5xx 错误：抛出 ClientResponseError（可重试）
          - 4xx 错误：抛出 ClientError（不可重试）
          - 网络错误：自动抛出对应异常（可重试）
        """
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        session = self._get_session()
        
        async with session.post(
            self.retrieval_service_url,
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=self.timeout),
        ) as response:
            if response.status >= 400:
                # get error detail.
                detail = ""
                try:
                    error_payload = await response.json()
                    detail = str(error_payload.get("detail", error_payload))
                except Exception:
                    try:
                        detail = await response.text()
                    except Exception:
                        detail = "Unable to read error detail"
                
                if response.status in {500, 502, 503, 504}:
                    # 5xx server error - raise retryable exception.
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status,
                        message=f"Server Error ({response.status}) detail={detail}",
                        headers=response.headers,
                    )
                else:
                    # 4xx client error - raise non-retryable exception.
                    raise ClientError(f"API Request Error: Client Error ({response.status}) detail={detail}")
            
            # successful response.
            try:
                return await response.json()
            except json.JSONDecodeError as e:
                # JSON decode error - no retry.
                raise ClientError(f"API Response JSON Decode Error: {e}")

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        total_start = time.perf_counter()
        normalize_start = total_start

        # args validation.
        query_list = self._normalize_query_list(parameters)
        if not query_list:
            await self._record_profile(time.perf_counter() - normalize_start, 0.0, 0.0, time.perf_counter() - total_start)
            return ToolResponse(text="query_list is empty"), 0.0, {"hits": 0, "status": "invalid_query"}

        k_list = self._normalize_k_list(parameters.get("k"), len(query_list))
        if not k_list:
            await self._record_profile(time.perf_counter() - normalize_start, 0.0, 0.0, time.perf_counter() - total_start)
            return ToolResponse(text="k length mismatch with query_list"), 0.0, {"hits": 0, "status": "invalid_k"}

        normalize_end = time.perf_counter()

        # prepare payload.
        request_k: int | list[int] = k_list[0] if len(set(k_list)) == 1 else k_list
        payload = {
            "query_list": query_list,
            "k": request_k,
            "return_scores": self.return_scores,
        }

        # start request with retry.
        request_start = time.perf_counter()
        try:
            # 调用带重试的 API 请求方法
            response_json = await self._make_api_request(payload)
            request_end = time.perf_counter()
            
        except (asyncio.TimeoutError, aiohttp.ClientError, ClientError) as e:
            # 捕获所有可能的异常（包括重试后仍失败的）
            request_end = time.perf_counter()
            
            # 格式化错误信息
            if isinstance(e, asyncio.TimeoutError):
                error_text = f"Timeout Error: {e}"
            elif isinstance(e, aiohttp.ClientConnectionError):
                error_text = f"Connection Error: {e}"
            elif isinstance(e, ClientError):
                error_text = str(e)  # 4xx 错误
            elif isinstance(e, aiohttp.ClientResponseError):
                error_text = f"API Request Error: Server Error ({e.status})"  # 5xx 错误
            else:
                error_text = f"API Request Error: {e}"
            
            await self._record_profile(
                normalize_end - normalize_start,
                request_end - request_start,
                0.0,
                time.perf_counter() - total_start,
            )
            return ToolResponse(text=json.dumps({"error": error_text}, ensure_ascii=False)), 0.0, {
                "hits": 0,
                "status": "api_error",
                "error": error_text,
            }
        
        except Exception as e:
            # unexpected error.
            request_end = time.perf_counter()
            error_text = f"Unexpected Error: {e}"
            await self._record_profile(
                normalize_end - normalize_start,
                request_end - request_start,
                0.0,
                time.perf_counter() - total_start,
            )
            return ToolResponse(text=json.dumps({"error": error_text}, ensure_ascii=False)), 0.0, {
                "hits": 0,
                "status": "api_error",
                "error": error_text,
            }

        parse_start = time.perf_counter()
        raw_results = response_json.get("result", [])
        hits_per_query = self._normalize_results(raw_results, len(query_list))
        trimmed = [hits_per_query[i][:k_list[i]] for i in range(len(query_list))]
        parse_end = time.perf_counter()
        total_end = time.perf_counter()
        await self._record_profile(
            normalize_end - normalize_start,
            request_end - request_start,
            parse_end - parse_start,
            total_end - total_start,
        )
        return ToolResponse(text=json.dumps(trimmed, ensure_ascii=False)), 0.0, {
            "hits": sum(len(h) for h in trimmed),
            "status": "success",
            "query_count": len(query_list),
        }

    @classmethod
    async def _record_profile(cls, normalize_s: float, request_s: float, parse_s: float, total_s: float) -> None:
        if cls._profile_lock is None:
            cls._profile_lock = asyncio.Lock()
        async with cls._profile_lock:
            cls._profile_count += 1
            cls._profile_sums["normalize"] += normalize_s
            cls._profile_sums["request"] += request_s
            cls._profile_sums["parse"] += parse_s
            cls._profile_sums["total"] += total_s
            if cls._profile_count >= cls._profile_target:
                averages = {
                    key: (value / cls._profile_count if cls._profile_count else 0.0)
                    for key, value in cls._profile_sums.items()
                }
                payload = {
                    "count": cls._profile_count,
                    "average_seconds": averages,
                }
                cls._profile_output_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

class ErrorToolCall(BaseTool):
    """Error tool call."""
    def __init__(self, config: dict, tool_schema: Optional[OpenAIFunctionToolSchema] = None):
        schema = tool_schema or self._build_schema()
        super().__init__(config, schema)
    
    def _build_schema(self) -> OpenAIFunctionToolSchema:
        return OpenAIFunctionToolSchema(
            type="function",
            function=OpenAIFunctionSchema(
                name="error_tool_call",
                description="It will be called when the tool call cannot be extracted.",
            ),
        )
    
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        return ToolResponse(text=json.dumps(parameters, ensure_ascii=False)), 0.0, {"hits": 0, "status": "error_tool_call"}



__all__ = ["LocalSearchTool", "LocalEmbeddingSearchTool"]

if __name__ == "__main__":
    tool = LocalEmbeddingSearchTool(config={
        "retrieval_service_url": "http://127.0.0.1:8765/search",
        "topk_default": 3,
        "timeout": 30,
        "return_scores": True,
    })
    result = asyncio.run(tool.execute(instance_id="test", parameters={"query_list": ["What is the capital of France?"]}))
    print(result)