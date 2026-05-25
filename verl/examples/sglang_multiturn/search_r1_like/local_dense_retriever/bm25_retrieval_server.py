#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 Search-R1 Contributors
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
#
# A lightweight BM25-only retrieval server.
# This keeps the corpus/index in memory and avoids per-step IO during rollout.

import argparse
import json
import warnings
from typing import Optional

import datasets
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


def load_corpus(corpus_path: str):
    return datasets.load_dataset("json", data_files=corpus_path, split="train", num_proc=4)


def _ensure_contents(doc: dict) -> dict:
    if "contents" in doc and isinstance(doc["contents"], str):
        return doc
    title = doc.get("title") or ""
    text = doc.get("text") or doc.get("passage") or doc.get("body") or ""
    if title or text:
        doc["contents"] = f"{title}\n{text}".strip()
    else:
        doc["contents"] = json.dumps(doc, ensure_ascii=False)
    return doc


class BM25Retriever:
    def __init__(self, index_path: str, corpus_path: str, topk: int):
        from pyserini.search.lucene import LuceneSearcher

        self.searcher = LuceneSearcher(index_path)
        self.topk = topk
        self.contain_doc = self.searcher.doc(0).raw() is not None
        self.corpus = None
        if not self.contain_doc and corpus_path:
            self.corpus = load_corpus(corpus_path)

    def _load_docs(self, doc_ids: list[int]) -> list[dict]:
        if self.contain_doc:
            results = []
            for doc_id in doc_ids:
                raw = self.searcher.doc(doc_id).raw()
                doc = json.loads(raw)
                results.append(_ensure_contents(doc))
            return results
        if self.corpus is None:
            return []
        results = [self.corpus[int(idx)] for idx in doc_ids]
        return [_ensure_contents(doc) for doc in results]

    def search(self, query: str, num: Optional[int] = None, return_score: bool = False):
        if num is None:
            num = self.topk
        hits = self.searcher.search(query, num)
        if len(hits) < 1:
            return ([], []) if return_score else []
        scores = [hit.score for hit in hits]
        if len(hits) < num:
            warnings.warn("Not enough documents retrieved!", stacklevel=2)
        else:
            hits = hits[:num]
            scores = scores[:num]
        doc_ids = [hit.docid for hit in hits]
        results = self._load_docs(doc_ids)
        return (results, scores) if return_score else results

    def batch_search(self, query_list: list[str], num: Optional[int] = None, return_score: bool = False):
        results = []
        scores = []
        for query in query_list:
            item_result, item_score = self.search(query, num, True)
            results.append(item_result)
            scores.append(item_score)
        return (results, scores) if return_score else results


class QueryRequest(BaseModel):
    queries: list[str]
    topk: Optional[int] = None
    return_scores: bool = False


app = FastAPI()
retriever: Optional[BM25Retriever] = None


@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    if not request.topk:
        request.topk = retriever.topk
    results, scores = retriever.batch_search(
        query_list=request.queries, num=request.topk, return_score=request.return_scores
    )
    resp = []
    for i, single_result in enumerate(results):
        if request.return_scores:
            combined = []
            for doc, score in zip(single_result, scores[i], strict=True):
                combined.append({"document": doc, "score": score})
            resp.append(combined)
        else:
            resp.append(single_result)
    return {"result": resp}


def parse_args():
    parser = argparse.ArgumentParser(description="Launch a BM25 retrieval server.")
    parser.add_argument("--index_path", type=str, required=True, help="BM25 Lucene index path.")
    parser.add_argument("--corpus_path", type=str, default="", help="Corpus JSONL path (optional).")
    parser.add_argument("--topk", type=int, default=3, help="Top-k documents to return.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host.")
    parser.add_argument("--port", type=int, default=8000, help="Server port.")
    return parser.parse_args()


def main():
    global retriever
    args = parse_args()
    retriever = BM25Retriever(index_path=args.index_path, corpus_path=args.corpus_path, topk=args.topk)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

