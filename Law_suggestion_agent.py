"""German Law Suggestion Agent

An intelligent agent powered by Pydantic AI and Ollama Qwen3 that answers
questions about German constitutional law using semantic search in a vector database.

The agent retrieves relevant articles and sections from the vector database and
provides context-aware answers with citations to the relevant legal provisions.

Requirements:
    - Ollama running with Qwen3 model: ollama pull qwen3
    - Vector database built from md2vdb.py
    - PydanticAI library installed

Usage:
    agent = GermanLawAgent()
    answer = agent.query("What are my rights to freedom of expression?")
    print(answer)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import requests
import chromadb
from pydantic import BaseModel


class SearchResult(BaseModel):
    article_id: str
    article_title: str
    section: str | None
    text: str
    similarity: float
    keywords: list[str] = []
    references: list[str] = []


class DatabaseContext(BaseModel):
    db_path: Path
    collection: Any = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, db_path: Path | None = None, **data):
        super().__init__(**data)
        if db_path is None:
            db_path = Path(r"D:\German_law_Chatbot\law_vector_db")
        self.db_path = db_path
        if not self.db_path.exists():
            raise FileNotFoundError(f"Vector DB not found at {self.db_path}")
        client = chromadb.PersistentClient(path=str(self.db_path))
        self.collection = client.get_collection(name="german_law")


class OllamaClient:
    """Simple Ollama HTTP client for generation calls."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")

    def generate(self, model: str, prompt: str, temperature: float = 0.0, max_tokens: int | None = None) -> str:
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        url = f"{self.base_url}/api/generate"
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        # Ollama returns 'text' or 'result' depending on version; be flexible
        if isinstance(data, dict):
            return data.get("text") or data.get("result") or json.dumps(data)
        return str(data)


class GermanLawAgent:
    """Agent that combines vector search with Ollama LLM generation.

    Use `model_name` and `generation_params` to control the LLM.
    """

    def __init__(self, db_path: Path | None = None, ollama_url: str = "http://localhost:11434", model_name: str = "qwen3"):
        self.db = DatabaseContext(db_path=db_path)
        self.ollama = OllamaClient(base_url=ollama_url)
        self.model_name = model_name
        # default generation params
        self.generation_params = {
            "temperature": 0.0,
            "max_tokens": 512,
        }

    def set_model(self, model_name: str) -> None:
        self.model_name = model_name

    def set_generation_params(self, temperature: float | None = None, max_tokens: int | None = None) -> None:
        if temperature is not None:
            self.generation_params["temperature"] = temperature
        if max_tokens is not None:
            self.generation_params["max_tokens"] = max_tokens

    def _search(self, query: str, n_results: int = 3) -> List[SearchResult]:
        results = self.db.collection.query(query_texts=[query], n_results=n_results)
        out: List[SearchResult] = []
        if not results.get("ids") or not results["ids"][0]:
            return out

        for i, doc_id in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            text = results["documents"][0][i]
            similarity = 1 - distance
            # parse json fields if present
            keywords = []
            references = []
            try:
                keywords = json.loads(metadata.get("keywords", "[]"))
            except Exception:
                keywords = []
            try:
                refs = json.loads(metadata.get("references_outbound", "[]"))
                references = [r.get("article") for r in refs if isinstance(r, dict)]
            except Exception:
                references = []

            out.append(SearchResult(
                article_id=metadata.get("article_id"),
                article_title=metadata.get("article_title"),
                section=metadata.get("section"),
                text=text,
                similarity=round(similarity, 3),
                keywords=keywords,
                references=references,
            ))
        return out

    def _build_context(self, results: List[SearchResult]) -> str:
        parts: List[str] = []
        for r in results:
            header = f"{r.article_id} {r.section or ''} — {r.article_title}"
            parts.append(header)
            parts.append(r.text)
            parts.append("--")
        return "\n".join(parts)

    def answer(self, question: str, n_results: int = 3) -> str:
        """Run retrieval + generation to answer question synchronously."""
        results = self._search(question, n_results=n_results)
        context = self._build_context(results)

        system_prompt = (
            "You are an expert legal assistant for German constitutional law. "
            "Use the provided context excerpts (citations) to answer the user's question, and cite articles. "
            "If context is insufficient, say so. Keep answers concise and factual."
        )

        prompt = (
            f"SYSTEM:\n{system_prompt}\n\nCONTEXT:\n{context}\n\nUSER QUESTION:\n{question}\n\nANSWERMODE: Provide a clear answer followed by citations (Article numbers)."
        )

        resp = self.ollama.generate(
            model=self.model_name,
            prompt=prompt,
            temperature=self.generation_params.get("temperature", 0.0),
            max_tokens=self.generation_params.get("max_tokens", None),
        )
        return resp

    def interactive_chat(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", help="Ollama model name", default=self.model_name)
        parser.add_argument("--temp", type=float, help="Temperature", default=self.generation_params["temperature"])
        parser.add_argument("--max-tokens", type=int, help="Max tokens", default=self.generation_params["max_tokens"])
        args, _ = parser.parse_known_args(sys.argv[1:])
        self.set_model(args.model)
        self.set_generation_params(temperature=args.temp, max_tokens=args.max_tokens)

        print("\n" + "=" * 60)
        print("German Law Assistant (Ollama + Vector DB)")
        print("=" * 60)
        print("Type 'exit' to quit.\n")

        while True:
            q = input("Your question: ").strip()
            if not q:
                continue
            if q.lower() in ("exit", "quit"):
                print("Goodbye!")
                break
            try:
                print("\n[Thinking...]")
                ans = self.answer(q, n_results=3)
                print(f"\nAnswer:\n{ans}\n")
            except Exception as e:
                print(f"Error during answer: {e}")


def demo(mode_args: dict | None = None) -> None:
    print("Running demo queries...")
    agent = GermanLawAgent()
    if mode_args:
        agent.set_model(mode_args.get("model", agent.model_name))
        agent.set_generation_params(temperature=mode_args.get("temperature"), max_tokens=mode_args.get("max_tokens"))

    questions = [
        "What rights does Article 5 provide regarding freedom of expression?",
        "Summarize the fundamental rights in Part I.",
    ]

    for q in questions:
        print("\n" + "-" * 60)
        print(f"Q: {q}")
        try:
            ans = agent.answer(q, n_results=3)
            print(f"A: {ans}\n")
        except Exception as e:
            print(f"Error: {e}\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("demo", nargs="?", help="Run demo", default=None)
    parser.add_argument("--model", help="Ollama model name", default="qwen3")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    if args.demo is not None:
        demo({"model": args.model, "temperature": args.temperature, "max_tokens": args.max_tokens})
        return 0

    try:
        agent = GermanLawAgent(model_name=args.model)
        agent.set_generation_params(temperature=args.temperature, max_tokens=args.max_tokens)
        agent.interactive_chat()
        return 0
    except FileNotFoundError as e:
        print(f"Error: {e}\nEnsure vector DB exists and Ollama is running.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
