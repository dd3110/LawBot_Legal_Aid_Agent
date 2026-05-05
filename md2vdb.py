"""German Law Chatbot: Vector Database Builder

This module provides tools to parse German law markdown documents and build
a semantic vector database with cross-referencing capabilities.

Main components:
  - GermanLawChunker: Parses markdown and extracts reference-aware chunks
  - VectorDBBuilder: Manages vector database creation and embedding
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import chromadb
import requests


class GermanLawChunker:
    """Parse German law markdown and extract reference-aware chunks."""

    ARTICLE_PATTERN = re.compile(
        r"^## Article (\d+)([a-z]?)\s*\[([^\]]+)\]?", re.MULTILINE
    )
    SECTION_PATTERN = re.compile(r"^- \((\d+)\)\s*(.+?)(?=\n- \(|\n##|$)", re.MULTILINE | re.DOTALL)
    REFERENCE_PATTERN = re.compile(
        r"(?:pursuant to|in accordance with|see also|of|to)\s+"
        r"(?:paragraph\s*\((\d+)\)\s+of\s+)?Article\s+(\d+)([a-z]?)"
        r"(?:\s*,?\s*(?:paragraphs?\s*\(([0-9]+(?:\s*-\s*[0-9]+)?)\))?)?",
        re.IGNORECASE,
    )

    def __init__(self, md_file: Path):
        self.md_file = md_file
        self.content = md_file.read_text(encoding="utf-8")
        self.chunks: list[dict[str, Any]] = []
        self.article_map: dict[str, int] = {}
        self._build_article_map()

    def _build_article_map(self) -> None:
        """Build a map of article IDs to enable inbound reference detection."""
        for match in self.ARTICLE_PATTERN.finditer(self.content):
            article_num = match.group(1)
            article_letter = match.group(2) or ""
            article_id = f"Article_{article_num}{article_letter}"
            self.article_map[article_id] = len(self.article_map)

    def _extract_references(self, text: str) -> list[dict[str, Any]]:
        """Extract all outbound references (Article X, Article Y(n), etc.) from text."""
        references = []
        for match in self.REFERENCE_PATTERN.finditer(text):
            para_num = match.group(1)
            article_num = match.group(2)
            article_letter = match.group(3) or ""
            section_range = match.group(4)

            article_id = f"Article_{article_num}{article_letter}"
            if article_id in self.article_map:
                ref_entry = {
                    "article": article_id,
                    "section": para_num or section_range,
                    "context": match.group(0).strip(),
                }
                references.append(ref_entry)

        return list({r["article"]: r for r in references}.values())

    def _parse_part_heading(self, text: str) -> tuple[str | None, str | None]:
        """Extract part number and title from a Part heading (e.g., '## I. Basic Rights')."""
        match = re.match(r"^##\s+([IVX]+(?:\.[a-z]+)?)\.\s+(.+?)$", text, re.MULTILINE)
        if match:
            return match.group(1), match.group(2)
        return None, None

    def chunk(self) -> list[dict[str, Any]]:
        """Parse the markdown and create reference-aware chunks."""
        self.chunks = []
        current_part = None
        current_part_title = None
        article_counter = 0

        lines = self.content.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("## ") and not line.startswith("## Article"):
                part, part_title = self._parse_part_heading(line)
                if part:
                    current_part = part
                    current_part_title = part_title
                continue

            article_match = self.ARTICLE_PATTERN.match(line)
            if not article_match:
                continue

            article_num = article_match.group(1)
            article_letter = article_match.group(2) or ""
            article_title = article_match.group(3) or ""
            article_id = f"Article_{article_num}{article_letter}"
            article_counter += 1

            start_line = i + 1
            end_line = i + 1
            for j in range(i + 1, len(lines)):
                if lines[j].startswith("## "):
                    end_line = j
                    break
                end_line = j + 1

            article_text = "\n".join(lines[start_line:end_line]).strip()
            sections = self._extract_sections(article_id, article_text, article_title, current_part, current_part_title)
            self.chunks.extend(sections)

        return self.chunks

    def _extract_sections(
        self,
        article_id: str,
        article_text: str,
        article_title: str,
        part: str | None,
        part_title: str | None,
    ) -> list[dict[str, Any]]:
        """Extract individual sections from an article."""
        sections = []
        section_matches = list(self.SECTION_PATTERN.finditer(article_text))

        if not section_matches:
            if article_text.strip():
                chunk = self._create_chunk(
                    article_id=article_id,
                    article_title=article_title,
                    section=None,
                    text=article_text.strip(),
                    part=part,
                    part_title=part_title,
                )
                sections.append(chunk)
            return sections

        for match in section_matches:
            section_num = match.group(1)
            section_text = match.group(2).strip()
            chunk = self._create_chunk(
                article_id=article_id,
                article_title=article_title,
                section=f"({section_num})",
                text=section_text,
                part=part,
                part_title=part_title,
            )
            sections.append(chunk)

        return sections

    def _create_chunk(
        self,
        article_id: str,
        article_title: str,
        section: str | None,
        text: str,
        part: str | None,
        part_title: str | None,
    ) -> dict[str, Any]:
        """Create a single reference-aware chunk with metadata."""
        outbound_refs = self._extract_references(text)
        word_count = len(text.split())
        token_count = int(word_count * 1.33)

        section_full_id = f"{article_id}_{section.strip('()')}" if section else article_id

        chunk = {
            "article_id": article_id,
            "article_number": int(re.search(r"\d+", article_id).group()),
            "article_letter": article_id[-1] if article_id[-1].isalpha() else "",
            "article_title": article_title,
            "part": part or "Preamble",
            "part_title": part_title or "Preamble",
            "section": section,
            "section_full_id": section_full_id,
            "is_subsection": section is not None,
            "text": text,
            "word_count": word_count,
            "token_count": token_count,
            "references_outbound": outbound_refs,
            "references_inbound": [],
            "restricting_keywords": self._extract_restricting_keywords(text),
            "basic_right_type": self._extract_basic_right_type(article_id),
            "keywords": self._extract_keywords(text),
            "entity_types": self._extract_entity_types(text),
            "legal_category": self._extract_legal_category(article_id, article_title),
            "sort_key": self._create_sort_key(article_id, section),
        }

        return chunk

    @staticmethod
    def _extract_restricting_keywords(text: str) -> list[str]:
        """Extract keywords indicating legal restrictions."""
        keywords = []
        patterns = [
            r"may be restricted",
            r"restricted\b",
            r"regulated by",
            r"pursuant to",
            r"only if",
            r"shall not",
            r"prohibition",
            r"forbidden",
        ]
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                keywords.append(pattern.lower().replace(r"\b", "").strip("()"))
        return keywords

    @staticmethod
    def _extract_basic_right_type(article_id: str) -> str | None:
        """Determine basic right type from article ID (Part I)."""
        article_num = int(re.search(r"\d+", article_id).group())
        if 1 <= article_num <= 19:
            return "basic_right"
        return None

    @staticmethod
    def _extract_keywords(text: str) -> list[str]:
        """Extract key technical terms for semantic search."""
        keywords_map = {
            "human.*dignity": "human dignity",
            "freedom.*of.*expression": "freedom of expression",
            "freedom.*of.*religion": "freedom of religion",
            "freedom.*of.*movement": "freedom of movement",
            "property": "property rights",
            "marriage.*family": "marriage and family",
            "association": "freedom of association",
            "asylum": "right of asylum",
            "citizenship": "citizenship",
            "military.*service": "military service",
            "education": "education",
            "worker": "worker rights",
            "equal.*right": "equality",
        }
        found_keywords = []
        for pattern, label in keywords_map.items():
            if re.search(pattern, text, re.IGNORECASE):
                found_keywords.append(label)
        return found_keywords if found_keywords else ["general"]

    @staticmethod
    def _extract_entity_types(text: str) -> list[str]:
        """Classify chunk by entity type (right, restriction, procedure, etc.)."""
        types = []
        if re.search(r"shall (have the )?right", text, re.IGNORECASE):
            types.append("right")
        if re.search(r"may be restricted|restricted|prohibited", text, re.IGNORECASE):
            types.append("limitation")
        if re.search(r"pursuant to|by.*law|regulation", text, re.IGNORECASE):
            types.append("procedure")
        return types if types else ["general"]

    @staticmethod
    def _extract_legal_category(article_id: str, article_title: str) -> list[str]:
        """Extract legal categories based on article title."""
        title_lower = article_title.lower()
        categories = []
        if "freedom" in title_lower or "right" in title_lower:
            categories.append("fundamental_right")
        if "family" in title_lower or "marriage" in title_lower:
            categories.append("family_law")
        if "property" in title_lower:
            categories.append("property_law")
        if "citizenship" in title_lower:
            categories.append("citizenship")
        if "federation" in title_lower or "land" in title_lower:
            categories.append("constitutional_structure")
        if "political" in title_lower or "party" in title_lower:
            categories.append("political_system")
        return categories if categories else ["general"]

    @staticmethod
    def _create_sort_key(article_id: str, section: str | None) -> str:
        """Create a sortable key for document ordering."""
        match = re.match(r"Article_(\d+)([a-z]?)", article_id)
        article_num = int(match.group(1))
        article_letter = ord(match.group(2)) - ord("a") + 1 if match.group(2) else 0
        section_num = int(section.strip("()")) if section else 0
        return f"I.{article_num:03d}{article_letter:02d}.{section_num:03d}"


class VectorDBBuilder:
    """Build and manage the vector database."""

    def __init__(self, db_path: Path = None, embedding_model: str = "mxbai-embed-large:latest", ollama_url: str = "http://localhost:11434"):
        self.db_path = db_path or Path("./law_vector_db")
        self.embedding_model = embedding_model
        self.ollama_url = ollama_url
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        self.collection = self.client.get_or_create_collection(
            name="german_law",
            metadata={"hnsw:space": "cosine"},
        )

    def _check_ollama_health(self) -> None:
        """Verify Ollama service is running and embedding model is loaded.
        
        Raises:
            ConnectionError: If Ollama service is unreachable
            ValueError: If embedding model is not available
        """
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            response.raise_for_status()
            models = [m["name"] for m in response.json().get("models", [])]
            if self.embedding_model not in models:
                raise ValueError(
                    f"Model '{self.embedding_model}' not found in Ollama. "
                    f"Available models: {models}. "
                    f"Run: ollama pull {self.embedding_model}"
                )
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.ollama_url}. "
                f"Make sure Ollama is running: ollama serve"
            )

    def _get_embedding(self, text: str) -> list[float]:
        """Generate semantic embedding for text using Ollama.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of floats representing the embedding vector
            
        Raises:
            TimeoutError: If Ollama request exceeds timeout
            RuntimeError: If embedding request fails
        """
        try:
            response = requests.post(
                f"{self.ollama_url}/api/embeddings",
                json={"model": self.embedding_model, "prompt": text},
                timeout=300,
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except requests.exceptions.Timeout:
            raise TimeoutError(
                "Ollama request timed out after 300s. "
                "The embedding model may be slow or unresponsive."
            )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama embedding request failed: {e}")

    def add_chunks(self, chunks: list[dict[str, Any]], clear_existing: bool = True) -> None:
        """Add reference-aware chunks to the vector database.
        
        Processes chunks through deduplication, embedding generation, and storage.
        
        Args:
            chunks: List of chunk dictionaries with metadata and text
            clear_existing: If True, clear existing collection before adding
        """
        print("\n  → Verifying Ollama connection...")
        self._check_ollama_health()
        print("    ✓ Ollama service is available")

        if clear_existing:
            print("  → Clearing existing collection...")
            try:
                self.client.delete_collection(name="german_law")
                self.collection = self.client.get_or_create_collection(
                    name="german_law",
                    metadata={"hnsw:space": "cosine"},
                )
                print("    ✓ Collection cleared")
            except Exception:
                pass

        # Deduplicate chunks by unique identifier (section_full_id)
        seen_ids = set()
        unique_chunks = []
        for chunk in chunks:
            chunk_id = chunk["section_full_id"]
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                unique_chunks.append(chunk)
            else:
                print(f"    ⊘ Skipping duplicate: {chunk_id}")

        print(f"  → Processing {len(unique_chunks)} unique chunks\n")

        ids = []
        embeddings = []
        metadatas = []
        documents = []

        for i, chunk in enumerate(unique_chunks):
            chunk_id = chunk["section_full_id"]
            text = chunk["text"]
            try:
                embedding = self._get_embedding(text)
            except (TimeoutError, RuntimeError) as e:
                print(f"  ⚠️  Skipping {chunk_id}: {e}")
                continue
            if (i + 1) % 10 == 0 or i == len(unique_chunks) - 1:
                print(f"    [{i+1}/{len(unique_chunks)}] Processed: {chunk_id}")

            metadata = {
                "article_id": chunk["article_id"],
                "article_title": chunk["article_title"],
                "section": chunk["section"] or "main",
                "part": chunk["part"],
                "part_title": chunk["part_title"],
                "word_count": chunk["word_count"],
                "references_outbound": json.dumps(chunk["references_outbound"]),
                "keywords": json.dumps(chunk["keywords"]),
                "entity_types": json.dumps(chunk["entity_types"]),
                "legal_category": json.dumps(chunk["legal_category"]),
                "restricting_keywords": json.dumps(chunk["restricting_keywords"]),
            }

            ids.append(chunk_id)
            embeddings.append(embedding)
            metadatas.append(metadata)
            documents.append(text)

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )

        print(f"  ✓ Successfully added {len(ids)} chunks to database")

    def query(self, query_text: str, n_results: int = 5) -> list[dict[str, Any]]:
        """Search vector database for semantically similar chunks.
        
        Args:
            query_text: Search query string
            n_results: Maximum number of results to return
            
        Returns:
            List of result dictionaries with metadata and similarity scores
        """
        results = self.collection.query(query_texts=[query_text], n_results=n_results)

        output = []
        for i, doc_id in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            output.append({
                "id": doc_id,
                "article": metadata["article_id"],
                "title": metadata["article_title"],
                "section": metadata["section"],
                "text": results["documents"][0][i],
                "similarity_score": 1 - distance,
                "references": json.loads(metadata["references_outbound"]),
                "keywords": json.loads(metadata["keywords"]),
            })
        return output


def main() -> int:
    """Build vector database from German law markdown document.
    
    Workflow:
      1. Parse markdown and extract reference-aware chunks
      2. Build bidirectional reference links between articles
      3. Generate embeddings and populate vector database
      4. Demonstrate with sample queries
    
    Returns:
        0 on success, 1 on error
    """
    md_file = Path(r"D:\German_law_Chatbot\Law2.md")
    db_path = Path(r"D:\German_law_Chatbot\law_vector_db")

    if not md_file.exists():
        raise FileNotFoundError(f"Markdown file not found: {md_file}")

    print("\n" + "="*60)
    print("German Law Vector Database Builder")
    print("="*60)
    
    print("\n[1/4] Parsing German law markdown...")
    chunker = GermanLawChunker(md_file)
    chunks = chunker.chunk()
    print(f"      ✓ Extracted {len(chunks)} chunks from {len(chunker.article_map)} articles")

    print("\n[2/4] Building bidirectional references...")
    outbound_to_chunks = {}
    for chunk in chunks:
        for ref in chunk["references_outbound"]:
            article_ref = ref["article"]
            if article_ref not in outbound_to_chunks:
                outbound_to_chunks[article_ref] = []
            outbound_to_chunks[article_ref].append(chunk["section_full_id"])

    for chunk in chunks:
        article_id = chunk["article_id"]
        if article_id in outbound_to_chunks:
            chunk["references_inbound"] = [
                {"article": chunk_id.split("_")[1], "refers_from": chunk_id}
                for chunk_id in outbound_to_chunks[article_id]
            ]

    print("      ✓ Cross-reference mapping complete")

    print("\n[3/4] Building vector database...")
    db_builder = VectorDBBuilder(db_path)
    try:
        db_builder.add_chunks(chunks)
    except (ConnectionError, ValueError) as e:
        print(f"\n      ERROR: {e}")
        return 1

    print("\n[4/4] Validating with sample queries...")
    sample_queries = [
        "What are the rights related to freedom of expression?",
        "What articles restrict the right to work?",
        "Which articles reference Article 5?",
    ]
    for i, query in enumerate(sample_queries, 1):
        print(f"\n      Query {i}: {query}")
        results = db_builder.query(query, n_results=2)
        for result in results:
            preview = result['text'][:60].replace('\n', ' ')
            print(f"        → {result['article']} {result['section']}: {preview}...")

    print("\n" + "="*60)
    print(f"✓ Vector database ready at: {db_path}")
    print("="*60 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())