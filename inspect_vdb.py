"""Vector Database Inspector

Provides detailed inspection of the Chroma vector database structure, contents,
and metadata to understand what's stored and how data is organized.

Usage:
    python inspect_vdb.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chromadb


class VectorDBInspector:
    """Inspect and visualize vector database contents and structure."""

    def __init__(self, db_path: Path = None):
        """Initialize inspector.
        
        Args:
            db_path: Path to vector database directory
        """
        self.db_path = db_path or Path(r"D:\German_law_Chatbot\law_vector_db")
        if not self.db_path.exists():
            raise FileNotFoundError(f"Vector DB not found at {self.db_path}")

        self.client = chromadb.PersistentClient(path=str(self.db_path))
        self.collection = self.client.get_collection(name="german_law")

    def print_separator(self, char: str = "=", width: int = 70) -> None:
        """Print a separator line."""
        print(char * width)

    def print_subsection(self, title: str, char: str = "-", width: int = 70) -> None:
        """Print a subsection header."""
        print(f"\n{char * width}")
        print(f"  {title}")
        print(char * width)

    def get_database_stats(self) -> dict[str, Any]:
        """Get basic statistics about the database.
        
        Returns:
            Dictionary with count, articles, and metadata field info
        """
        results = self.collection.get()
        
        total_chunks = len(results["ids"])
        
        # Extract unique articles
        articles = set()
        for metadata in results["metadatas"]:
            articles.add(metadata["article_id"])
        
        # Extract all metadata keys
        metadata_keys = set()
        for metadata in results["metadatas"]:
            metadata_keys.update(metadata.keys())
        
        return {
            "total_chunks": total_chunks,
            "unique_articles": len(articles),
            "articles": sorted(articles),
            "metadata_keys": sorted(metadata_keys),
        }

    def display_database_overview(self) -> None:
        """Display high-level database overview."""
        print("\n")
        self.print_separator()
        print("VECTOR DATABASE OVERVIEW")
        self.print_separator()
        
        stats = self.get_database_stats()
        
        print(f"\n  Total Chunks:        {stats['total_chunks']}")
        print(f"  Unique Articles:     {stats['unique_articles']}")
        print(f"  Metadata Fields:     {len(stats['metadata_keys'])}")

    def display_articles(self) -> None:
        """Display list of all articles in the database."""
        self.print_subsection("ARTICLES IN DATABASE")
        
        stats = self.get_database_stats()
        articles = stats["articles"]
        
        # Group articles by number
        articles_by_num = {}
        for article in articles:
            # Extract number from Article_1, Article_5a, etc.
            num_part = article.replace("Article_", "")
            base_num = "".join(filter(str.isdigit, num_part))
            if base_num not in articles_by_num:
                articles_by_num[base_num] = []
            articles_by_num[base_num].append(article)
        
        # Display grouped
        for num in sorted(articles_by_num.keys(), key=int):
            article_variants = articles_by_num[num]
            print(f"  Article {num:3s}:  {', '.join(article_variants)}")

    def display_metadata_schema(self) -> None:
        """Display the metadata schema with sample values."""
        self.print_subsection("METADATA SCHEMA")
        
        results = self.collection.get(limit=1)
        
        if not results["metadatas"]:
            print("  No data available")
            return
        
        sample_metadata = results["metadatas"][0]
        
        print(f"\n  Field Name                    Type            Sample Value")
        print("  " + "-" * 66)
        
        for key, value in sorted(sample_metadata.items()):
            # Determine value type
            if isinstance(value, str):
                if len(value) > 40:
                    if value.startswith("["):
                        val_display = f"{value[:37]}...]"
                    else:
                        val_display = f"{value[:37]}..."
                else:
                    val_display = value
                val_type = "string"
            else:
                val_display = str(value)[:40]
                val_type = type(value).__name__
            
            print(f"  {key:<30} {val_type:<15} {val_display}")

    def display_sample_chunk(self, chunk_index: int = 0) -> None:
        """Display a complete sample chunk with all data.
        
        Args:
            chunk_index: Index of chunk to display (0-based)
        """
        self.print_subsection("SAMPLE CHUNK (FULL DETAILS)")
        
        results = self.collection.get(limit=chunk_index + 1)
        
        if not results["ids"] or len(results["ids"]) <= chunk_index:
            print("  Chunk not found")
            return
        
        chunk_id = results["ids"][chunk_index]
        metadata = results["metadatas"][chunk_index]
        document = results["documents"][chunk_index]
        
        print(f"\n  Chunk ID: {chunk_id}")
        
        print("\n  METADATA:")
        for key, value in sorted(metadata.items()):
            # Try to parse JSON fields for better display
            if isinstance(value, str) and value.startswith("["):
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, list):
                        if len(parsed) == 0:
                            print(f"    {key}: []")
                        elif len(parsed) == 1:
                            print(f"    {key}: {parsed}")
                        else:
                            print(f"    {key}:")
                            for item in parsed:
                                print(f"      - {item}")
                        continue
                except json.JSONDecodeError:
                    pass
            
            # Regular display
            if len(str(value)) > 60:
                print(f"    {key}: {str(value)[:60]}...")
            else:
                print(f"    {key}: {value}")
        
        print("\n  TEXT CONTENT:")
        text_lines = document.split("\n")
        for i, line in enumerate(text_lines[:10], 1):
            if line.strip():
                print(f"    {line[:65]}")
        
        if len(text_lines) > 10:
            print(f"    ... ({len(text_lines) - 10} more lines)")

    def display_chunk_distribution(self) -> None:
        """Show distribution of chunks across articles."""
        self.print_subsection("CHUNK DISTRIBUTION BY ARTICLE")
        
        results = self.collection.get()
        
        # Count chunks per article
        article_counts = {}
        for metadata in results["metadatas"]:
            article_id = metadata["article_id"]
            article_counts[article_id] = article_counts.get(article_id, 0) + 1
        
        # Display with bar chart
        max_count = max(article_counts.values()) if article_counts else 0
        bar_width = 40
        
        print()
        for article in sorted(article_counts.keys()):
            count = article_counts[article]
            bar_length = int((count / max_count) * bar_width) if max_count > 0 else 0
            bar = "█" * bar_length
            print(f"  {article:<15} {count:>3} chunks  {bar}")

    def display_metadata_statistics(self) -> None:
        """Show statistics about metadata fields."""
        self.print_subsection("METADATA FIELD STATISTICS")
        
        results = self.collection.get()
        
        # Collect all values for each key
        field_values = {}
        for metadata in results["metadatas"]:
            for key, value in metadata.items():
                if key not in field_values:
                    field_values[key] = []
                field_values[key].append(value)
        
        print()
        for key in sorted(field_values.keys()):
            values = field_values[key]
            
            # For JSON fields, try to parse and analyze
            if values and isinstance(values[0], str) and values[0].startswith("["):
                try:
                    parsed_values = [json.loads(v) for v in values if v]
                    if parsed_values and isinstance(parsed_values[0], list):
                        total_items = sum(len(v) for v in parsed_values)
                        avg_items = total_items / len(parsed_values) if parsed_values else 0
                        print(f"  {key:<30} (list): avg {avg_items:.1f} items/chunk, total {total_items}")
                        continue
                except json.JSONDecodeError:
                    pass
            
            # String fields
            non_empty = sum(1 for v in values if v and str(v).strip())
            print(f"  {key:<30} {non_empty}/{len(values)} non-empty values")

    def display_parts_distribution(self) -> None:
        """Show how chunks are distributed across different parts."""
        self.print_subsection("DISTRIBUTION BY PART")
        
        results = self.collection.get()
        
        # Count chunks per part
        part_counts = {}
        for metadata in results["metadatas"]:
            part = metadata.get("part", "unknown")
            part_counts[part] = part_counts.get(part, 0) + 1
        
        print()
        total = sum(part_counts.values())
        for part in sorted(part_counts.keys()):
            count = part_counts[part]
            percentage = (count / total * 100) if total > 0 else 0
            print(f"  {part:<25} {count:>4} chunks  ({percentage:>5.1f}%)")

    def inspect_all(self) -> None:
        """Run complete inspection and display all information."""
        self.display_database_overview()
        self.display_articles()
        self.display_metadata_schema()
        self.display_metadata_statistics()
        self.display_parts_distribution()
        self.display_chunk_distribution()
        self.display_sample_chunk()
        
        self.print_separator()
        print("INSPECTION COMPLETE")
        self.print_separator()
        print()


def main() -> int:
    """Run the database inspector.
    
    Returns:
        0 on success, 1 on error
    """
    try:
        inspector = VectorDBInspector()
        inspector.inspect_all()
        return 0
    except FileNotFoundError as e:
        print(f"\nError: {e}\n")
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {e}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
