"""German Law Vector Database Query Interface

Provides interactive and programmatic access to the German law vector database.
Supports semantic search, article retrieval, and formatted result display.

Example:
    db = LawVectorDBQuery()
    db.query("What articles guarantee freedom of expression?")
    db.interactive_query()  # Start interactive mode
"""
from __future__ import annotations

import json
from pathlib import Path

import chromadb


class LawVectorDBQuery:
    """Query interface for German law vector database.
    
    Provides methods to search the vector database, retrieve specific articles,
    and display formatted results with metadata and cross-references.
    """

    def __init__(self, db_path: Path = None):
        """Initialize database connection.
        
        Args:
            db_path: Path to vector database directory. Defaults to standard location.
            
        Raises:
            FileNotFoundError: If database directory doesn't exist
        """
        self.db_path = db_path or Path(r"D:\German_law_Chatbot\law_vector_db")
        if not self.db_path.exists():
            raise FileNotFoundError(f"Vector DB not found at {self.db_path}")
        
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        self.collection = self.client.get_collection(name="german_law")
        count = self.collection.count()
        print(f"\n  Database loaded: {count} chunks available\n")

    def query(self, query_text: str, n_results: int = 5) -> None:
        """Search database for semantically similar articles.
        
        Displays formatted results including similarity scores, article metadata,
        keywords, and cross-references.
        
        Args:
            query_text: Natural language search query
            n_results: Maximum number of results to return
        """
        print(f"  Search: {query_text}\n")
        
        results = self.collection.query(query_texts=[query_text], n_results=n_results)
        
        if not results["ids"] or not results["ids"][0]:
            print("  No results found.\n")
            return
        
        for i, doc_id in enumerate(results["ids"][0], 1):
            metadata = results["metadatas"][0][i-1]
            distance = results["distances"][0][i-1]
            similarity = 1 - distance
            text = results["documents"][0][i-1]
            
            print("  " + "─" * 40)
            print(f"  Result {i} | Similarity: {similarity:.1%}")
            print("  " + "─" * 40)
            print(f"  Article: {metadata['article_id']} — {metadata['article_title']}")
            print(f"  Section: {metadata['section']}")
            print(f"  Part: {metadata['part_title']}\n")
            
            print(f"  {text}\n")
            
            keywords = json.loads(metadata.get("keywords", "[]"))
            references = json.loads(metadata.get("references_outbound", "[]"))
            
            if keywords:
                print(f"  Keywords: {', '.join(keywords)}")
            
            if references:
                ref_articles = [r['article'] for r in references]
                print(f"  References: {', '.join(ref_articles)}")
            
            print()

    def list_articles(self) -> None:
        """Display all articles available in the database."""
        print("\n  Available Articles:")
        print("  " + "─" * 40)
        
        results = self.collection.get()
        articles = set()
        
        for metadata in results["metadatas"]:
            article = metadata["article_id"]
            articles.add(article)
        
        for article in sorted(articles):
            print(f"  → {article}")
        
        print(f"\n  Total: {len(articles)} articles\n")

    def get_article(self, article_id: str) -> None:
        """Retrieve and display all sections of a specific article.
        
        Args:
            article_id: Article identifier (e.g., 'Article_1')
        """
        print(f"\n  Article {article_id}")
        print("  " + "─" * 40 + "\n")
        
        results = self.collection.get(
            where={"article_id": article_id}
        )
        
        if not results["ids"]:
            print(f"  Article {article_id} not found.\n")
            return
        
        for i, doc_id in enumerate(results["ids"], 1):
            metadata = results["metadatas"][i-1]
            text = results["documents"][i-1]
            
            section = metadata.get("section", "main")
            print("  " + "─" * 40)
            print(f"  Section: {section}")
            print("  " + "─" * 40)
            print(f"  {text}\n")
            
            references = json.loads(metadata.get("references_outbound", "[]"))
            if references:
                ref_articles = [r['article'] for r in references]
                print(f"  References to: {', '.join(ref_articles)}\n")

    def interactive_query(self) -> None:
        """Start interactive query loop.
        
        Supported commands:
          - Text input: Search database
          - 'list': Show all articles
          - 'article <id>': Retrieve specific article
          - 'help': Show available commands
          - 'exit': Quit
        """
        print("\n  Type your question or command ('help' for options, 'exit' to quit)\n")
        
        while True:
            query = input("  > ").strip()
            
            if not query:
                continue
            elif query.lower() == "exit":
                print("\n  Goodbye!\n")
                break
            elif query.lower() in ("list", "help"):
                if query.lower() == "help":
                    print("\n  Commands:")
                    print("    list            — Show all available articles")
                    print("    article <id>    — Display specific article (e.g., article Article_1)")
                    print("    exit            — Quit\n")
                else:
                    self.list_articles()
            elif query.lower().startswith("article "):
                article_id = query[8:].strip()
                self.get_article(article_id)
            elif query:
                self.query(query, n_results=3)
            print()


def main() -> int:
    """Start the interactive query interface.
    
    Displays welcome message and command help, then enters interactive loop.
    
    Returns:
        0 on clean exit
    """
    db = LawVectorDBQuery()
    
    print("\n" + "="*50)
    print("German Law Database Query Tool")
    print("="*50)
    
    print("\nCommands:")
    print("  • Enter a question to search the database")
    print("  • Type 'list' to see all available articles")
    print("  • Type 'article Article_5' to view a specific article")
    print("  • Type 'help' for more options")
    print("  • Type 'exit' to quit")
    
    db.interactive_query()
    
    print("="*50 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
