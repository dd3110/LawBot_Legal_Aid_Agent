# German Law Chatbot

A semantic search system for German constitutional law using vector embeddings and cross-referencing.

## Overview

This project converts German law documents into a searchable vector database with intelligent cross-referencing capabilities. Users can query the database using natural language to find relevant articles, sections, and related legal provisions.

## Architecture

### Components

1. **pdf_2_md.py** — PDF to Markdown Conversion
   - Converts PDF documents to markdown using Docling
   - Handles large/scanned PDFs with automatic chunking
   - Built-in error recovery on problematic pages
   - Configurable page ranges and processing strategies

2. **md2vdb.py** — Vector Database Builder
   - Parses markdown and extracts reference-aware chunks
   - Builds bidirectional cross-references between articles
   - Generates semantic embeddings using Ollama
   - Stores chunks with rich metadata in ChromaDB

3. **query_vdb.py** — Query Interface
   - Interactive command-line query tool
   - Semantic search with similarity scoring
   - Article retrieval and browsing
   - Formatted result display with cross-reference information

4. **Law_suggestion_agent.py** — Intelligent Legal Assistant (PydanticAI)
   - Agent-based interface powered by Qwen3 LLM
   - Semantic search integration with vector database
   - Structured tool use for database queries
   - Context-aware answers with article citations
   - Interactive chat and demo modes

## Requirements

- Python 3.9+
- Ollama (with Qwen3 model)
- Dependencies: `chromadb`, `requests`, `docling`, `pydantic-ai`

## Installation

```bash
# Install Python dependencies
pip install chromadb requests docling pydantic-ai

# Start Ollama service
ollama serve

# In another terminal, pull embedding and chat models
ollama pull mxbai-embed-large:latest
ollama pull qwen3
```

## Usage

### Step 1: Convert PDF to Markdown

```bash
python pdf_2_md.py
```

Converts `englisch_gg.pdf` to `Law2.md` with configurable chunking.

### Step 2: Build Vector Database

```bash
python md2vdb.py
```

Processes markdown, extracts chunks with references, and builds the vector database.

### Step 3: Query the Database

```bash
python query_vdb.py
```

Starts interactive query interface. Commands:
- Enter a question to search
- `list` — Show all articles
- `article Article_5` — Display specific article
- `help` — Show available commands
- `exit` — Quit

### Step 4: Chat with AI Agent (Recommended)

```bash
python Law_suggestion_agent.py
```

Interactive chat with intelligent agent powered by Qwen3. The agent:
- Understands natural language questions
- Searches vector database for relevant provisions
- Provides context-aware legal answers
- Cites relevant articles and sections

Or run demo queries:
```bash
python Law_suggestion_agent.py demo
```

## Example Queries

**Vector Database Search (query_vdb.py):**
- "What are the rights related to freedom of expression?"
- "Which articles protect property rights?"
- "What restrictions apply to freedom of movement?"

**AI Agent Chat (Law_suggestion_agent.py):**
- "Tell me about my fundamental rights as a citizen."
- "How is the executive branch structured in Germany?"
- "What are the conditions for amending the constitution?"
- "Explain the relationship between federal and state governments."

## Configuration

Edit constants at the top of each file to customize:

- `pdf_2_md.py`: Input PDF path, output path, chunk size
- `md2vdb.py`: Database path, embedding model, Ollama URL
- `query_vdb.py`: Database path

## Database Structure

The vector database stores chunks with metadata:

- **article_id** — Article identifier (e.g., Article_1, Article_5a)
- **section** — Subsection number if applicable
- **text** — Full text of the section
- **keywords** — Extracted semantic keywords
- **references** — Inbound and outbound article references
- **legal_category** — Classification (fundamental_right, family_law, etc.)

## Agent Architecture

The AI agent uses a multi-component architecture:

**Pydantic AI Framework:**
- Structured tool definitions for reproducible behavior
- Type-safe request/response handling
- Ollama backend integration

**Vector Database Integration:**
- Semantic search tool for finding relevant articles
- Context retrieval for informed responses
- Metadata-enriched results (keywords, references, citations)

**Language Model:**
- Qwen3 running locally via Ollama
- No data sent to external APIs
- Full privacy for sensitive legal queries

**Data Flow:**
```
User Question
    ↓
Agent (PydanticAI)
    ↓
Semantic Search Tool
    ↓
Vector Database (ChromaDB)
    ↓
Relevant Articles + Metadata
    ↓
LLM Context Building
    ↓
Structured Answer with Citations
```

## Performance & Troubleshooting

**Performance Notes:**
- Embedding generation is one-time cost during database build
- Agent response time depends on Qwen3 inference (typically 1-5 seconds)
- Increase Ollama context window for better reasoning
- Reduce chunk count in vector search for faster responses

**Common Issues:**

| Issue | Solution |
|-------|----------|
| "Vector DB not found" | Run `python md2vdb.py` to rebuild database |
| "Cannot connect to Ollama" | Run `ollama serve` in separate terminal |
| "Model 'qwen3' not found" | Run `ollama pull qwen3` |
| Slow responses | Reduce search results, check Ollama logs, or use faster model |
| Out of memory | Reduce Ollama context size or use smaller model |

## License

This project is provided as-is for educational and research purposes.
