"""PDF to Markdown Converter for Large Documents

Converts PDF documents to Markdown format using Docling, with built-in resilience
for problematic pages. Handles memory issues on scanned/large PDFs through chunked
processing and per-page fallback strategies.

Features:
  - Automatic chunk-based processing for stability
  - Per-page retry on chunk failures
  - Comprehensive error logging
  - Page range support for selective conversion

Example:
    from pdf_2_md import convert_pdf_to_markdown
    output = convert_pdf_to_markdown(
        pdf_path=Path("input.pdf"),
        output_path=Path("output.md"),
        chunk_size=5
    )
"""
from __future__ import annotations

import logging
from pathlib import Path

from docling.document_converter import DocumentConverter


LOGGER = logging.getLogger("pdf_2_md")
INPUT_PDF_PATH = Path(r"D:\German_law_Chatbot\englisch_gg.pdf")
OUTPUT_MD_PATH = Path(r"D:\German_law_Chatbot\Law2.md")
CHUNK_SIZE = 5
PAGE_START = 1
PAGE_END: int | None = None


def convert_page_range(converter: DocumentConverter, pdf_path: Path, start_page: int, end_page: int) -> tuple[str, list[object]]:
    """Convert a page range to markdown.
    
    Args:
        converter: Docling document converter instance
        pdf_path: Path to PDF file
        start_page: Starting page number (1-indexed)
        end_page: Ending page number (inclusive)
        
    Returns:
        Tuple of (markdown_text, error_list)
    """
    result = converter.convert(pdf_path, raises_on_error=False, page_range=(start_page, end_page))
    markdown = ""
    if result.document is not None:
        markdown = result.document.export_to_markdown()
    errors = list(result.errors or [])
    return markdown, errors


def convert_pdf_to_markdown(
    pdf_path: Path,
    output_path: Path,
    chunk_size: int,
    page_start: int,
    page_end: int | None,
) -> Path:
    """Convert PDF to markdown with automatic chunking and error recovery.
    
    Strategy:
      1. Try chunk-based conversion (chunk_size pages per request)
      2. On chunk failure, fall back to per-page conversion
      3. Skip pages that fail completely
      4. Write successful pages to output file
    
    Args:
        pdf_path: Path to input PDF file
        output_path: Path to output markdown file
        chunk_size: Pages per conversion request (higher = faster but less stable)
        page_start: First page to convert (1-indexed)
        page_end: Last page to convert (None = detect from PDF)
        
    Returns:
        Path to output markdown file
        
    Raises:
        ValueError: If parameters are invalid
        FileNotFoundError: If PDF file doesn't exist
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    if page_start < 1:
        raise ValueError("page_start must be at least 1")
    if page_end is not None and page_end < page_start:
        raise ValueError("page_end must be greater than or equal to page_start")

    converter = DocumentConverter()
    markdown_parts: list[str] = []
    current_page = page_start

    while True:
        chunk_end = current_page + chunk_size - 1
        if page_end is not None:
            chunk_end = min(chunk_end, page_end)

        LOGGER.info("Converting pages %s-%s", current_page, chunk_end)
        try:
            chunk_markdown, chunk_errors = convert_page_range(converter, pdf_path, current_page, chunk_end)
        except Exception as exc:
            LOGGER.warning("Chunk %s-%s failed with %s; retrying one page at a time.", current_page, chunk_end, exc)
            chunk_markdown = ""
            chunk_errors = [exc]

        if chunk_markdown and not chunk_errors:
            markdown_parts.append(chunk_markdown.strip())
        else:
            if chunk_errors:
                LOGGER.warning("Chunk %s-%s produced %s error(s); converting pages individually.", current_page, chunk_end, len(chunk_errors))

            for page_number in range(current_page, chunk_end + 1):
                LOGGER.info("Converting page %s", page_number)
                try:
                    page_markdown, page_errors = convert_page_range(converter, pdf_path, page_number, page_number)
                except Exception as exc:
                    LOGGER.error("Page %s failed: %s", page_number, exc)
                    continue

                if page_markdown:
                    markdown_parts.append(f"<!-- Page {page_number} -->\n\n{page_markdown.strip()}")

                if page_errors:
                    LOGGER.warning("Page %s produced %s error(s) and may be partially converted.", page_number, len(page_errors))

                if not page_markdown and not page_errors:
                    LOGGER.info("No more pages found after page %s", page_number)
                    output_path.write_text("\n\n".join(part for part in markdown_parts if part).strip() + "\n", encoding="utf-8")
                    return output_path

        if page_end is not None and chunk_end >= page_end:
            break

        if page_end is None and not chunk_markdown and not chunk_errors:
            break

        current_page = chunk_end + 1

    final_markdown = "\n\n".join(part for part in markdown_parts if part).strip()
    if final_markdown:
        final_markdown += "\n"
    output_path.write_text(final_markdown, encoding="utf-8")
    return output_path


def main() -> int:
    """Convert PDF to markdown and save output.
    
    Workflow:
      1. Validate input PDF file exists
      2. Ensure output directory exists
      3. Configure logging
      4. Perform conversion
      5. Output result path
    
    Returns:
        0 on success, 1 on error
    """
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    input_pdf = INPUT_PDF_PATH.expanduser().resolve()
    if not input_pdf.exists():
        raise FileNotFoundError(f"Input PDF does not exist: {input_pdf}")
    if input_pdf.suffix.lower() != ".pdf":
        raise ValueError("Input file must be a PDF")

    output_path = OUTPUT_MD_PATH.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("PDF to Markdown Converter")
    print("="*60)
    print(f"\nInput:  {input_pdf}")
    print(f"Output: {output_path}")
    print(f"Chunk size: {CHUNK_SIZE} pages\n")

    convert_pdf_to_markdown(
        pdf_path=input_pdf,
        output_path=output_path,
        chunk_size=CHUNK_SIZE,
        page_start=PAGE_START,
        page_end=PAGE_END,
    )

    print("\n" + "="*60)
    print(f"Conversion complete: {output_path}")
    print("="*60 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
