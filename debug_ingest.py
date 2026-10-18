"""
debug_ingest.py

Benchmark the fast DocChat ingestion pipeline.

Pipeline being tested:

    PDF
      ↓
    PyMuPDF + structural chapter/topic detection
      ↓
    Fixed-window chunks + overlap
      ↓
    Batched FastEmbed
      ↓
    [Chroma is NOT written in this debug script]

Usage:

    python debug_ingest.py "path/to/book.pdf"

The script reports:

    - PDF / structural extraction time
    - number of chapters/topics
    - chunking time
    - number of chunks
    - chunk statistics
    - embedding model loading time
    - embedding time
    - embedding throughput
    - total time

This makes it easy to identify the actual bottleneck.
"""

import sys
import time

from ingest import (
    read_pdf_once,
    group_by_heading,
    fixed_window_chunks,
    FastEmbedder,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBED_BATCH_SIZE,
    EMBED_PARALLEL,
    EMBEDDING_MODEL,
)


# --------------------------------------------------------------------------
# Display helpers
# --------------------------------------------------------------------------

def separator(char="=", length=80):
    print("\n" + char * length)


def print_time(label, seconds):
    print(
        f"{label:<30} {seconds:>9.2f}s"
    )


# --------------------------------------------------------------------------
# Main debug benchmark
# --------------------------------------------------------------------------

def debug_ingest(pdf_path: str):

    total_start = time.perf_counter()

    separator()
    print("DOCCHAT INGESTION BENCHMARK")
    separator()

    print(f"\nPDF: {pdf_path}")

    # ======================================================================
    # 1. PDF extraction + structural analysis
    # ======================================================================

    separator()
    print("1. PDF EXTRACTION + STRUCTURAL ANALYSIS")
    separator()

    structure_start = time.perf_counter()

    pdf_data = read_pdf_once(
        pdf_path
    )

    sections = group_by_heading(
        pdf_data
    )

    structure_time = (
            time.perf_counter()
            - structure_start
    )

    print(
        f"\nDetected sections/topics: "
        f"{len(sections)}"
    )

    # Print detected structure.
    for index, section in enumerate(
            sections
    ):

        heading = section.get(
            "heading",
            "Unknown",
        )

        page = section.get(
            "start_page",
            0,
        ) + 1

        characters = len(
            section.get(
                "text",
                "",
            )
        )

        print(
            f"  [{index:03d}] "
            f"Page {page:<4} "
            f"{heading} "
            f"({characters:,} chars)"
        )

    print_time(
        "Structural analysis:",
        structure_time,
    )

    # ======================================================================
    # 2. Fixed-window chunking
    # ======================================================================

    separator()
    print("2. FIXED-WINDOW CHUNKING")
    separator()

    print(
        f"Chunk size:     {CHUNK_SIZE} characters"
    )

    print(
        f"Chunk overlap:  {CHUNK_OVERLAP} characters"
    )

    chunk_start = time.perf_counter()

    chunks = fixed_window_chunks(
        sections,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunk_time = (
            time.perf_counter()
            - chunk_start
    )

    print(
        f"\nTotal chunks: "
        f"{len(chunks)}"
    )

    print_time(
        "Chunking:",
        chunk_time,
    )

    # ======================================================================
    # 3. Chunk statistics
    # ======================================================================

    separator()
    print("3. CHUNK STATISTICS")
    separator()

    if not chunks:

        print(
            "No chunks were generated."
        )

        return

    lengths = [
        len(chunk.page_content)
        for chunk in chunks
    ]

    print(
        f"Minimum size:  "
        f"{min(lengths):,} chars"
    )

    print(
        f"Maximum size:  "
        f"{max(lengths):,} chars"
    )

    print(
        f"Average size:  "
        f"{sum(lengths) / len(lengths):,.1f} chars"
    )

    # ======================================================================
    # 4. Chapter/topic distribution
    # ======================================================================

    separator()
    print("4. CHAPTER / TOPIC DISTRIBUTION")
    separator()

    chapter_counts = {}

    for chunk in chunks:

        chapter = chunk.metadata.get(
            "chapter",
            "Unknown",
        )

        chapter_counts[chapter] = (
                chapter_counts.get(
                    chapter,
                    0,
                )
                + 1
        )

    for chapter, count in (
            chapter_counts.items()
    ):

        print(
            f"{count:4d} chunks  |  "
            f"{chapter}"
        )

    # ======================================================================
    # 5. FastEmbed model loading
    # ======================================================================

    separator()
    print("5. FASTEMBED MODEL")
    separator()

    print(
        f"Model:       {EMBEDDING_MODEL}"
    )

    print(
        f"Batch size:  {EMBED_BATCH_SIZE}"
    )

    print(
        f"Parallel:    {EMBED_PARALLEL}"
    )

    model_start = time.perf_counter()

    embedder = FastEmbedder(
        model_name=EMBEDDING_MODEL,
        batch_size=EMBED_BATCH_SIZE,
        parallel=EMBED_PARALLEL,
    )

    model_load_time = (
            time.perf_counter()
            - model_start
    )

    print_time(
        "Model loading:",
        model_load_time,
    )

    # ======================================================================
    # 6. Batch embedding
    # ======================================================================

    separator()
    print("6. BATCH EMBEDDING")
    separator()

    embedding_start = time.perf_counter()

    embeddings = embedder.embed_documents(
        chunks
    )

    embedding_time = (
            time.perf_counter()
            - embedding_start
    )

    print(
        f"\nEmbeddings generated: "
        f"{len(embeddings)}"
    )

    if embeddings:

        print(
            f"Embedding dimension: "
            f"{len(embeddings[0])}"
        )

    print_time(
        "Embedding:",
        embedding_time,
    )

    # ======================================================================
    # 7. Embedding throughput
    # ======================================================================

    separator()
    print("7. EMBEDDING THROUGHPUT")
    separator()

    if embedding_time > 0:

        chunks_per_second = (
                len(chunks)
                / embedding_time
        )

        print(
            f"Chunks / second: "
            f"{chunks_per_second:.2f}"
        )

        print(
            f"Chunks / minute: "
            f"{chunks_per_second * 60:.0f}"
        )

    # ======================================================================
    # 8. Chunk previews
    # ======================================================================

    separator()
    print("8. CHUNK PREVIEW")
    separator()

    preview_count = min(
        15,
        len(chunks),
    )

    for index in range(
            preview_count
    ):

        chunk = chunks[index]

        print(
            f"\n--- Chunk {index} ---"
        )

        print(
            f"Chapter/topic: "
            f"{chunk.metadata.get('chapter')}"
        )

        print(
            f"Section index: "
            f"{chunk.metadata.get('section_index')}"
        )

        print(
            f"Chunk index: "
            f"{chunk.metadata.get('chunk_index')}"
        )

        print(
            f"Page: "
            f"{chunk.metadata.get('start_page', 0) + 1}"
        )

        print(
            f"Characters: "
            f"{len(chunk.page_content)}"
        )

        text = (
            chunk.page_content
            .replace("\n", " ")
            .strip()
        )

        if len(text) > 400:

            text = (
                    text[:400]
                    + "..."
            )

        print(text)

    if len(chunks) > preview_count:

        print(
            f"\n... "
            f"{len(chunks) - preview_count} "
            f"additional chunks not displayed."
        )

    # ======================================================================
    # 9. Final benchmark
    # ======================================================================

    total_time = (
            time.perf_counter()
            - total_start
    )

    separator()
    print("9. FINAL BENCHMARK")
    separator()

    print(
        f"\nPDF: "
        f"{pdf_path}"
    )

    print(
        f"Sections: "
        f"{len(sections)}"
    )

    print(
        f"Chunks: "
        f"{len(chunks)}"
    )

    print()

    print_time(
        "PDF + structure:",
        structure_time,
    )

    print_time(
        "Chunking:",
        chunk_time,
    )

    print_time(
        "Model loading:",
        model_load_time,
    )

    print_time(
        "Embedding:",
        embedding_time,
    )

    print_time(
        "TOTAL:",
        total_time,
    )

    # ======================================================================
    # 10. Bottleneck identification
    # ======================================================================

    separator()
    print("10. BOTTLENECK")
    separator()

    stages = {
        "PDF + structure": structure_time,
        "Chunking": chunk_time,
        "Model loading": model_load_time,
        "Embedding": embedding_time,
    }

    bottleneck = max(
        stages,
        key=stages.get,
    )

    bottleneck_time = stages[
        bottleneck
    ]

    percentage = (
        bottleneck_time
        / total_time
        * 100
        if total_time > 0
        else 0
    )

    print(
        f"Main bottleneck: "
        f"{bottleneck}"
    )

    print(
        f"Time: "
        f"{bottleneck_time:.2f}s"
    )

    print(
        f"Share of total: "
        f"{percentage:.1f}%"
    )

    # ======================================================================
    # 11. Target assessment
    # ======================================================================

    separator()
    print("11. TARGET")
    separator()

    if total_time < 10:

        print(
            "🚀 UNDER 10 SECONDS"
        )

        print(
            "The ingestion pipeline "
            "has reached the target."
        )

    elif total_time < 30:

        print(
            "⚡ UNDER 30 SECONDS"
        )

        print(
            "Significantly faster, "
            "but still above the 10-second target."
        )

    elif total_time < 60:

        print(
            "🟡 UNDER 1 MINUTE"
        )

        print(
            "There is still a significant "
            "performance bottleneck."
        )

    else:

        print(
            "🔴 OVER 1 MINUTE"
        )

        print(
            "We need to optimize the "
            "identified bottleneck."
        )

    separator()

    print(
        "\nNOTE:"
    )

    print(
        "This debug script does NOT write "
        "anything to Chroma."
    )

    print(
        "Chroma insertion will be benchmarked "
        "separately by ingest.py."
    )

    separator()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "\nUsage:"
        )

        print(
            '  python debug_ingest.py '
            '"path/to/book.pdf"\n'
        )

        sys.exit(1)

    debug_ingest(
        sys.argv[1]
    )

