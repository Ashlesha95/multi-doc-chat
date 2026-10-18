import fitz  # PyMuPDF


def render_highlighted_page(pdf_path: str, page_no: int, text: str, zoom: float = 1.6):
    """Render a PDF page (page_no is 1-based) with the given passage highlighted.
    Returns (png_bytes, number_of_highlights)."""
    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(page_no - 1)
        words = text.split()
        found = 0
        # search in short windows, because long strings often fail across line breaks
        for i in range(0, len(words), 6):
            phrase = " ".join(words[i:i + 6])
            if len(phrase) < 8:
                continue
            for rect in page.search_for(phrase):
                page.add_highlight_annot(rect)
                found += 1
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        return pix.tobytes("png"), found
    finally:
        doc.close()