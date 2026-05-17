"""
pdf_processor.py

Responsibility: take a PDF file path, return a list of clean chunks.
Each chunk is a dict with text + metadata.

We do NOT do embeddings here. That is the vector store's job.
This module only answers: "what is in this PDF and how is it structured?"
"""

import re
import fitz 


# ── Constants ─────────────────────────────────────────────────────────────────

# If a text block's font size is this much bigger than the median body font,
# we call it a heading. We'll calculate the median dynamically per document.
HEADING_FONT_RATIO = 1.15

# A chunk should not exceed this many words. When it does, we flush it and
# start a new one. 300 words ≈ one solid paragraph — enough context for the
# LLM, small enough for precise retrieval.
MAX_CHUNK_WORDS = 300

# If a block's y-position (top of block) is in the bottom X% of the page,
# treat it as a footnote and skip it. 0.88 = bottom 12% of page height.
FOOTNOTE_Y_THRESHOLD = 0.88

# Math patterns — if any of these appear in a chunk, we flag has_math=True.
# These are common LaTeX patterns found in research papers.
MATH_PATTERNS = [
    r"\$\$",          # display math:  $$...$$
    r"\$[^$]+\$",     # inline math:   $...$
    r"\\frac",        # fractions
    r"\\sum",         # summation
    r"\\int",         # integral
    r"\\alpha",       # greek letters (just checking alpha as a proxy)
    r"\^[\{\d]",      # superscripts: x^2 or x^{ij}
    r"_[\{\d]",       # subscripts:   x_i or x_{ij}
    r"\\mathbf",      # bold math
    r"\\left",        # large brackets
]


# ── Data structures ────────────────────────────────────────────────────────────

def make_chunk(chunk_id, section, page, text, block_types):
    """
    Build the chunk dictionary that everything downstream will consume.
    
    Keeping this as a plain dict (not a class) keeps things simple.
    We'll add more fields here as the project grows.
    """
    return {
        "chunk_id":   chunk_id,
        "section":    section,
        "page":       page,        # page where this chunk STARTS
        "text":       text.strip(),
        "word_count": len(text.split()),
        "has_math":   _detect_math(text),
        "has_code":   _detect_code(text),
    }


# ── Internal helpers ───────────────────────────────────────────────────────────

def _clean_text(text):
    """
    Raw PDF text has a lot of noise. Clean it up.
    
    - Multiple spaces → single space
    - Hyphenated line breaks (re-\nsearch → research) → rejoin
    - Strip leading/trailing whitespace from each line
    """
    # Rejoin words hyphenated across lines: "atten-\ntion" → "attention"
    text = re.sub(r"-\n(\w)", r"\1", text)
    # Collapse multiple whitespace characters into one space
    text = re.sub(r"[ \t]+", " ", text)
    # Remove lines that are just a number (page numbers)
    lines = text.split("\n")
    lines = [ln for ln in lines if not re.match(r"^\s*\d+\s*$", ln)]
    return "\n".join(lines).strip()


def _detect_math(text):
    """Return True if the text likely contains mathematical notation."""
    for pattern in MATH_PATTERNS:
        if re.search(pattern, text):
            return True
    return False


def _detect_code(text):
    """
    Return True if the text looks like a code block.
    Heuristic: 3+ lines starting with spaces/tabs, or contains 'def '/'for '/'import '.
    """
    code_keywords = ["def ", "for ", "import ", "return ", "class ", "if __name__"]
    if any(kw in text for kw in code_keywords):
        return True
    # Count lines that start with indentation
    lines = text.split("\n")
    indented = sum(1 for ln in lines if ln.startswith("  ") or ln.startswith("\t"))
    return indented >= 3


def _get_median_font_size(page):
    """
    Find the most common font size on a page.
    This becomes our baseline for detecting headings.
    
    PyMuPDF's get_text("dict") gives us each character's font size.
    We collect all sizes and find the median.
    """
    sizes = []
    blocks = page.get_text("dict")["blocks"]
    for block in blocks:
        if block["type"] != 0:  # type 0 = text, type 1 = image
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                sizes.append(round(span["size"]))  # font size in points
    
    if not sizes:
        return 11.0  # fallback: assume 11pt body text
    
    sizes.sort()
    return sizes[len(sizes) // 2]  # median


def _is_footnote(block, page_height):
    """
    A block is a footnote if its top y-coordinate is in the bottom 12% of the page.
    'bbox' is (x0, y0, x1, y1) — y0 is the TOP of the block.
    """
    y0 = block["bbox"][1]
    return (y0 / page_height) > FOOTNOTE_Y_THRESHOLD


def _is_figure_caption(text):
    """Figure captions start with 'Figure N' or 'Fig.' or 'Table N'."""
    text_stripped = text.strip()
    return bool(re.match(r"^(Figure|Fig\.|Table)\s+\d+", text_stripped, re.IGNORECASE))


def _classify_block(block, median_font, page_height):
    """
    Given a text block, decide what type it is.
    Returns one of: 'heading', 'body', 'caption', 'footnote', 'skip'
    
    This is where all our heuristics live. In phase 2 we can replace
    this with a small ML classifier, but heuristics work well for now.
    """
    if _is_footnote(block, page_height):
        return "footnote"
    
    # Get the dominant font size in this block
    block_sizes = []
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            block_sizes.append(span["size"])
    
    if not block_sizes:
        return "skip"
    
    block_font = max(set(block_sizes), key=block_sizes.count)  # mode font size
    
    # Get the text content
    text = " ".join(
        span["text"]
        for line in block.get("lines", [])
        for span in line.get("spans", [])
    )
    text = text.strip()
    
    if not text:
        return "skip"
    
    if _is_figure_caption(text):
        return "caption"
    
    # Heading: font is noticeably larger than body
    if block_font > median_font * HEADING_FONT_RATIO:
        return "heading"
    
    return "body"


# ── Public API ─────────────────────────────────────────────────────────────────

def extract_blocks(pdf_path):
    """
    Open a PDF and return a list of classified text blocks.
    
    Each block is a dict:
    {
        "type":  "heading" | "body" | "caption" | "footnote",
        "text":  str,
        "page":  int,   (1-indexed)
        "bbox":  (x0, y0, x1, y1)
    }
    
    This is the raw material. chunk_blocks() will process this further.
    """
    doc = fitz.open(pdf_path)
    all_blocks = []
    
    for page_num, page in enumerate(doc, start=1):
        page_height = page.rect.height
        median_font = _get_median_font_size(page)
        
        # get_text("dict") returns structured data with font info
        raw_blocks = page.get_text("dict")["blocks"]
        
        for block in raw_blocks:
            if block["type"] != 0:
                # Skip image blocks (type 1)
                continue
            
            block_type = _classify_block(block, median_font, page_height)
            
            if block_type == "footnote":
                continue  # discard footnotes entirely
            
            # Reconstruct text from spans (spans are runs of same-font text)
            raw_text = " ".join(
                span["text"]
                for line in block.get("lines", [])
                for span in line.get("spans", [])
            )
            clean = _clean_text(raw_text)
            
            if not clean:
                continue
            
            all_blocks.append({
                "type": block_type,
                "text": clean,
                "page": page_num,
                "bbox": block["bbox"],
            })
    
    doc.close()
    return all_blocks


def chunk_blocks(blocks):
    """
    Convert classified blocks into chunks with metadata.
    
    Strategy:
    - A new HEADING resets the current section and flushes the current chunk.
    - BODY text accumulates into the current chunk.
    - When a chunk exceeds MAX_CHUNK_WORDS, flush it and start a new one.
    - CAPTIONs get their own single-block chunk (they're self-contained).
    
    Returns a list of chunk dicts.
    """
    chunks = []
    chunk_id = 0
    current_section = "unknown"
    current_page = 1
    current_text = ""
    
    def flush_chunk():
        """Save current_text as a chunk if it has enough content."""
        nonlocal chunk_id, current_text
        text = current_text.strip()
        if len(text.split()) < 10:
            # Too short to be useful — discard
            current_text = ""
            return
        chunks.append(make_chunk(chunk_id, current_section, current_page, text, []))
        chunk_id += 1
        current_text = ""
    
    for block in blocks:
        
        if block["type"] == "heading":
            # A new heading means we're entering a new section.
            # Flush whatever we have, then update section name.
            flush_chunk()
            current_section = block["text"]
            current_page = block["page"]
        
        elif block["type"] == "caption":
            # Captions are standalone — flush current chunk, add caption, continue.
            flush_chunk()
            chunks.append(make_chunk(chunk_id, current_section, block["page"], block["text"], ["caption"]))
            chunk_id += 1
        
        elif block["type"] == "body":
            # Record the page of the first block in this chunk
            if not current_text:
                current_page = block["page"]
            
            current_text += " " + block["text"]
            
            # If we've exceeded the word limit, flush and start fresh
            if len(current_text.split()) >= MAX_CHUNK_WORDS:
                flush_chunk()
    
    # Don't forget the last chunk
    flush_chunk()
    
    return chunks


def process_pdf(pdf_path):
    """
    Main entry point. Takes a file path, returns chunks.
    
    Usage:
        chunks = process_pdf("attention_is_all_you_need.pdf")
        for chunk in chunks:
            print(chunk["section"], "—", chunk["word_count"], "words")
    """
    blocks = extract_blocks(pdf_path)
    chunks = chunk_blocks(blocks)
    return chunks