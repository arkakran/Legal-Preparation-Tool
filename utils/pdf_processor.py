import re
from typing import List, Dict, Optional
import pdfplumber

# Heuristics and Patterns
EXCLUDED_HEADERS = {
    "TABLE OF CONTENTS", "TABLE OF AUTHORITIES", "INDEX", "APPENDIX",
    "CERTIFICATE OF SERVICE", "SIGNATURE",
}

BOILERPLATE_PATTERNS = [
    r'^Case\s+\d+:\d{2}-[A-Za-z0-9-]+?\s+Document\s+\d+\s+Filed\s+\d{2}/\d{2}/\d{2}\s+Page\s+\d+\s+of\s+\d+\s+PageID\s+\d+\s*$',
    r'^Page\s+\d+\s+of\s+\d+\s*$',
    r'^\s*—+\s*$',
    r'^\s*[-•]+\s*$',
]

HEADER_PATTERN = r'^[A-Z][A-Z0-9\s\-\.:,]{4,}$'

def is_boilerplate(line: str) -> bool:
    for pat in BOILERPLATE_PATTERNS:
        if re.search(pat, line, re.IGNORECASE):
            return True
    return False

def is_header(line: str) -> bool:
    text = line.strip()
    if len(text) < 5:
        return False
    return bool(re.match(HEADER_PATTERN, text))

def looks_index_like(text: str) -> bool:
    if "Page(s)" in text or re.search(r'\.{5,}', text):
        return True
    if text.count(" v. ") >= 2:
        return True
    tail_nums = len(re.findall(r'\s\d{1,3}\s*$', text))
    if tail_nums and len(text) < 200:
        return True
    return False

def clean_line_text(t: str) -> str:
    t = re.sub(r'https?://\S+', '', t)
    t = re.sub(r'\s*\b(\d{1,2})\s*$', '', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def split_into_paragraphs(pages: List[Dict]) -> List[Dict]:
    paragraphs = []
    for page in pages:
        current_lines = []
        current_start = None
        page_no = page['page_number']
        last_header = None
        exclude_section = False

        for ln in page['lines']:
            raw = (ln['text'] or "").rstrip()
            if not raw:
                if current_lines:
                    para_text = " ".join(current_lines).strip()
                    if para_text and not looks_index_like(para_text) and not exclude_section:
                        paragraphs.append({
                            "text": para_text,
                            "page": page_no,
                            "line_start": current_start,
                            "line_end": ln['line_number'] - 1,
                            "header_context": last_header,
                        })
                    current_lines, current_start = [], None
                continue

            if is_boilerplate(raw):
                if current_lines:
                    para_text = " ".join(current_lines).strip()
                    if para_text and not looks_index_like(para_text) and not exclude_section:
                        paragraphs.append({
                            "text": para_text,
                            "page": page_no,
                            "line_start": current_start,
                            "line_end": ln['line_number'] - 1,
                            "header_context": last_header,
                        })
                    current_lines, current_start = [], None
                continue

            if is_header(raw):
                if current_lines:
                    para_text = " ".join(current_lines).strip()
                    if para_text and not looks_index_like(para_text) and not exclude_section:
                        paragraphs.append({
                            "text": para_text,
                            "page": page_no,
                            "line_start": current_start,
                            "line_end": ln['line_number'] - 1,
                            "header_context": last_header,
                        })
                    current_lines, current_start = [], None

                last_header = raw.strip().upper()
                exclude_section = last_header in EXCLUDED_HEADERS
                continue

            if current_start is None:
                current_start = ln['line_number']
            current_lines.append(clean_line_text(raw))

        if current_lines:
            para_text = " ".join(current_lines).strip()
            if para_text and not looks_index_like(para_text) and not exclude_section:
                paragraphs.append({
                    "text": para_text,
                    "page": page_no,
                    "line_start": current_start,
                    "line_end": page['lines'][-1]['line_number'] if page['lines'] else current_start,
                    "header_context": last_header,
                })

    paragraphs = [p for p in paragraphs if len(p["text"].split()) >= 5]
    return paragraphs

class PDFProcessor:
    def extract_text_with_coordinates(self, file_path: str) -> Optional[Dict]:
        doc = {'pages': [], 'full_text': '', 'total_pages': 0}
        try:
            with pdfplumber.open(file_path) as pdf:
                doc['total_pages'] = len(pdf.pages)
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text() or ""
                    lines = page_text.split("\n") if page_text else []
                    page_data = {'page_number': page_num, 'lines': [], 'text': page_text}
                    for i, line in enumerate(lines, 1):
                        page_data['lines'].append({
                            'line_number': i,
                            'text': line,
                            'page_number': page_num
                        })
                    doc['pages'].append(page_data)
                    doc['full_text'] += (page_text + "\n")
        except Exception as e:
            print(f"Error extracting PDF: {e}")
            return None
        return doc