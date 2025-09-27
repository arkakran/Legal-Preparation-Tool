from dataclasses import dataclass
from typing import List

@dataclass
class KeyItem:
    title: str
    content: str
    stance: str  # "for", "against", "neutral"
    page_number: int
    line_start: int
    line_end: int
    confidence_score: float
    category: str  # "rule_based" | "statistical" | "llm"

@dataclass
class DocumentAnalysis:
    key_items: List[KeyItem]
    document_title: str
    total_pages: int
    processing_time: float