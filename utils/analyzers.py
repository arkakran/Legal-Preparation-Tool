import re
import numpy as np
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from .llm_client import LLMAnalyzer

class RuleBasedAnalyzer:
    def __init__(self):
        self.legal_patterns = {
            'case_citations': r'(\d+\s+\w+\.?\s+\d+)|(\b[A-Z][A-Za-z]+ v\. [A-Z][A-Za-z]+\b)',
            'statutes': r'(\d+\s+U\.S\.C\.?\s+§?\s*\d+)|(\d+\s+CFR\s+§?\s*\d+)',
            'legal_terms': r'\b(whereas|therefore|contends|argues|holds|ruled|decided|granted|denied|precedent|constitutional|pre-?emption)\b',
            'argument_markers': r'\b(first|second|third|finally|moreover|furthermore|however|nevertheless|contrary)\b',
        }

    def score_paragraph(self, text: str):
        score = 0.0
        cats = []
        for name, pat in self.legal_patterns.items():
            if re.search(pat, text, re.IGNORECASE):
                score += 2.0
                cats.append(name)
        if len(text.split()) > 40:
            score += 1.0
        return score, cats

    def analyze(self, paragraphs: List[Dict]) -> List[Dict]:
        out = []
        for p in paragraphs:
            s, cats = self.score_paragraph(p["text"])
            if s > 0:
                out.append({
                    "text": p["text"],
                    "page": p["page"],
                    "line_start": p["line_start"],
                    "line_end": p["line_end"],
                    "score": s,
                    "categories": cats,
                    "header_context": p.get("header_context"),
                    "method": "rule_based",
                })
        return sorted(out, key=lambda x: x["score"], reverse=True)

class StatisticalAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=4000,
            stop_words='english',
            ngram_range=(1, 2)
        )

    def analyze(self, paragraphs: List[Dict]) -> List[Dict]:
        if not paragraphs:
            return []
        corpus = [p["text"] for p in paragraphs]
        try:
            X = self.vectorizer.fit_transform(corpus)
            raw = np.asarray(X.sum(axis=1)).ravel()
            rmin, rmax = float(raw.min()), float(raw.max())
            norm = (raw - rmin) / (rmax - rmin + 1e-9)
            results = []
            for p, s in zip(paragraphs, norm):
                pos_weight = 1.0 / (1.0 + 0.06 * p["page"])
                total = 0.7 * s + 0.3 * pos_weight
                results.append({
                    "text": p["text"],
                    "page": p["page"],
                    "line_start": p["line_start"],
                    "line_end": p["line_end"],
                    "score": float(total),
                    "header_context": p.get("header_context"),
                    "method": "statistical",
                })
            return sorted(results, key=lambda x: x["score"], reverse=True)
        except Exception as e:
            print(f"Statistical analyzer error: {e}")
            return []