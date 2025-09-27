import re
import time
from typing import List, Dict, Tuple, Optional
from .models import KeyItem, DocumentAnalysis
from .pdf_processor import PDFProcessor, split_into_paragraphs
from .analyzers import RuleBasedAnalyzer, StatisticalAnalyzer, LLMAnalyzer

TITLE_THEMES = [
    ("Comstock Act (18 U.S.C. §§ 1461–1462)", ["18 u.s.c. § 1461", "18 u.s.c. § 1462", "comstock"]),
    ("Clear Statement & Preemption", ["pre-empt", "preempt", "clear and manifest", "rice v. santa fe"]),
    ("Public Interest & Equities", ["public interest", "equities", "injunctive relief"]),
    ("REMS & Safety Controls", ["rems", "risk evaluation", "safety requirements", "in-person"]),
    ("Subpart H & Serious Illness", ["subpart h", "serious or life-threatening", "21 c.f.r. § 314.500"]),
    ("Federalism & State Police Powers", ["historic police powers", "sovereign interest", "states' laws"]),
    ("FDA Authority & Limits", ["fda", "approval", "authority", "unlawful"]),
    ("Dobbs & Post-Dobbs Framework", ["dobbs", "elected representatives", "states may"]),
]

COUNTER_CUES = ["however", "nevertheless", "but", "contrary", "respondents", "intervenors"]
PRO_AMICI_CUES = ["amici", "plaintiffs", "states", "sovereign interest", "injunctive relief"]
CRITICISM_OF_FEDERAL = ["fda", "administration", "white house", "justice department", "doj"]
NEGATIVE_VERBS = ["violates", "undermine", "contravene", "defy", "illegal", "unlawful", "preempt"]

class HybridProcessor:
    def __init__(self, groq_api_key: str):
        self.pdf = PDFProcessor()
        self.rule = RuleBasedAnalyzer()
        self.stat = StatisticalAnalyzer()
        self.llm = LLMAnalyzer(groq_api_key)

    def stance_for_amici(self, text: str) -> str:
        tl = (text or "").lower()
        if any(k in tl for k in CRITICISM_OF_FEDERAL) and any(v in tl for v in NEGATIVE_VERBS):
            return "for"
        if any(k in tl for k in PRO_AMICI_CUES):
            return "for"
        if any(k in tl for k in COUNTER_CUES):
            return "against"
        if re.search(r'\b(respondents?|defendants?)\b', tl) and re.search(r'\b(argue|contend|assert|claim)\b', tl):
            return "against"
        return "neutral"

    def build_title(self, paragraph: Dict) -> str:
        txt = (paragraph.get("text") or "").lower()
        for title, keys in TITLE_THEMES:
            if any(k in txt for k in keys):
                return title
        if paragraph.get("header_context"):
            return paragraph["header_context"].title()[:120]
        first = re.split(r'(?<=[\.\?\!])\s+', paragraph.get("text", "").strip())
        return (first[0] if first else paragraph.get("text", ""))[:140]

    def map_quote_to_span(self, quote: str, pages: List[Dict]) -> Optional[Tuple[int, int, int]]:
        q = " ".join((quote or "").split()).strip()
        if not q or len(q.split()) < 3:
            return None
        q_lower = q.lower()
        for page in pages:
            text = (page.get("text") or "")
            if not text:
                continue
            text_lower = text.lower()
            if q_lower in text_lower:
                for ln in page["lines"]:
                    if q_lower[:40] in (ln.get("text") or "").lower():
                        s = ln["line_number"]
                        return page["page_number"], s, s
                return page["page_number"], 1, 1
        return None

    def _normalize_text_key(self, t: str) -> str:
        return re.sub(r'\s+', ' ', (t or "").strip().lower())[:800]

    def combine(self, rule_res: List[Dict], stat_res: List[Dict], llm_res: List[Dict], doc: Dict) -> List[KeyItem]:
        items: List[KeyItem] = []
        seen_text = set()

        def norm_scores(arr: List[float]) -> List[float]:
            if not arr:
                return []
            amin, amax = min(arr), max(arr)
            if amax - amin < 1e-9:
                return [0.5 for _ in arr]
            return [(x-amin)/(amax-amin) for x in arr]

        rb_scores = norm_scores([x["score"] for x in rule_res])
        st_scores = norm_scores([x["score"] for x in stat_res])

        def already_seen(txt: str) -> bool:
            k = self._normalize_text_key(txt)
            for s in seen_text:
                if k in s or s in k:
                    return True
            return False

        # Rule-based results
        for idx, x in enumerate(rule_res[:60]):
            t = x["text"]
            if already_seen(t):
                continue
            stance = self.stance_for_amici(t)
            title = self.build_title({"text": t, "header_context": x.get("header_context")})
            conf = min(0.6 * (rb_scores[idx] if idx < len(rb_scores) else 0.5) + 0.3, 0.98)
            items.append(KeyItem(
                title=title, content=t, stance=stance,
                page_number=x["page"], line_start=x["line_start"], line_end=x["line_end"],
                confidence_score=conf, category="rule_based",
            ))
            seen_text.add(self._normalize_text_key(t))

        # Statistical results  
        for idx, x in enumerate(stat_res[:60]):
            t = x["text"]
            if already_seen(t):
                continue
            stance = self.stance_for_amici(t)
            title = self.build_title({"text": t, "header_context": x.get("header_context")})
            conf = min(0.65 * (st_scores[idx] if idx < len(st_scores) else 0.5) + 0.25, 0.98)
            items.append(KeyItem(
                title=title, content=t, stance=stance,
                page_number=x["page"], line_start=x["line_start"], line_end=x["line_end"],
                confidence_score=conf, category="statistical",
            ))
            seen_text.add(self._normalize_text_key(t))

        # LLM results (with enhanced context)
        paragraphs = split_into_paragraphs(doc["pages"])
        for pt in llm_res:
            quote = (pt.get("quote") or "").strip()
            mapped = self.map_quote_to_span(quote, doc["pages"]) if quote else None

            context_text = pt.get("summary", "")
            page, ls, le = (1, 1, 1)
            if mapped:
                page, ls, le = mapped
                # Expand context with nearby lines
                page_obj = next((p for p in doc["pages"] if p["page_number"] == page), None)
                if page_obj:
                    lines = page_obj["lines"]
                    s_idx = max(0, (ls - 1) - 3)
                    e_idx = min(len(lines), (le - 1) + 4)
                    window = " ".join((ln.get("text") or "") for ln in lines[s_idx:e_idx])
                    if window.strip() and len(window) > len(context_text):
                        context_text = window
            else:
                # Fallback: look for related text in paragraphs
                summary_lower = (pt.get("summary") or "").lower()
                for p in doc["pages"]:
                    page_text = (p.get("text") or "").lower()
                    if any(word in page_text for word in summary_lower.split()[:5]):
                        page = p["page_number"]
                        for para in paragraphs:
                            if para["page"] == page and len(para["text"]) > 200:
                                if any(word in para["text"].lower() for word in summary_lower.split()[:3]):
                                    context_text = para["text"]
                                    ls, le = para["line_start"], para["line_end"]
                                    break
                        break

            stance = self.stance_for_amici(context_text)
            importance = pt.get("importance", 5)
            conf = min(max((importance / 10.0) * 0.95, 0.12), 0.98)

            # Prefer longer context over bare summary
            content_text = context_text if len(context_text) > 50 else pt.get("summary", "")
            if already_seen(content_text):
                continue

            items.append(KeyItem(
                title=(pt.get("summary") or "LLM Point")[:140],
                content=content_text,
                stance=stance,
                page_number=page,
                line_start=ls,
                line_end=le,
                confidence_score=conf,
                category=pt.get("category", "llm"),
            ))
            seen_text.add(self._normalize_text_key(content_text))

        return items

    def select_top10_strict_balance(self, items: List[KeyItem]) -> List[KeyItem]:
        items = sorted(items, key=lambda k: k.confidence_score, reverse=True)
        for_list = [it for it in items if it.stance == "for"]
        against_list = [it for it in items if it.stance == "against"]
        neutral_list = [it for it in items if it.stance == "neutral"]

        selected: List[KeyItem] = []
        take_for = min(5, len(for_list))
        take_against = min(5, len(against_list))

        selected.extend(for_list[:take_for])
        selected.extend(against_list[:take_against])

        while len(selected) < 10:
            need_for = 5 - len([s for s in selected if s.stance == "for"]) 
            need_against = 5 - len([s for s in selected if s.stance == "against"]) 
            
            if need_for > 0 and len(for_list) > len([s for s in selected if s.stance == "for"]):
                next_for = [f for f in for_list if f not in selected]
                if next_for:
                    selected.append(next_for[0])
                    continue
                    
            if need_against > 0 and len(against_list) > len([s for s in selected if s.stance == "against"]):
                next_against = [a for a in against_list if a not in selected]
                if next_against:
                    selected.append(next_against[0])
                    continue
                    
            next_neutral = [n for n in neutral_list if n not in selected]
            if next_neutral:
                selected.append(next_neutral[0])
                continue
                
            remaining = [it for it in items if it not in selected]
            if remaining:
                selected.append(remaining[0])
                continue
            break

        return selected[:10]

    def process(self, file_path: str, filename: str) -> Optional[DocumentAnalysis]:
        t0 = time.time()
        
        doc = self.pdf.extract_text_with_coordinates(file_path)
        if not doc:
            return None

        paragraphs = split_into_paragraphs(doc["pages"])
        rule_res = self.rule.analyze(paragraphs)
        stat_res = self.stat.analyze(paragraphs)

        top_mix = []
        seen = set()
        for x in (rule_res[:18] + stat_res[:18]):
            if x["text"] not in seen:
                top_mix.append(x["text"])
                seen.add(x["text"])

        for p in paragraphs[:20]:
            if p["text"] not in seen and len(top_mix) < 60:
                top_mix.append(p["text"])
                seen.add(p["text"])

        llm_block = "\n\n".join(top_mix[:60])
        llm_res = self.llm.analyze_points(llm_block)

        combined = self.combine(rule_res, stat_res, llm_res, doc)
        top10 = self.select_top10_strict_balance(combined)

        return DocumentAnalysis(
            key_items=top10,
            document_title=filename,
            total_pages=doc["total_pages"],
            processing_time=time.time() - t0
        )
