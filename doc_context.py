"""
Document identity and context extraction helpers for Sengoku-domain HTML pages.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


CATEGORY_LABELS = {
    "bushou": "武将",
    "hime": "姫",
    "meibutsu": "名物",
    "kassen": "合戦",
    "itsuwa": "逸話",
}

TERM_TOKEN_RE = re.compile(r"[一-龥ぁ-んァ-ヶA-Za-z0-9・ー]{2,24}")
TITLE_QUALIFIER_RE = re.compile(r"^(?P<name>.*?)(?:\s*[（(](?P<qualifier>[^()（）]+)[)）])?\s*$")

RELATION_VALUE_PATTERNS = [
    re.compile(r"(?:父|母|養父|養母|祖父|祖母|夫|妻|兄|弟|姉|妹|主君|当主|正室|側室|家臣|嫡男|次男|三男)は([一-龥ぁ-んァ-ヶA-Za-z0-9・ー]{2,24})"),
    re.compile(r"([一-龥ぁ-んァ-ヶA-Za-z0-9・ー]{2,24})の(?:父|母|養父|養母|祖父|祖母|夫|妻|兄|弟|姉|妹|子|娘|主君|家臣|嫡男|次男|三男|正室|側室)"),
]

DOMAIN_TERM_PATTERNS = [
    re.compile(r"([一-龥ぁ-んァ-ヶA-Za-z0-9・ー]{2,24}(?:家|氏|流|党|衆|藩|国|郡|城|寺|院|一門|宗家|分家|領主|大名|当主|正室|側室|合戦|戦い|の乱|逸話|伝承|名物))"),
]

STOP_TERMS = {
    "戦国時代",
    "戦国武将",
    "日本史",
    "歴史",
    "人物",
    "生涯",
    "評価",
    "目的",
    "報告書",
    "本報告書",
    "序章",
    "終章",
    "第一章",
    "第二章",
    "第三章",
    "第四章",
    "第五章",
    "第六章",
    "武将",
    "姫",
    "名物",
    "合戦",
    "逸話",
    "時代",
    "日本",
    "天下",
    "当時",
    "一方",
    "存在",
    "関係",
    "周辺",
    "詳細",
    "功績",
    "研究",
    "背景",
    "役割",
    "勢力",
    "家中",
    "領国",
    "戦い",
}


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"https?://\S+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def parse_relpath_identity(relpath: str) -> Tuple[str, str]:
    stem = Path(relpath).stem
    match = re.match(r"page-([^-]+)-(.+)$", stem)
    if not match:
        return "", stem
    return match.group(1), match.group(2).strip()


def split_title_qualifier(title: str) -> Tuple[str, str]:
    text = normalize_text(title)
    if not text:
        return "", ""
    match = TITLE_QUALIFIER_RE.match(text)
    if not match:
        return text, ""
    name = normalize_text(match.group("name") or "")
    qualifier = normalize_text(match.group("qualifier") or "")
    return name, qualifier


def _tokenize(text: str) -> List[str]:
    return [normalize_text(t) for t in TERM_TOKEN_RE.findall(text)]


def _is_useful_term(term: str, canonical_name: str, qualifier_terms: Iterable[str]) -> bool:
    if not term or term in STOP_TERMS:
        return False
    if len(term) < 2:
        return False
    if term == canonical_name:
        return False
    if canonical_name and term in canonical_name and len(term) <= max(2, len(canonical_name) // 2):
        return False
    if term in qualifier_terms:
        return True
    if term.endswith(("時代", "人物", "歴史", "意味", "評価", "役割", "研究", "目的")):
        return False
    return True


def _collect_terms(text: str, canonical_name: str, qualifier_terms: List[str]) -> Counter[str]:
    counter: Counter[str] = Counter()
    normalized = normalize_text(text)

    for pattern in RELATION_VALUE_PATTERNS:
        for match in pattern.findall(normalized):
            term = normalize_text(match)
            if _is_useful_term(term, canonical_name, qualifier_terms):
                counter[term] += 4

    for pattern in DOMAIN_TERM_PATTERNS:
        for match in pattern.findall(normalized):
            term = normalize_text(match)
            if _is_useful_term(term, canonical_name, qualifier_terms):
                counter[term] += 3

    for token in _tokenize(normalized):
        if _is_useful_term(token, canonical_name, qualifier_terms):
            counter[token] += 1

    return counter


def build_doc_profile(relpath: str, title: str, h1: str, headings: str, body: str) -> Dict[str, Any]:
    category, fallback_name = parse_relpath_identity(relpath)
    display_title = normalize_text(title) or normalize_text(h1) or normalize_text(fallback_name)
    canonical_name, qualifier = split_title_qualifier(display_title)
    qualifier_terms = [t for t in _tokenize(qualifier) if _is_useful_term(t, canonical_name, [])]

    # Focus on the most identity-rich parts first, then broaden into the opening body.
    context_source = " ".join(
        part for part in [
            qualifier,
            headings,
            body[:2400],
        ] if part
    )
    counts = _collect_terms(context_source, canonical_name, qualifier_terms)
    for token in qualifier_terms:
        counts[token] += 6
    if category:
        counts[CATEGORY_LABELS.get(category, category)] += 2
        counts[f"category:{category}"] += 2

    context_terms = [term for term, _ in counts.most_common(24)]

    return {
        "category": category,
        "category_label": CATEGORY_LABELS.get(category, category),
        "display_title": display_title,
        "canonical_name": canonical_name or display_title,
        "qualifier": qualifier,
        "qualifier_terms": qualifier_terms,
        "context_terms": context_terms,
    }
