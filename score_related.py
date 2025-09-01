"""
SPDX-License-Identifier: MIT
Copyright (c) 2025 Akitsugu Komiyama
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import joblib
import numpy as np
from bs4 import BeautifulSoup
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

# 取り回しのため、build_index.py の一部関数をこちらにも軽量実装
import re
import unicodedata

def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"https?://\S+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def extract_text_from_html(html: str, drop_selectors: List[str]) -> Dict[str, str]:
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "noscript"]):
        t.decompose()
    for sel in drop_selectors:
        for t in soup.select(sel):
            t.decompose()
    title = (soup.title.get_text(" ", strip=True) if soup.title else "").strip()
    headings = " ".join(h.get_text(" ", strip=True) for h in soup.select("h1, h2, h3")).strip()
    body_node = soup.body if soup.body else soup
    body = body_node.get_text(" ", strip=True)
    return {"title": title, "headings": headings, "body": body}


def load_index(index_dir: Path):
    vectorizer = joblib.load(index_dir / "tfidf_vectorizer.pkl")
    X = sparse.load_npz(index_dir / "tfidf_matrix.npz")
    id_map = json.loads((index_dir / "id_map.json").read_text(encoding="utf-8"))
    src_root = (index_dir / "source_root.txt").read_text(encoding="utf-8").strip()
    return vectorizer, X, id_map, Path(src_root)


def build_query_vector(query_html_path: Path, drop_selectors: List[str], title_weight: int, heading_weight: int, vectorizer) -> sparse.csr_matrix:
    raw = query_html_path.read_text(encoding="utf-8", errors="ignore")
    parts = extract_text_from_html(raw, drop_selectors)
    weighted_text = (
        ((parts["title"] + " ") * max(1, title_weight)) +
        ((parts["headings"] + " ") * max(1, heading_weight)) +
        parts["body"]
    )
    q = normalize_text(weighted_text)
    return vectorizer.transform([q])  # 1 x V の疎行列


def main():
    ap = argparse.ArgumentParser(description="Score related HTML files against a query HTML (Japanese)")
    ap.add_argument("--index", type=Path, required=True, help="Index directory from build_index.py")
    ap.add_argument("--query", type=Path, required=True, help="Query HTML file (A.html)")
    ap.add_argument("--topk", type=int, default=10, help="Top K results to show (default: 10)")
    ap.add_argument("--tau", type=float, default=0.25, help="Similarity threshold (default: 0.25)")
    ap.add_argument("--format", type=str, default="table", choices=["table", "json"], help="Output format")
    ap.add_argument(
        "--drop-selectors",
        type=str,
        default="nav,footer,header,aside,.sidebar,.breadcrumbs,.breadcrumb,.global-nav,.site-footer",
        help="Comma-separated CSS selectors to drop before extracting text",
    )
    ap.add_argument("--title-weight", type=int, default=3, help="title weight (default: 3)")
    ap.add_argument("--heading-weight", type=int, default=2, help="headings weight (default: 2)")

    args = ap.parse_args()
    vectorizer, X, id_map, src_root = load_index(args.index)

    drop_selectors = [s.strip() for s in args.drop_selectors.split(",") if s.strip()]
    qvec = build_query_vector(args.query, drop_selectors, args.title_weight, args.heading_weight, vectorizer)

    # 類似度: 1 × N
    sims = cosine_similarity(qvec, X).ravel()

    # スコアの高い順にソート
    order = np.argsort(-sims)

    # 表示用にまとめる
    results = []
    for idx in order[: max(args.topk * 5, args.topk)]:  # しきい値で削れる分の余裕を持つ
        score = float(sims[idx])
        if score < args.tau:
            continue
        meta = id_map[str(idx)]
        results.append({
            "rank": len(results) + 1,
            "score": round(score, 4),
            "path": str(Path(src_root) / meta["path"]),
            "relpath": meta["path"],
            "title": meta.get("title", ""),
        })
        if len(results) >= args.topk:
            break

    if args.format == "json":
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        # 表形式
        if not results:
            print("No results above tau. Try lowering --tau.")
            return
        # 簡易テーブル出力
        colw = {
            "rank": 4,
            "score": 6,
            "relpath": 50,
            "title": 40,
        }
        print(f"{'#':>{colw['rank']}}  {'score':>{colw['score']}}  {'relpath':<{colw['relpath']}}  {'title':<{colw['title']}}")
        print("-" * (sum(colw.values()) + 8))
        for r in results:
            print(f"{r['rank']:>{colw['rank']}}  {r['score']:>{colw['score']}.4f}  {r['relpath']:<{colw['relpath']}}  {r['title']:<{colw['title']}}")


if __name__ == "__main__":
    main()
