import argparse
import json
import os
import re
import hashlib
import unicodedata
from pathlib import Path
from typing import List, Dict, Any

import joblib
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse


def normalize_text(s: str) -> str:
    """日本語向けの軽い正規化。
    - 全角/半角の正規化(NFKC)
    - URL除去
    - 連続空白の圧縮
    """
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"https?://\S+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def extract_text_from_html(html: str, drop_selectors: List[str]) -> Dict[str, str]:
    """HTMLからタイトル/見出し/本文を抽出。不要要素を除去します。"""
    # まず lxml、無ければ標準パーサ
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    # 明確に不要なタグ
    for t in soup(["script", "style", "noscript"]):
        t.decompose()

    # CSSセレクタで指定された要素を除去（ナビ・フッター・サイドバー等）
    for sel in drop_selectors:
        for t in soup.select(sel):
            t.decompose()

    title = (soup.title.get_text(" ", strip=True) if soup.title else "").strip()

    # 見出しは h1~h3 を採用（必要なら拡張）
    headings = " ".join(
        h.get_text(" ", strip=True) for h in soup.select("h1, h2, h3")
    ).strip()

    # 本文は body から。bodyが無い場合は全体から。
    body_node = soup.body if soup.body else soup
    body = body_node.get_text(" ", strip=True)

    return {
        "title": title,
        "headings": headings,
        "body": body,
    }


def build_index(src_dir: Path, out_dir: Path, ngram: int, min_df: int, max_df: float,
                title_weight: int, heading_weight: int, drop_selectors: List[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    doc_infos: List[Dict[str, Any]] = []
    corpus: List[str] = []

    html_files = sorted([p for p in src_dir.rglob("*.html")])
    if not html_files:
        raise SystemExit(f"No .html found under: {src_dir}")

    for p in html_files:
        raw = p.read_text(encoding="utf-8", errors="ignore")
        parts = extract_text_from_html(raw, drop_selectors)

        # 重み付け: タイトル×w1 + 見出し×w2 + 本文
        weighted_text = (
            ((parts["title"] + " ") * max(1, title_weight)) +
            ((parts["headings"] + " ") * max(1, heading_weight)) +
            parts["body"]
        )

        text_norm = normalize_text(weighted_text)
        body_norm = normalize_text(parts["body"])  # ハッシュは本文ベース
        content_hash = hashlib.sha1(body_norm.encode("utf-8")).hexdigest()

        relpath = str(p.relative_to(src_dir))
        doc_infos.append({
            "path": relpath,
            "title": parts["title"],
            "length": len(text_norm),
            "hash": content_hash,
        })
        corpus.append(text_norm)

    # 文字 n-gram TF-IDF
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(ngram, ngram),
        min_df=min_df,
        max_df=max_df,
    )
    X = vectorizer.fit_transform(corpus)  # sparse matrix

    # 保存
    (out_dir / "docs.jsonl").write_text(
        "\n".join(json.dumps(d, ensure_ascii=False) for d in doc_infos), encoding="utf-8"
    )

    # ID→メタデータ
    id_map = {i: doc_infos[i] for i in range(len(doc_infos))}
    (out_dir / "id_map.json").write_text(
        json.dumps(id_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ベクトライザと行列
    joblib.dump(vectorizer, out_dir / "tfidf_vectorizer.pkl")
    sparse.save_npz(out_dir / "tfidf_matrix.npz", X)

    # ルート（ソース）も保存しておく
    (out_dir / "source_root.txt").write_text(str(src_dir.resolve()), encoding="utf-8")

    print(f"Indexed {len(doc_infos)} HTML files → {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="Build TF-IDF index from HTML files (Japanese)")
    ap.add_argument("--src", type=Path, required=True, help="HTML root directory")
    ap.add_argument("--out", type=Path, required=True, help="Output index directory")
    ap.add_argument("--ngram", type=int, default=3, help="character n-gram size (default: 3)")
    ap.add_argument("--min-df", type=int, default=2, help="min_df for TF-IDF (default: 2)")
    ap.add_argument("--max-df", type=float, default=0.95, help="max_df for TF-IDF (default: 0.95)")
    ap.add_argument("--title-weight", type=int, default=3, help="title weight (default: 3)")
    ap.add_argument("--heading-weight", type=int, default=2, help="headings weight (default: 2)")
    ap.add_argument(
        "--drop-selectors",
        type=str,
        default="nav,footer,header,aside,.sidebar,.breadcrumbs,.breadcrumb,.global-nav,.site-footer",
        help="Comma-separated CSS selectors to drop before extracting text",
    )
    args = ap.parse_args()

    drop_selectors = [s.strip() for s in args.drop_selectors.split(",") if s.strip()]

    build_index(
        src_dir=args.src,
        out_dir=args.out,
        ngram=args.ngram,
        min_df=args.min_df,
        max_df=args.max_df,
        title_weight=args.title_weight,
        heading_weight=args.heading_weight,
        drop_selectors=drop_selectors,
    )


if __name__ == "__main__":
    main()