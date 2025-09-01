"""
SPDX-License-Identifier: MIT
Copyright (c) 2025 Akitsugu Komiyama
"""

import argparse
import json
import os
import re
import hashlib
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Tuple

import joblib
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import numpy as np


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


def _chunk_text(text: str, max_chars: int, overlap: int) -> List[Tuple[int, int, str]]:
    """Split text into overlapping character chunks.
    Returns list of tuples: (start_index, end_index, chunk_text)
    """
    if max_chars <= 0:
        return [(0, len(text), text)]
    n = len(text)
    if n <= max_chars:
        return [(0, n, text)]
    chunks: List[Tuple[int, int, str]] = []
    start = 0
    step = max(1, max_chars - max(0, overlap))
    while start < n:
        end = min(n, start + max_chars)
        chunks.append((start, end, text[start:end]))
        if end >= n:
            break
        start = end - max(0, overlap)
    return chunks


def build_index(src_dir: Path, out_dir: Path, ngram: int, min_df: int, max_df: float,
                title_weight: int, heading_weight: int, drop_selectors: List[str],
                embed_model: str = "", embed_max_chars: int = 800, embed_overlap: int = 200,
                embed_batch_size: int = 32) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    doc_infos: List[Dict[str, Any]] = []
    corpus: List[str] = []
    embed_texts: List[str] = []

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
        embed_text = normalize_text((parts["title"] + " ") + (parts["headings"] + " ") + parts["body"])  # 埋め込み用
        content_hash = hashlib.sha1(body_norm.encode("utf-8")).hexdigest()

        relpath = str(p.relative_to(src_dir))
        doc_infos.append({
            "path": relpath,
            "title": parts["title"],
            "length": len(text_norm),
            "hash": content_hash,
        })
        corpus.append(text_norm)
        embed_texts.append(embed_text)

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

    # 任意: 事前埋め込みの計算（外部API不使用。ローカル/指定モデルのみ）
    if embed_model:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise SystemExit("sentence-transformers is required for --embed-model. Install via: pip install sentence-transformers")

        model = SentenceTransformer(embed_model)

        # 文書をチャンク化してまとめて埋め込み
        all_chunk_texts: List[str] = []
        all_chunk_meta: List[Dict[str, Any]] = []
        doc_chunk_index: Dict[str, List[int]] = {}

        for doc_id, text in enumerate(embed_texts):
            start_idx = len(all_chunk_texts)
            chunks = _chunk_text(text, embed_max_chars, embed_overlap)
            for s, e, t in chunks:
                all_chunk_texts.append(t)
                all_chunk_meta.append({
                    "doc_id": doc_id,
                    "relpath": doc_infos[doc_id]["path"],
                    "title": doc_infos[doc_id].get("title", ""),
                    "start": s,
                    "end": e,
                })
            end_idx = len(all_chunk_texts)
            doc_chunk_index[str(doc_id)] = [start_idx, end_idx]

        if not all_chunk_texts:
            return

        emb = model.encode(
            all_chunk_texts,
            batch_size=embed_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if emb.dtype != np.float32:
            emb = emb.astype(np.float32)

        np.save(out_dir / "embeddings.npy", emb)
        (out_dir / "emb_chunks.jsonl").write_text(
            "\n".join(
                json.dumps({"chunk_id": i, **all_chunk_meta[i]}, ensure_ascii=False)
                for i in range(len(all_chunk_meta))
            ),
            encoding="utf-8",
        )
        (out_dir / "emb_doc_index.json").write_text(
            json.dumps(doc_chunk_index, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (out_dir / "emb_model.txt").write_text(str(embed_model), encoding="utf-8")
        print(f"Precomputed embeddings for {len(all_chunk_texts)} chunks using {embed_model}")

        # HNSW support removed


def main():
    ap = argparse.ArgumentParser(description="Build TF-IDF index from HTML files (Japanese)")
    ap.add_argument("--src", type=Path, required=True, help="HTML root directory")
    ap.add_argument("--out", type=Path, required=True, help="Output index directory")
    ap.add_argument("--ngram", type=int, default=3, help="character n-gram size (default: 3)")
    ap.add_argument("--min-df", type=int, default=2, help="min_df for TF-IDF (default: 2)")
    ap.add_argument("--max-df", type=float, default=0.95, help="max_df for TF-IDF (default: 0.95)")
    ap.add_argument("--title-weight", type=int, default=3, help="title weight (default: 3)")
    ap.add_argument("--heading-weight", type=int, default=2, help="headings weight (default: 2)")
    # 事前埋め込み関連（任意）
    ap.add_argument("--embed-model", type=str, default="", help="Sentence-Transformers model name or local path to precompute embeddings (optional)")
    ap.add_argument("--embed-max-chars", type=int, default=800, help="Max characters per chunk for embeddings (default: 800)")
    ap.add_argument("--embed-overlap", type=int, default=200, help="Overlap characters between chunks (default: 200)")
    ap.add_argument("--embed-batch-size", type=int, default=32, help="Batch size for embedding encoding (default: 32)")
    # HNSW options removed
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
        embed_model=args.embed_model,
        embed_max_chars=args.embed_max_chars,
        embed_overlap=args.embed_overlap,
        embed_batch_size=args.embed_batch_size,
    )


if __name__ == "__main__":
    main()
