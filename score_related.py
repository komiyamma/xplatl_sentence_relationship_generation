"""
SPDX-License-Identifier: MIT
Copyright (c) 2025 Akitsugu Komiyama
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

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


def _chunk_text(text: str, max_chars: int, overlap: int) -> List[Tuple[int, int, str]]:
    if max_chars <= 0:
        return [(0, len(text), text)]
    n = len(text)
    if n <= max_chars:
        return [(0, n, text)]
    chunks: List[Tuple[int, int, str]] = []
    start = 0
    while start < n:
        end = min(n, start + max_chars)
        chunks.append((start, end, text[start:end]))
        if end >= n:
            break
        start = end - max(0, overlap)
    return chunks


def _load_precomputed_embeddings(index_dir: Path):
    emb_path = index_dir / "embeddings.npy"
    chunks_path = index_dir / "emb_chunks.jsonl"
    doc_index_path = index_dir / "emb_doc_index.json"
    model_info_path = index_dir / "emb_model.txt"
    if not (emb_path.exists() and chunks_path.exists() and doc_index_path.exists() and model_info_path.exists()):
        return None
    emb = np.load(emb_path)
    # chunks metadata is not strictly needed for scoring, but keep for completeness
    # Build doc index mapping
    doc_index = json.loads(doc_index_path.read_text(encoding="utf-8"))
    model_name = model_info_path.read_text(encoding="utf-8").strip()
    return {
        "emb": emb,
        "doc_index": {int(k): v for k, v in doc_index.items()},
        "model": model_name,
    }


def _load_hnsw_index(index_dir: Path):
    """Load HNSW index if present. Returns dict or None.
    Requires hnswlib installed. Uses cosine space; distances -> cosine distance (1 - sim).
    """
    try:
        import hnswlib  # type: ignore
    except Exception:
        return None
    idx_path = index_dir / "emb_hnsw.bin"
    meta_path = index_dir / "emb_hnsw_meta.json"
    doc_index_path = index_dir / "emb_doc_index.json"
    if not (idx_path.exists() and meta_path.exists() and doc_index_path.exists()):
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    hidx = hnswlib.Index(space=str(meta.get("space", "cosine")), dim=int(meta["dim"]))
    hidx.load_index(str(idx_path))
    if "ef_search_default" in meta:
        try:
            hidx.set_ef(int(meta["ef_search_default"]))
        except Exception:
            pass
    # need doc_index to map chunk -> doc
    doc_index = json.loads(doc_index_path.read_text(encoding="utf-8"))
    return {"index": hidx, "meta": meta, "doc_index": {int(k): v for k, v in doc_index.items()}}


def _embed_texts(model, texts: List[str], batch_size: int = 32) -> np.ndarray:
    emb = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    if emb.dtype != np.float32:
        emb = emb.astype(np.float32)
    return emb


def _minmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx - mn < 1e-9:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


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
    # Rerank options (all local; no external APIs)
    ap.add_argument("--rerank-mode", type=str, default="none", choices=["none", "embed", "hybrid"], help="Rerank using embeddings or hybrid with TF-IDF")
    ap.add_argument("--rerank-model", type=str, default="", help="Sentence-Transformers model name or local path. If empty and precomputed exists, use that model.")
    ap.add_argument("--rerank-topk", type=int, default=200, help="Number of TF-IDF candidates to rerank (default: 200)")
    ap.add_argument("--alpha", type=float, default=0.5, help="Hybrid weight: alpha*tfidf + (1-alpha)*embed")
    ap.add_argument("--embed-max-chars", type=int, default=800, help="Max characters per chunk for embeddings (default: 800)")
    ap.add_argument("--embed-overlap", type=int, default=200, help="Overlap characters between chunks (default: 200)")
    # ANN options (auto-detect HNSW if present)
    ap.add_argument("--ann-mode", type=str, default="auto", choices=["auto", "none", "hnsw"], help="Approximate NN backend usage")
    ap.add_argument("--hnsw-ef", type=int, default=0, help="Override HNSW efSearch during query (0=use stored)")
    ap.add_argument("--ann-topk-mult", type=int, default=5, help="Multiplier for ANN chunk neighbors relative to rerank_topk")

    args = ap.parse_args()
    vectorizer, X, id_map, src_root = load_index(args.index)

    drop_selectors = [s.strip() for s in args.drop_selectors.split(",") if s.strip()]
    qvec = build_query_vector(args.query, drop_selectors, args.title_weight, args.heading_weight, vectorizer)

    # 類似度: 1 × N（TF-IDF）
    sims = cosine_similarity(qvec, X).ravel()
    order = np.argsort(-sims)

    # Rerank candidates selection
    cand_k = max(args.rerank_topk, args.topk * 5)
    cand_indices = order[:cand_k]

    final_scores = None

    if args.rerank_mode != "none":
        # Try precomputed embeddings
        pre = _load_precomputed_embeddings(args.index)
        model_name = args.rerank_model
        use_pre = pre is not None and (not model_name or model_name.strip() == pre["model"])  # prefer pre if model not forced

        # Load model (needed at least for query embedding)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise SystemExit("sentence-transformers is required for rerank. Install via: pip install sentence-transformers")

        if use_pre:
            model = SentenceTransformer(pre["model"])  # local cache expected
        else:
            if not model_name:
                raise SystemExit("--rerank-model is required if no precomputed embeddings are found")
            model = SentenceTransformer(model_name)

        # Build query embedding (chunked)
        raw_q = args.query.read_text(encoding="utf-8", errors="ignore")
        parts_q = extract_text_from_html(raw_q, [s.strip() for s in args.drop_selectors.split(",") if s.strip()])
        q_text = normalize_text((parts_q["title"] + " ") + (parts_q["headings"] + " ") + parts_q["body"])
        q_chunks = _chunk_text(q_text, args.embed_max_chars, args.embed_overlap)
        q_emb = _embed_texts(model, [t for _, _, t in q_chunks])

        # Optionally use ANN(HNSW) if available/selected to obtain doc-level embedding scores
        ann_info = None
        if args.ann_mode != "none":
            ann_info = _load_hnsw_index(args.index)
            if args.ann_mode == "hnsw" and ann_info is None:
                raise SystemExit("HNSW index not found. Build with build_index.py --build-hnsw")

        emb_scores = None
        emb_doc_ids = None

        if ann_info is not None and pre is not None:
            # Build inverse map: chunk_id -> doc_id
            # pre['doc_index']: dict[int, [start, end)]
            max_end = max(int(v[1]) for v in pre["doc_index"].values()) if pre["doc_index"] else 0
            chunk_to_doc = np.empty(max_end, dtype=np.int32)
            for d_id, (s, e) in pre["doc_index"].items():
                chunk_to_doc[int(s):int(e)] = int(d_id)

            hidx = ann_info["index"]
            if int(args.hnsw_ef) > 0:
                try:
                    hidx.set_ef(int(args.hnsw_ef))
                except Exception:
                    pass
            # k: request more chunk neighbors than needed and aggregate per doc (take max)
            k = max(int(args.rerank_topk) * int(args.ann_topk_mult), int(args.topk) * 10)
            doc_scores: Dict[int, float] = {}
            for v in q_emb:
                labels, dists = hidx.knn_query(v, k=k)
                labs = labels[0]
                dst = dists[0]
                # cosine distance -> similarity (avoid shadowing TF-IDF sims)
                chunk_sims = 1.0 - dst
                for lab, sim in zip(labs, chunk_sims):
                    if int(lab) < 0 or int(lab) >= len(chunk_to_doc):
                        continue
                    did = int(chunk_to_doc[int(lab)])
                    prev = doc_scores.get(did, 0.0)
                    if float(sim) > prev:
                        doc_scores[did] = float(sim)
            if doc_scores:
                emb_doc_ids = np.array(list(doc_scores.keys()), dtype=np.int32)
                emb_scores = np.array([doc_scores[int(d)] for d in emb_doc_ids], dtype=np.float32)

        if emb_scores is None:
            # Fallback: direct scoring against candidate docs (original behavior)
            emb_scores = np.zeros(len(cand_indices), dtype=np.float32)
            emb_doc_ids = cand_indices.astype(np.int32)
            if use_pre:
                emb_all = pre["emb"]  # shape: (C, D), normalized
                doc_index = pre["doc_index"]  # doc_id -> [start, end]
                for i, idx in enumerate(emb_doc_ids):
                    if int(idx) not in doc_index:
                        emb_scores[i] = 0.0
                        continue
                    s, e = doc_index[int(idx)]
                    doc_emb = emb_all[int(s):int(e)]  # (m, D)
                    sim = np.max(q_emb @ doc_emb.T) if doc_emb.size and q_emb.size else 0.0
                    emb_scores[i] = float(sim)
            else:
                # On-the-fly encode candidate docs
                for i, idx in enumerate(emb_doc_ids):
                    meta = id_map[str(int(idx))]
                    p = Path(src_root) / meta["path"]
                    raw = p.read_text(encoding="utf-8", errors="ignore")
                    parts = extract_text_from_html(raw, [s.strip() for s in args.drop_selectors.split(",") if s.strip()])
                    d_text = normalize_text((parts["title"] + " ") + (parts["headings"] + " ") + parts["body"]) 
                    d_chunks = _chunk_text(d_text, args.embed_max_chars, args.embed_overlap)
                    if not d_chunks:
                        emb_scores[i] = 0.0
                        continue
                    d_emb = _embed_texts(model, [t for _, _, t in d_chunks])
                    sim = np.max(q_emb @ d_emb.T) if d_emb.size and q_emb.size else 0.0
                    emb_scores[i] = float(sim)

        # Combine with TF-IDF
        # Build final doc list depending on rerank mode and source of emb_scores
        if emb_doc_ids is None or np.array_equal(emb_doc_ids, cand_indices.astype(np.int32)):
            # Same set as TF-IDF candidates (or no ANN). Keep existing pathway.
            tfidf_scores = sims[cand_indices]
            if args.rerank_mode == "embed":
                fused = emb_scores
            elif args.rerank_mode == "hybrid":
                fused = args.alpha * _minmax(tfidf_scores) + (1.0 - args.alpha) * _minmax(emb_scores)
            else:
                fused = tfidf_scores
            sort_idx = np.argsort(-fused)
            cand_indices = cand_indices[sort_idx]
            final_scores = fused[sort_idx]
        else:
            # emb_doc_ids are from ANN (union path): fuse with TF-IDF by union of doc IDs
            tfidf_map = {int(idx): float(sims[int(idx)]) for idx in cand_indices}
            docs_union = np.array(sorted(set(tfidf_map.keys()) | set(int(d) for d in emb_doc_ids)), dtype=np.int32)
            tfidf_scores = np.array([tfidf_map.get(int(d), 0.0) for d in docs_union], dtype=np.float32)
            emb_map = {int(d): float(s) for d, s in zip(emb_doc_ids, emb_scores)}
            emb_scores_u = np.array([emb_map.get(int(d), 0.0) for d in docs_union], dtype=np.float32)
            if args.rerank_mode == "embed":
                fused = emb_scores_u
            elif args.rerank_mode == "hybrid":
                fused = args.alpha * _minmax(tfidf_scores) + (1.0 - args.alpha) * _minmax(emb_scores_u)
            else:
                fused = tfidf_scores
            sort_idx = np.argsort(-fused)
            cand_indices = docs_union[sort_idx]
            final_scores = fused[sort_idx]
    else:
        final_scores = sims[cand_indices]

    # Build results with final scores and tau filtering
    results = []
    for rank, (idx, score) in enumerate(zip(cand_indices, final_scores), start=1):
        if float(score) < args.tau:
            continue
        meta = id_map[str(int(idx))]
        results.append({
            "rank": len(results) + 1,
            "score": round(float(score), 4),
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
