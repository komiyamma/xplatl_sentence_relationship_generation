"""
SPDX-License-Identifier: MIT
Copyright (c) 2025 Akitsugu Komiyama
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import joblib
import numpy as np
from bs4 import BeautifulSoup
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

# 取り回しのため、build_index.py の一部関数をこちらにも軽量実装
import re
import unicodedata

def normalize_text(s: str) -> str:
    """日本語向けの軽い正規化（NFKC・URL除去・空白圧縮）。"""
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"https?://\S+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def extract_text_from_html(html: str, drop_selectors: List[str]) -> Dict[str, str]:
    """HTMLからタイトル・見出し・本文を抽出（指定セレクタは除去）。"""
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
    """build_index.py が出力したインデックス一式を読み込む。"""
    vectorizer = joblib.load(index_dir / "tfidf_vectorizer.pkl")
    X = sparse.load_npz(index_dir / "tfidf_matrix.npz")
    id_map = json.loads((index_dir / "id_map.json").read_text(encoding="utf-8"))
    src_root = (index_dir / "source_root.txt").read_text(encoding="utf-8").strip()
    return vectorizer, X, id_map, Path(src_root)


def build_query_vector(query_html_path: Path, drop_selectors: List[str], title_weight: int, heading_weight: int, vectorizer) -> sparse.csr_matrix:
    """単一HTMLファイルからクエリベクトル（TF‑IDF）を構築する。"""
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
    """テキストを重なり付きでチャンク分割する。戻り値は (開始, 終了, 文字列)。"""
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
    """インデックス側で事前計算された埋め込みとメタ情報を読み込む。"""
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


def _embed_texts(model, texts: List[str], batch_size: int = 32) -> np.ndarray:
    """Sentence-Transformers の `encode` をラップ（numpy/正規化込み）。"""
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
    """配列を min-max 正規化（定数配列はゼロ配列）。"""
    if x.size == 0:
        return x
    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx - mn < 1e-9:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def _select_candidates_from_sims(sims: np.ndarray, topk: int, rerank_topk: int) -> np.ndarray:
    """TF‑IDF類似度ベクトルから上位候補のインデックスを返す。"""
    order = np.argsort(-sims)
    cand_k = max(int(rerank_topk), int(topk) * 5)
    return order[:cand_k]


def _build_q_emb_for_doc(doc_id: int,
                         pre: Optional[Dict[str, Any]],
                         model,
                         id_map: Dict[str, Any],
                         src_root: Path,
                         drop_selectors: List[str],
                         embed_max_chars: int,
                         embed_overlap: int) -> np.ndarray:
    """`--all` 用: 文書IDに対応するクエリのチャンク埋め込みを取得/計算。"""
    if pre is not None and int(doc_id) in pre.get("doc_index", {}):
        s0, e0 = pre["doc_index"][int(doc_id)]
        return pre["emb"][int(s0):int(e0)]
    # fallback: compute from source HTML only if model is provided
    p = Path(src_root) / id_map[str(int(doc_id))]["path"]
    raw_q = p.read_text(encoding="utf-8", errors="ignore")
    parts_q = extract_text_from_html(raw_q, drop_selectors)
    q_text = normalize_text((parts_q["title"] + " ") + (parts_q["headings"] + " ") + parts_q["body"])
    q_chunks = _chunk_text(q_text, embed_max_chars, embed_overlap)
    if model is None or not q_chunks:
        return np.zeros((0, 0), dtype=np.float32)
    return _embed_texts(model, [t for _, _, t in q_chunks])


def _compute_emb_scores_for_candidates(q_emb: np.ndarray,
                                       cand_indices: np.ndarray,
                                       pre: Optional[Dict[str, Any]],
                                       model,
                                       id_map: Dict[str, Any],
                                       src_root: Path,
                                       drop_selectors: List[str],
                                       embed_max_chars: int,
                                       embed_overlap: int,
                                       exclude_doc_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """候補文書集合に対し、最大チャンク類似（内積）でスコアを計算する。"""
    emb_doc_ids = cand_indices.astype(np.int32)
    emb_scores = np.zeros(len(emb_doc_ids), dtype=np.float32)

    if q_emb.size == 0:
        return emb_doc_ids, emb_scores

    if pre is not None:
        emb_all = pre["emb"]
        doc_index = pre["doc_index"]
        for i2, idx in enumerate(emb_doc_ids):
            if int(idx) == int(exclude_doc_id):
                emb_scores[i2] = 0.0
                continue
            if int(idx) not in doc_index:
                emb_scores[i2] = 0.0
                continue
            s1, e1 = doc_index[int(idx)]
            doc_emb = emb_all[int(s1):int(e1)]
            sim = np.max(q_emb @ doc_emb.T) if doc_emb.size and q_emb.size else 0.0
            emb_scores[i2] = float(sim)
        return emb_doc_ids, emb_scores

    # no precomputed embeddings; compute on the fly if model provided
    if model is None:
        return emb_doc_ids, emb_scores
    for i2, idx in enumerate(emb_doc_ids):
        if int(idx) == int(exclude_doc_id):
            emb_scores[i2] = 0.0
            continue
        meta2 = id_map[str(int(idx))]
        p2 = Path(src_root) / meta2["path"]
        raw2 = p2.read_text(encoding="utf-8", errors="ignore")
        parts2 = extract_text_from_html(raw2, drop_selectors)
        d_text = normalize_text((parts2["title"] + " ") + (parts2["headings"] + " ") + parts2["body"])
        d_chunks = _chunk_text(d_text, embed_max_chars, embed_overlap)
        if not d_chunks:
            emb_scores[i2] = 0.0
            continue
        d_emb = _embed_texts(model, [t for _, _, t in d_chunks])
        sim = np.max(q_emb @ d_emb.T) if d_emb.size and q_emb.size else 0.0
        emb_scores[i2] = float(sim)
    return emb_doc_ids, emb_scores


def _fuse_and_sort_scores(cand_indices: np.ndarray,
                          sims: np.ndarray,
                          emb_doc_ids: np.ndarray,
                          emb_scores: np.ndarray,
                          rerank_mode: str,
                          alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """TF‑IDFと埋め込みスコアを融合し、降順にソートして返す。"""
    tfidf_scores = sims[cand_indices]
    if rerank_mode == "embed":
        fused = emb_scores
    elif rerank_mode == "hybrid":
        fused = alpha * _minmax(tfidf_scores) + (1.0 - alpha) * _minmax(emb_scores)
    else:
        fused = tfidf_scores
    sort_idx = np.argsort(-fused)
    return cand_indices[sort_idx], fused[sort_idx]


def _format_top_results(cand_indices: np.ndarray,
                        final_scores: np.ndarray,
                        id_map: Dict[str, Any],
                        src_root: Path,
                        topk: int,
                        tau: float,
                        exclude_doc_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """スコアとメタを整形してトップKの結果を返す。"""
    results: List[Dict[str, Any]] = []
    for idx, score in zip(cand_indices, final_scores):
        if exclude_doc_id is not None and int(idx) == int(exclude_doc_id):
            continue
        if float(score) < float(tau):
            continue
        meta = id_map[str(int(idx))]
        results.append({
            "rank": len(results) + 1,
            "score": round(float(score), 4),
            "path": str(Path(src_root) / meta["path"]),
            "relpath": meta["path"],
            "title": meta.get("title", ""),
        })
        if len(results) >= int(topk):
            break
    return results


def _prepare_rerank_for_all(index_dir: Path, rerank_mode: str, rerank_model: str) -> Tuple[Optional[Dict[str, Any]], Optional[object]]:
    """`--all` 用のリランク前準備（プリコンピュートとモデル読み込み）。"""
    if rerank_mode == "none":
        return None, None
    pre = _load_precomputed_embeddings(index_dir)
    model = None
    model_name = rerank_model
    use_pre = pre is not None and (not model_name or model_name.strip() == pre["model"])  # noqa: F841
    if not use_pre:
        if not model_name:
            raise SystemExit("--rerank-model is required for --all when no precomputed embeddings are found")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise SystemExit("sentence-transformers is required for rerank. Install via: pip install sentence-transformers")
        model = SentenceTransformer(model_name)
    return pre, model


def _prepare_rerank_for_single(index_dir: Path, rerank_mode: str, rerank_model: str) -> Tuple[Optional[Dict[str, Any]], object]:
    """単一クエリ用のリランク前準備（プリコンピュート/モデル）。"""
    if rerank_mode == "none":
        return None, None  # type: ignore
    pre = _load_precomputed_embeddings(index_dir)
    model_name = rerank_model
    use_pre = pre is not None and (not model_name or model_name.strip() == pre["model"])  # noqa: F841
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise SystemExit("sentence-transformers is required for rerank. Install via: pip install sentence-transformers")
    model = SentenceTransformer(pre["model"]) if pre is not None and (not model_name or model_name.strip() == pre["model"]) else (
        SentenceTransformer(model_name) if model_name else (_ for _ in ()).throw(SystemExit("--rerank-model is required if no precomputed embeddings are found"))
    )
    return pre, model


def _build_q_emb_from_html(query_html_path: Path,
                           drop_selectors: List[str],
                           model,
                           embed_max_chars: int,
                           embed_overlap: int) -> np.ndarray:
    """HTMLからクエリのチャンク埋め込みを生成。"""
    raw_q = query_html_path.read_text(encoding="utf-8", errors="ignore")
    parts_q = extract_text_from_html(raw_q, drop_selectors)
    q_text = normalize_text((parts_q["title"] + " ") + (parts_q["headings"] + " ") + parts_q["body"])
    q_chunks = _chunk_text(q_text, embed_max_chars, embed_overlap)
    return _embed_texts(model, [t for _, _, t in q_chunks]) if q_chunks else np.zeros((0, 0), dtype=np.float32)


def compute_results_for_doc(doc_id: int,
                            X: sparse.csr_matrix,
                            id_map: Dict[str, Any],
                            src_root: Path,
                            drop_selectors: List[str],
                            rerank_mode: str,
                            rerank_topk: int,
                            topk: int,
                            embed_max_chars: int,
                            embed_overlap: int,
                            alpha: float,
                            tau: float,
                            pre: Optional[Dict[str, Any]] = None,
                            model: Optional[object] = None) -> List[Dict[str, Any]]:
    """`--all` の1ドキュメントに対する検索と整形済み結果の取得。"""
    qvec = X[doc_id:doc_id+1]
    sims = cosine_similarity(qvec, X).ravel()
    sims[int(doc_id)] = -1.0
    cand_indices = _select_candidates_from_sims(sims, int(topk), int(rerank_topk))
    if rerank_mode != "none":
        q_emb = _build_q_emb_for_doc(
            int(doc_id), pre, model, id_map, src_root, drop_selectors, int(embed_max_chars), int(embed_overlap)
        )
        emb_doc_ids, emb_scores = _compute_emb_scores_for_candidates(
            q_emb, cand_indices, pre, model, id_map, src_root, drop_selectors, int(embed_max_chars), int(embed_overlap), int(doc_id)
        )
        cand_indices, final_scores = _fuse_and_sort_scores(cand_indices, sims, emb_doc_ids, emb_scores, str(rerank_mode), float(alpha))
    else:
        final_scores = sims[cand_indices]
    return _format_top_results(cand_indices, final_scores, id_map, src_root, int(topk), float(tau), exclude_doc_id=int(doc_id))


def _score_single_query(args,
                        vectorizer,
                        X: sparse.csr_matrix,
                        id_map: Dict[str, Any],
                        src_root: Path,
                        drop_selectors: List[str]) -> List[Dict[str, Any]]:
    """単一クエリ（--query）に対する検索と整形済み結果の取得。"""
    qvec = build_query_vector(args.query, drop_selectors, args.title_weight, args.heading_weight, vectorizer)
    sims = cosine_similarity(qvec, X).ravel()
    cand_indices = _select_candidates_from_sims(sims, int(args.topk), int(args.rerank_topk))
    if args.rerank_mode != "none":
        pre, model = _prepare_rerank_for_single(args.index, args.rerank_mode, args.rerank_model)
        q_emb = _build_q_emb_from_html(args.query, drop_selectors, model, int(args.embed_max_chars), int(args.embed_overlap))
        emb_doc_ids, emb_scores = _compute_emb_scores_for_candidates(
            q_emb, cand_indices, pre, model, id_map, src_root, drop_selectors, int(args.embed_max_chars), int(args.embed_overlap), exclude_doc_id=-1
        )
        cand_indices, final_scores = _fuse_and_sort_scores(cand_indices, sims, emb_doc_ids, emb_scores, str(args.rerank_mode), float(args.alpha))
    else:
        final_scores = sims[cand_indices]
    return _format_top_results(cand_indices, final_scores, id_map, src_root, int(args.topk), float(args.tau))


def _print_results(results: List[Dict[str, Any]], fmt: str) -> None:
    """結果を JSON またはテーブルで出力。"""
    if fmt == "json":
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return
    if not results:
        print("No results above tau. Try lowering --tau.")
        return
    colw = {"rank": 4, "score": 6, "relpath": 50, "title": 40}
    print(f"{'#':>{colw['rank']}}  {'score':>{colw['score']}}  {'relpath':<{colw['relpath']}}  {'title':<{colw['title']}}")
    print("-" * (sum(colw.values()) + 8))
    for r in results:
        print(f"{r['rank']:>{colw['rank']}}  {r['score']:>{colw['score']}.4f}  {r['relpath']:<{colw['relpath']}}  {r['title']:<{colw['title']}}")


def main():
    ap = argparse.ArgumentParser(description="Score related HTML files against a query HTML (Japanese)")
    ap.add_argument("--index", type=Path, required=True, help="Index directory from build_index.py")
    ap.add_argument("--query", type=Path, help="Query HTML file (A.html)")
    ap.add_argument("--all", action="store_true", help="Process all HTML files under index's source_root and output per-file JSON to --out-dir")
    ap.add_argument("--out-dir", type=Path, default=Path("json"), help="Output dir for --all mode (default: ./json)")
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

    args = ap.parse_args()
    vectorizer, X, id_map, src_root = load_index(args.index)
    drop_selectors = [s.strip() for s in args.drop_selectors.split(",") if s.strip()]

    if args.all:
        out_dir: Path = args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        pre, model = _prepare_rerank_for_all(args.index, args.rerank_mode, args.rerank_model)

        for sid in range(len(id_map)):
            meta_q = id_map[str(int(sid))]
            rel = Path(meta_q["path"])  # may include subdirs
            out_path = out_dir / rel.with_suffix(".json")
            out_path.parent.mkdir(parents=True, exist_ok=True)

            results = compute_results_for_doc(
                int(sid), X, id_map, src_root, drop_selectors,
                args.rerank_mode, args.rerank_topk, args.topk,
                args.embed_max_chars, args.embed_overlap, args.alpha, args.tau,
                pre, model
            )
            out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    # Single query mode
    if not args.query:
        raise SystemExit("--query is required unless --all is specified")
    results = _score_single_query(args, vectorizer, X, id_map, src_root, drop_selectors)
    _print_results(results, args.format)


if __name__ == "__main__":
    main()
