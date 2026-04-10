"""
score_related.py（単一クエリモード）を呼び出すクロスプラットフォーム用ラッパー。

- ./index に事前計算済み埋め込みがあればそれを使用
- 無い場合は ./.models/sentence-bert-base-ja-mean-tokens-v2 をローカルに要求
"""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import score_related as sr


def main() -> None:
    ap = argparse.ArgumentParser(description="Search related pages using project defaults")
    ap.add_argument("query", nargs="?", help="Query HTML path. Relative paths are resolved from the repo root.")
    ap.add_argument("--index", type=Path, help="Index directory (default: ./index)")
    ap.add_argument("--topk", type=int, default=30, help="Top K results (default: 30)")
    ap.add_argument("--tau", type=float, default=0.05, help="Threshold (default: 0.05)")
    cli_args = ap.parse_args()

    root = Path(__file__).resolve().parent
    index_dir = cli_args.index if cli_args.index else (root / "index")

    if not cli_args.query:
        print(
            "クエリHTMLを指定してください。例:\n"
            "  python .\\score_related_cli.py .\\html\\page-bushou-下間真頼.html",
            file=sys.stderr,
        )
        raise SystemExit(2)

    query_html = Path(cli_args.query)
    if not query_html.is_absolute():
        query_html = root / query_html
    if not query_html.exists():
        print(f"[エラー] クエリHTMLが見つかりません: \"{query_html}\"", file=sys.stderr)
        raise SystemExit(1)

    model_path = root / ".models" / "sentence-bert-base-ja-mean-tokens-v2"

    # リランク設定（.bat の既定に合わせる）
    rerank_mode = "hybrid"
    rerank_topk = 200
    alpha = 0.5

    # 事前計算済み埋め込みがあればそれを利用。無ければローカルモデルを必須とする。
    emb_path = index_dir / "embeddings.npy"
    if emb_path.exists():
        print(f"事前計算済みの埋め込みを利用します: \"{index_dir}\"")
        rerank_model = ""  # インデックス内の情報からモデル名を特定させる
    else:
        if not model_path.exists():
            print((
                f"[エラー] \"{index_dir}\" に事前計算済み埋め込みが見つからず、モデルパスも存在しません: \"{model_path}\"。\n"
                f"        先に build_index_cli.py を実行するか、ローカルにモデルを配置してこのファイル内の model_path を調整してください。"
            ), file=sys.stderr)
            raise SystemExit(1)
        rerank_model = str(model_path)

    # インデックスを読み込み、score_related の内部関数で単一クエリ検索を実行
    vectorizer, X, id_map, src_root = sr.load_index(index_dir)

    drop_selectors = [
        "nav",
        "footer",
        "header",
        "aside",
        ".sidebar",
        ".breadcrumbs",
        ".breadcrumb",
        ".global-nav",
        ".site-footer",
    ]

    args = NS(
        index=index_dir,
        query=query_html,
        all=False,
        out_dir=root / "json",
        topk=cli_args.topk,
        tau=cli_args.tau,
        format="table",
        drop_selectors=",".join(drop_selectors),
        title_weight=3,
        heading_weight=2,
        rerank_mode=rerank_mode,
        rerank_model=rerank_model,
        rerank_topk=rerank_topk,
        alpha=alpha,
        embed_max_chars=800,
        embed_overlap=200,
        neighbor_k=20,
        context_weight=0.18,
        graph_weight=0.10,
    )

    doc_profiles = sr.load_doc_profiles(index_dir, id_map, src_root, drop_selectors)
    results = sr._score_single_query(args, vectorizer, X, id_map, src_root, drop_selectors, doc_profiles, {})
    sr._print_results(results, args.format)


if __name__ == "__main__":
    main()
