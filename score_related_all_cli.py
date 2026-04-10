"""
score_related.py（--all バッチ JSON 出力）を呼び出すクロスプラットフォーム用ラッパー。

- index の source_root 配下の全ドキュメントを処理
- 各ドキュメントの結果を ./json 配下に（ディレクトリ構造を保ったまま）出力
- 事前計算済み埋め込みがあればそれを利用、無ければローカルモデルを要求
"""

import argparse
import sys
import json
from pathlib import Path

import score_related as sr


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch-export related pages JSON using project defaults")
    ap.add_argument("--index", type=Path, help="Index directory (default: ./index)")
    ap.add_argument("--out-dir", type=Path, help="Output directory (default: ./json)")
    ap.add_argument("--topk", type=int, default=30, help="Top K results per document (default: 30)")
    ap.add_argument("--tau", type=float, default=0.05, help="Threshold (default: 0.05)")
    cli_args = ap.parse_args()

    root = Path(__file__).resolve().parent
    index_dir = cli_args.index if cli_args.index else (root / "index")
    out_dir = cli_args.out_dir if cli_args.out_dir else (root / "json")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = root / ".models" / "sentence-bert-base-ja-mean-tokens-v2"

    # リランク設定（.bat の既定に合わせる）
    rerank_mode = "hybrid"
    rerank_topk = 200
    alpha = 0.5

    # 事前計算済み埋め込みの有無を確認。無い場合はローカルモデルの存在を確認。
    emb_path = index_dir / "embeddings.npy"
    if emb_path.exists():
        print(f"事前計算済みの埋め込みを利用します: \"{index_dir}\"")
        rerank_model = ""
    else:
        if not model_path.exists():
            print((
                f"[エラー] \"{index_dir}\" に事前計算済み埋め込みが見つからず、モデルパスも存在しません: \"{model_path}\"。\n"
                f"        先に build_index_cli.py を実行するか、ローカルにモデルを配置してこのファイル内の model_path を調整してください。"
            ), file=sys.stderr)
            raise SystemExit(1)
        rerank_model = str(model_path)

    # インデックスの読み込み
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

    # リランク用バックエンドの準備（事前計算済みがありモデル未指定ならそれを使用）
    pre, model = sr._prepare_rerank_for_all(index_dir, rerank_mode, rerank_model)
    doc_profiles = sr.load_doc_profiles(index_dir, id_map, src_root, drop_selectors)
    neighbor_cache = {}

    # インデックス内の全ドキュメントを処理
    for sid in range(len(id_map)):
        meta_q = id_map[str(int(sid))]
        rel = Path(meta_q["path"])  # may include subdirs
        out_path = out_dir / rel.with_suffix(".json")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        results = sr.compute_results_for_doc(
            int(sid), X, id_map, src_root, drop_selectors,
            rerank_mode, rerank_topk, cli_args.topk,
            800, 200, alpha, cli_args.tau,
            doc_profiles, neighbor_cache, 20, 0.18, 0.10,
            pre, model,
        )
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
