"""
score_related.py（単一クエリモード）を呼び出すクロスプラットフォーム用ラッパー。

- ./index に事前計算済み埋め込みがあればそれを使用
- 無い場合は ./.models/sentence-bert-base-ja-mean-tokens-v2 をローカルに要求
"""

import sys
from pathlib import Path
from types import SimpleNamespace as NS

import score_related as sr


def main() -> None:
    root = Path(__file__).resolve().parent
    index_dir = root / "index"

    # 注意: 実際に検索するクエリ HTML へのパスに必要に応じて変更してください
    query_html = root / "html" / "ABC.html"

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
        topk=30,
        tau=0.05,
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
    )

    results = sr._score_single_query(args, vectorizer, X, id_map, src_root, drop_selectors)
    sr._print_results(results, args.format)


if __name__ == "__main__":
    main()
