"""
build_index.py を既定値付きで呼び出すクロスプラットフォーム用ラッパー。
"""

from pathlib import Path

import build_index as bi


def main() -> None:
    root = Path(__file__).resolve().parent

    src = root / "html"
    out = root / "index"

    # .bat に合わせた既定値
    ngram = 3
    min_df = 2
    max_df = 0.95
    title_weight = 3
    heading_weight = 2
    drop_selectors = [
        "nav",
        "footer",
        "header",
        "aside",
        ".sidebar",
        ".breadcrumbs",
    ]

    # 事前埋め込みを計算するためのローカルモデル（任意）
    embed_model = str(root / ".models" / "sentence-bert-base-ja-mean-tokens-v2")
    embed_max_chars = 800
    embed_overlap = 200
    embed_batch_size = 32

    bi.build_index(
        src_dir=src,
        out_dir=out,
        ngram=ngram,
        min_df=min_df,
        max_df=max_df,
        title_weight=title_weight,
        heading_weight=heading_weight,
        drop_selectors=drop_selectors,
        embed_model=embed_model,
        embed_max_chars=embed_max_chars,
        embed_overlap=embed_overlap,
        embed_batch_size=embed_batch_size,
    )


if __name__ == "__main__":
    main()
