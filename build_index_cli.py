"""
build_index.py を既定値付きで呼び出すクロスプラットフォーム用ラッパー。
"""

import argparse
from pathlib import Path

import build_index as bi


def main() -> None:
    ap = argparse.ArgumentParser(description="Build index using project defaults")
    ap.add_argument("--src", type=Path, help="HTML source directory (default: ./html)")
    ap.add_argument("--out", type=Path, help="Index output directory (default: ./index)")
    ap.add_argument(
        "--embed-model",
        type=Path,
        help="Local embedding model path. If omitted, use ./.models/... when present, otherwise TF-IDF only.",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parent

    src = args.src if args.src else (root / "html")
    out = args.out if args.out else (root / "index")

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
    default_model_path = root / ".models" / "sentence-bert-base-ja-mean-tokens-v2"
    model_path = args.embed_model if args.embed_model else default_model_path
    embed_model = str(model_path) if model_path.exists() else ""
    embed_max_chars = 800
    embed_overlap = 200
    embed_batch_size = 32

    if embed_model:
        print(f"埋め込み事前計算を有効にします: \"{embed_model}\"")
    else:
        print("埋め込みモデルが見つからないため、TF-IDF のみで index を構築します。")

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
