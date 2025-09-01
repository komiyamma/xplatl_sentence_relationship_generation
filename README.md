[![GitHub release (latest by date)](https://img.shields.io/github/v/release/komiyamma/xplatl_sentence_relationship_generation)](https://github.com/komiyamma/xplatl_sentence_relationship_generation/releases/latest) [![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

# 日本語HTMLコーパスの関連ページ推定（TF‑IDF + 埋め込みリランク）

HTMLファイル群からインデックスを構築し、ある1枚のHTML（A.html）に関連する他HTMLをランキングします。基本は文字3‑gramのTF‑IDF + コサイン類似度。任意で Sentence-Transformers による「埋め込みリランク（再ランキング）」を使えます。外部APIには依存せず、ローカルモデル/ファイルのみで動作します。戦国時代テーマの日本語に最適化するため、埋め込みは日本語特化モデルの使用を推奨します。

---

## セットアップ

```bash
pip install beautifulsoup4 lxml scikit-learn scipy joblib
# リランク + 日本語SBERT用トークナイザ（ローカルにモデルを置いてパス指定する想定）
pip install sentence-transformers fugashi unidic-lite
# ANN(HNSW) を使う場合（任意）
pip install hnswlib
```

- BeautifulSoup + lxml: HTML抽出
- scikit-learn: TF‑IDF とコサイン類似度
- scipy: 疎行列の保存
- joblib: ベクトライザの保存
- sentence-transformers: 埋め込み（任意）
- fugashi + unidic-lite: 日本語トークナイザ（`sentence-bert-base-ja-mean-tokens-v2` で必須）
- 作者自身は、`Windows`の`python 3.13`にて動作を確認しています。

---

## 1) インデックス作成: `build_index.py`

HTMLディレクトリを走査し、テキスト抽出→正規化→文字N‑gram TF‑IDF行列を作成し、`index/` 以下に保存します。任意で埋め込み（チャンク単位）の事前計算も可能です。

### 使い方（TF‑IDFのみ）

```bash
python build_index.py \
  --src ./html \
  --out ./index \
  --ngram 3 \
  --min-df 2 \
  --max-df 0.95 \
  --title-weight 3 \
  --heading-weight 2 \
  --drop-selectors "nav,footer,header,aside,.sidebar,.breadcrumbs"
```

### 使い方（埋め込みも事前計算する場合 | 日本語特化モデル）

```bash
python build_index.py \
  --src ./html \
  --out ./index \
  --ngram 3 --min-df 2 --max-df 0.95 \
  --title-weight 3 --heading-weight 2 \
  --drop-selectors "nav,footer,header,aside,.sidebar,.breadcrumbs" \
  --embed-model ./.models/sentence-bert-base-ja-mean-tokens-v2 \
  --embed-max-chars 800 \
  --embed-overlap 200 \
  --embed-batch-size 32

# HNSWインデックスも構築（任意）
  --build-hnsw --hnsw-M 32 --hnsw-efC 200 --hnsw-efS 128
```

> `--embed-model` はモデル名またはローカルパス。外部ダウンロードを避ける場合はローカルパスを指定してください。純日本語コーパス中心なら `sonoisa/sentence-bert-base-ja-mean-tokens-v2` を推奨します（例のようにローカル配置してパス指定）。

### モデルのローカル配置（オフライン運用）

以下のいずれかで日本語モデルを `./.models/sentence-bert-base-ja-mean-tokens-v2` に配置してください。

- Hugging Face CLI を使う方法
  - `pip install -U "huggingface_hub[cli]"`
  - `huggingface-cli download sonoisa/sentence-bert-base-ja-mean-tokens-v2 --local-dir ./.models/sentence-bert-base-ja-mean-tokens-v2`
- Git LFS を使う方法
  - `git lfs install`
  - `git clone https://huggingface.co/sonoisa/sentence-bert-base-ja-mean-tokens-v2 ./.models/sentence-bert-base-ja-mean-tokens-v2`
- 手動ダウンロード
  - モデルページから一式をダウンロードして `./.models/sentence-bert-base-ja-mean-tokens-v2` に展開

中身の目安: `config.json`, `modules.json`, `pytorch_model.bin`（または `model.safetensors`）, `tokenizer.json`/`vocab.txt`, `tokenizer_config.json`, `special_tokens_map.json`, `1_Pooling/config.json` などが含まれていればOKです。

---

## 実行手順（まとめ）

1) 依存をインストール（上記の pip コマンド2行）
2) 日本語モデルをローカル配置（上記の「モデルのローカル配置」）
3) インデックス構築
   - バッチ: `build_index.bat`
   - もしくはコマンド例（上記の「使い方」セクション）
4) 検索（リランク推奨）
   - バッチ: `score_related.bat`（`QUERY` を実在HTMLに合わせて変更可）
   - またはコマンド例（プリコンピュート利用/オンザフライ）

---

## よくあるエラーと対処

- ModuleNotFoundError: The unidic_lite dictionary is not installed
  - 対処: `pip install fugashi unidic-lite`

- `--rerank-model is required if no precomputed embeddings are found`
  - 対処: 事前埋め込みを作る（`build_index.bat` 実行）か、`score_related.py` 実行時に `--rerank-model ./.models/sentence-bert-base-ja-mean-tokens-v2` を指定

- FileNotFoundError: `./html/A.html`
  - 対処: 実在するHTMLファイルを指定（例: `--query "./html/page-bushou-....html"`）。`score_related.bat` の `QUERY` を編集しても可。

- モデルが見つからない（Path ... not found）
  - 対処: モデルのローカルパスが正しいか、`%CD%`（実行カレント）に依存していないか確認。付属バッチは `%~dp0` ベースで動きます。

生成物（`index/`）

- `tfidf_vectorizer.pkl`（ベクトライザ）
- `tfidf_matrix.npz`（疎行列X）
- `id_map.json` / `docs.jsonl`（メタ情報: 相対パス・タイトル等）
- `source_root.txt`（ソースのルートパス）
- 埋め込みを有効にした場合:
 - `embeddings.npy`（全チャンクの埋め込み行列; float32, L2正規化済）
  - `emb_chunks.jsonl`（チャンクのメタ情報）
  - `emb_doc_index.json`（文書ID→チャンク範囲）
  - `emb_model.txt`（使用モデル名/パス）
  - HNSWを有効にした場合:
    - `emb_hnsw.bin`（HNSWlibインデックス）
    - `emb_hnsw_meta.json`（HNSWメタ情報: 次元やM/efなど）

---

## 2) 類似スコアリング: `score_related.py`

TF‑IDFで上位候補を絞り込み、任意で埋め込みリランク（embed）またはTF‑IDFと埋め込みのハイブリッド（hybrid）で再ランキングします。

### 基本（TF‑IDFのみ）

```bash
python score_related.py \
  --index ./index \
  --query ./html/A.html \
  --topk 10 \
  --tau 0.25 \
  --format table
```

### リランク（プリコンピュート利用 | 日本語特化モデル）

```bash
python score_related.py \
  --index ./index \
  --query ./html/A.html \
  --topk 10 --tau 0.25 \
  --rerank-mode hybrid \
  --rerank-topk 200 \
  --alpha 0.5
```

`index/` に埋め込みがない場合は、候補のみオンザフライで埋め込みを計算できます:

```bash
python score_related.py \
  --index ./index \
  --query ./html/A.html \
  --topk 10 --tau 0.25 \
  --rerank-mode embed \
  --rerank-topk 200 \
  --rerank-model ./.models/sentence-bert-base-ja-mean-tokens-v2
```

主な引数（追加分）

- `--rerank-mode`: `none` / `embed` / `hybrid`
- `--rerank-model`: 事前埋め込みが無い場合に使用するモデル名/ローカルパス
- `--rerank-topk`: TF‑IDFの上位Kをリランク対象に
- `--alpha`: ハイブリッドの重み（`alpha*TFIDF + (1-alpha)*Embedding`）
- `--embed-max-chars`, `--embed-overlap`: チャンク分割設定
 - ANN関連（自動認識）:
   - `--ann-mode`: `auto`/`none`/`hnsw`（既定`auto`。`emb_hnsw.bin`があれば自動で使用）
   - `--hnsw-ef`: 検索時efSearchの上書き（省略時は保存済み既定値）
   - `--ann-topk-mult`: ANNで取得するチャンク近傍数の倍率

---

## パラメータのヒント（人名・通称などの揺れ対策）

- `--title-weight`, `--heading-weight` をやや高めに（例: 4/3）にすると固有名を含む見出しの寄与が増えます。
- `--rerank-mode hybrid` を使うと、表記揺れ（別名・通称）に対して文脈で補える場合が増えます。埋め込みモデルは日本語特化の `sentence-bert-base-ja-mean-tokens-v2` を推奨。
- `--tau` は0.2～0.35付近から調整。母集団が多いほどやや高めが安定。
- 旧字体や歴史的仮名が混在する場合は、HTML側での表記統一や追加の正規化を検討してください。

---

## 技術詳細

計算・設計の詳細は `TECHNICAL_DETAILS.md` を参照してください。

---

## ライセンス

MIT License
