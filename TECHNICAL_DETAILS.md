# 技術詳細ドキュメント

このドキュメントは、`build_index.py` と `score_related.py` の設計・実装・パラメータの詳細、ならびに戦国時代テーマの日本語コーパスに対する配慮についてまとめます。既存のTF‑IDFベースの設計を踏襲しつつ、埋め込みによる再ランキング（リランク）を追記しました。

---

## 1. 全体像

目的は、HTMLコーパス内で「ある1枚のHTML（A.html）に関連する他のHTML」を見つけ、ランキングすることです。

1) 抽出: HTMLからタイトル/見出し/本文を抽出（不要要素はCSSセレクタで除去）
2) 正規化: NFKC・URL除去・空白正規化
3) TF‑IDF: 文字N‑gram（既定=3）でベクトル化し疎行列として保存
4) 類似度: クエリ（A.html）を同手順でベクトル化し、全体とのコサイン類似度
5) リランク（任意）: 上位K候補だけを日本語特化の埋め込みモデルで再スコア or TF‑IDFと融合

---

## 1.1. 文字 N‑gram

文字ベースの N‑gram（例: n=3）を用いてテキストを疎に表現します。単語分割に依存しないため、固有名や記号が多いテキスト、語彙の揺れに比較的ロバストです。n は 2〜4 程度でタスクに応じて調整可能です。

---

## 1.2. TF‑IDF

- TF（Term Frequency）: 文書内での N‑gram の相対頻度
- IDF（Inverse Document Frequency）: コーパス内での希少性
- TF×IDF により、特定文書の特徴的な N‑gram が強調されます。

実装: `scikit-learn` の `TfidfVectorizer(analyzer="char", ngram_range=(n, n), min_df, max_df)` を使用。保存は `tfidf_vectorizer.pkl`（ベクトライザ）と `tfidf_matrix.npz`（疎行列）。

---

## 1.3. コサイン類似度

クエリベクトル q と文書行列 X のコサイン類似度を `sklearn.metrics.pairwise.cosine_similarity` で計算します。スコア降順でソートし、しきい値 `tau` や上限 `topk` で出力を整形します。

---

## 2. テキスト抽出と正規化

抽出（BeautifulSoup）:
- 除外タグ: `script`, `style`, `noscript`
- 除外セレクタ（任意指定）: `nav, footer, header, aside, .sidebar, .breadcrumbs, ...`
- タイトル: `<title>`
- 見出し: `h1, h2, h3`
- 本文: `<body>`（無い場合はドキュメント全体）

正規化:
- NFKC（全角/半角を含む互換正規化）
- URL除去（`https?://\S+`）
- 連続空白の圧縮（1スペース）

タイトル/見出しは `--title-weight`, `--heading-weight` で加重し本文と連結します。

---

## 3. 出力（インデックス側）

- `tfidf_vectorizer.pkl`: TF‑IDF ベクトライザ
- `tfidf_matrix.npz`: TF‑IDF 疎行列
- `docs.jsonl`: 文書メタ（`path`, `title`, `length`, `hash`）
- `id_map.json`: 行番号→文書メタのマップ
- `source_root.txt`: 元HTMLルート

---

## 4. 埋め込みリランク（任意）

TF‑IDF はキーワード一致に強い一方、別表記や言い換えには弱い面があります。そこで、上位候補K件だけを埋め込みモデルで再スコアし、順位を入れ替える再ランキングを実装しています。外部APIは使わず、ローカルのモデル/ファイルのみで動作します。

### 4.1 事前計算（インデックス時）

`build_index.py` に以下を指定すると、埋め込みをチャンク単位で事前計算して保存します。
- `--embed-model ./.models/sentence-bert-base-ja-mean-tokens-v2`
- `--embed-max-chars 800`, `--embed-overlap 200`, `--embed-batch-size 32`

保存物:
- `embeddings.npy`: すべてのチャンクの埋め込み（float32, L2 正規化済み）
- `emb_chunks.jsonl`: チャンクのメタ情報（文書ID・範囲など）
- `emb_doc_index.json`: 文書ID → チャンク範囲 [start, end)
- `emb_model.txt`: 使用したモデル識別子（整合性確認用）

### 4.2 検索時のリランク

`score_related.py` のオプションで制御します。
- `--rerank-mode`: `none` / `embed` / `hybrid`
- `--rerank-topk`: TF‑IDF 上位K件をリランク対象に
- `--alpha`: ハイブリッド時の重み（`alpha*TFIDF + (1-alpha)*Embedding`）
- `--rerank-model`: 事前計算が無い場合のローカルモデルパス
- `--embed-max-chars`, `--embed-overlap`: クエリ/候補のチャンク分割設定

スコアリングは、クエリと文書の各チャンク埋め込みの内積（=コサイン）を取り、その最大値を文書スコアとします。`hybrid` では TF‑IDF と埋め込みスコアを min‑max 正規化後に加重平均で融合します。

---

## 5. 推奨モデル（戦国時代の日本語コーパス前提）

- 日本語特化: `sonoisa/sentence-bert-base-ja-mean-tokens-v2` を推奨。固有名・地名・通称の揺れに対して安定。
- 運用: モデルはローカルに配置し、例のようにパス指定（`./.models/sentence-bert-base-ja-mean-tokens-v2`）。

E5 系（`intfloat/multilingual-e5-*`）は検索用途に強力ですが、英語寄りの前提（query/passsage 接頭辞）もあり、純和文・歴史語彙中心ではまず SBERT 系からの適用が無難です。

---

## 6. パラメータ例（初期値の目安）

- TF‑IDF: `ngram=3, min_df=2, max_df=0.95`
- 重み: `title_weight=3, heading_weight=2`
- 埋め込み: `embed_max_chars=800, embed_overlap=200`
- リランク: `rerank_topk=200, alpha=0.5, tau=0.25`

---

## 7. 日本語（旧字体/歴史的仮名）への配慮

- 追加正規化: 旧→新字体変換、長音・撥音・拗音のゆらぎ吸収などを前段で適用すると頑健性が向上します。
- 見出し強調: 人名や城名が含まれる見出し配下の要約段落に追加ウェイトを加えると、TF‑IDFの候補抽出が安定します。

---

## 8. 計算量・速度設計

- インデックス生成は時間がかかっても良い前提に合わせ、埋め込みを事前計算して保存します。
- 検索時は、TF‑IDF 類似度で絞り込んだ上位Kに対してのみ埋め込みスコアを評価（プリコンピュートありなら高速）。
- 必要に応じて ANN（HNSW/FAISS）の導入も可能です（現実装はコサインでの直接スコアリング）。

---

## 9. 代表的な実行例

事前埋め込み付きインデックス作成:

```bash
python build_index.py \
  --src ./html \
  --out ./index \
  --ngram 3 --min-df 2 --max-df 0.95 \
  --title-weight 3 --heading-weight 2 \
  --drop-selectors "nav,footer,header,aside,.sidebar,.breadcrumbs" \
  --embed-model ./.models/sentence-bert-base-ja-mean-tokens-v2 \
  --embed-max-chars 800 --embed-overlap 200
```

リランク検索（ハイブリッド）:

```bash
python score_related.py \
  --index ./index \
  --query ./html/A.html \
  --topk 10 --tau 0.25 \
  --rerank-mode hybrid --rerank-topk 200 --alpha 0.5
```

プリコンピュートが無い場合（オンザフライ埋め込み）:

```bash
python score_related.py \
  --index ./index \
  --query ./html/A.html \
  --topk 10 --tau 0.25 \
  --rerank-mode embed --rerank-topk 200 \
  --rerank-model ./.models/sentence-bert-base-ja-mean-tokens-v2
```
