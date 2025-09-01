# 日本語HTML文書の自動リンク（インデクサ & スコアラー）

「HTMLファイル群をデータ化（インデックス化）」→「特定のA.htmlに関連しそうな他ファイルへスコア付きリンク」を出す、2つのスクリプトです。日本語前提で、**文字3-gram TF‑IDF + コサイン類似度**を使い、分かち書き不要でまず“速く正確に”動きます。

---

## セットアップ

```bash
pip install beautifulsoup4 lxml scikit-learn scipy joblib
```

* **BeautifulSoup + lxml**: HTML抽出
* **scikit-learn**: TF‑IDF と 類似度
* **scipy**: 疎行列保存
* **joblib**: モデル保存

---

## 1) インデクサ: `build_index.py`

HTMLディレクトリを走査し、テキスト抽出→正規化→文字3-gram TF‑IDF を作成し、`index/` 以下に保存します。

---

## 2) スコアラー: `score_related.py`

`build_index.py` で作ったインデックスを読み込み、**A.html**（コーパス内でも外部ファイルでも可）に**関連しそうなファイルをスコア付きで列挙**します。

```python
# score_related.py
```

---

## 使い方

### 1) インデックス作成

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

> `./html` 配下の `**/*.html` を全走査して `./index/` に以下が生成されます：
>
> * `tfidf_vectorizer.pkl`（ベクトライザ）
> * `tfidf_matrix.npz`（疎行列）
> * `id_map.json` / `docs.jsonl`（メタデータ）
> * `source_root.txt`（元ディレクトリ）

### 2) A.html に関連するファイルを出す

```bash
python score_related.py \
  --index ./index \
  --query ./html/A.html \
  --topk 10 \
  --tau 0.25 \
  --format table
```

* `--tau` はしきい値（0〜1）。高いほど厳しくなります。
* `--format json` で機械可読なJSON出力も可能。

---

## 精度調整のヒント

* **ngram=3** がまずは無難。短文や固有名詞が多いなら 2〜4 を試す。
* **min\_df** を 2〜3 に上げるとノイズが減る（小規模コーパスでは1でもOK）。
* **max\_df** を 0.85〜0.95 に下げると“どこにでも出る断片”を抑えられます。
* タイトル/見出しの**重み**でリンクの質が上がることが多いです。
* ナビ/フッター/パンくず等の **drop\_selectors** はサイト構造に合わせて調整。

---

## 次のステップ（必要になったら）

* 上位候補だけ \*\*埋め込み（Sentence-Transformers等）\*\*で再ランク → 言い換えに強化。
* **相互リンクにボーナス**を付与（A→B かつ B→A を優先）。
* 章・見出し単位でのリンク生成（H2/H3ごとにTF‑IDF→スコアリング）。
* 大規模なら **FAISS/hnswlib** で近傍検索をANNにしてミリ秒応答化。

必要なら、サイト構造に合わせて `--drop-selectors` や重み、ngram など調整してください。
他にも「相互リンク優先」や「埋め込みで再ランク」などの拡張も入れられます。

---

## ライセンス

MIT License
