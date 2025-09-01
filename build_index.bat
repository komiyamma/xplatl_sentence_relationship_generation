@echo off
setlocal
set "ROOT=%~dp0"

python "%ROOT%build_index.py" ^
  --src "%ROOT%html" ^
  --out "%ROOT%index" ^
  --ngram 3 ^
  --min-df 2 ^
  --max-df 0.95 ^
  --title-weight 10 ^
  --heading-weight 4 ^
  --drop-selectors "nav,footer,header,aside,.sidebar,.breadcrumbs" ^
  --embed-model "%ROOT%.models\sentence-bert-base-ja-mean-tokens-v2" ^
  --embed-max-chars 800 ^
  --embed-overlap 200 ^
  --embed-batch-size 32 ^
  --build-hnsw ^
  --hnsw-M 32 ^
  --hnsw-efC 200 ^
  --hnsw-efS 128 ^
  --hnsw-threads 0

endlocal
