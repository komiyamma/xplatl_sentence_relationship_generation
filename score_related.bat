@echo off
setlocal
set "ROOT=%~dp0"
set "INDEX=%ROOT%index"

REM NOTE: Set QUERY to an existing HTML file path.
REM If your path contains non-ASCII characters, edit this in an editor that preserves Shift-JIS.
set "QUERY=%ROOT%html\page-bushou-êDìcêMí∑.html"

set "MODEL=%ROOT%.models\sentence-bert-base-ja-mean-tokens-v2"

REM Prefer precomputed embeddings if present; otherwise require local model
set "RERANK_MODE=hybrid"
set "RERANK_ARGS=--rerank-mode %RERANK_MODE% --rerank-topk 200 --alpha 0.5"

REM Prefer HNSW ANN if index exists; otherwise fall back to auto/TF-IDF-only
set "ANN_ARGS=--ann-mode auto"
if exist "%INDEX%\emb_hnsw.bin" (
  echo Using HNSW index under "%INDEX%".
  set "ANN_ARGS=--ann-mode hnsw --hnsw-ef 256 --ann-topk-mult 5"
) else (
  echo No HNSW index found; continuing without ANN acceleration.
)

if exist "%INDEX%\embeddings.npy" (
  echo Using precomputed embeddings under "%INDEX%".
) else (
  if not exist "%MODEL%" (
    echo [ERROR] No precomputed embeddings found under "%INDEX%" and model path not found: "%MODEL%".
    echo         Either run build_index.bat first or place the model locally and adjust MODEL path.
    exit /b 1
  )
  set "RERANK_ARGS=%RERANK_ARGS% --rerank-model %MODEL%"
)

python "%ROOT%score_related.py" ^
  --index "%INDEX%" ^
  --query "%QUERY%" ^
  --topk 10 ^
  --tau 0.05 ^
  %RERANK_ARGS% ^
  %ANN_ARGS%

endlocal
