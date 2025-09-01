@echo off
setlocal
set "ROOT=%~dp0"
set "INDEX=%ROOT%index"

REM NOTE: QUERYに既存のHTMLファイルのパスを設定する。
REM パスにASCII以外の文字が含まれている場合は、Shift-JISを保持するエディタで編集してください。
set "QUERY=%ROOT%html\ABC.html"

set "MODEL=%ROOT%.models\sentence-bert-base-ja-mean-tokens-v2"

REM もしあれば、事前に計算された埋め込みを優先する。そうでない場合はローカルモデルを必要とする。
set "RERANK_MODE=hybrid"
set "RERANK_ARGS=--rerank-mode %RERANK_MODE% --rerank-topk 200 --alpha 0.5"

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
  --topk 30 ^
  --tau 0.05 ^
  %RERANK_ARGS%

endlocal
