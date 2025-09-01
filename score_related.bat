@echo off
setlocal
set "ROOT=%~dp0"
set "INDEX=%ROOT%index"

REM NOTE: QUERY�Ɋ�����HTML�t�@�C���̃p�X��ݒ肷��B
REM �p�X��ASCII�ȊO�̕������܂܂�Ă���ꍇ�́AShift-JIS��ێ�����G�f�B�^�ŕҏW���Ă��������B
set "QUERY=%ROOT%html\ABC.html"

set "MODEL=%ROOT%.models\sentence-bert-base-ja-mean-tokens-v2"

REM ��������΁A���O�Ɍv�Z���ꂽ���ߍ��݂�D�悷��B�����łȂ��ꍇ�̓��[�J�����f����K�v�Ƃ���B
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
