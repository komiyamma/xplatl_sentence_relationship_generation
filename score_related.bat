python score_related.py ^
  --index "./index" ^
  --query "./html/page-bushou-êDìcêMí∑.html" ^
  --topk 10 --tau 0.05 ^
  --rerank-model ./.models/sentence-bert-base-ja-mean-tokens-v2 ^
  --rerank-mode hybrid ^
  --rerank-topk 200 ^
  --alpha 0.5
