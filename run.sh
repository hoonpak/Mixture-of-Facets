python /script/get_user_embs.py \
    -c /configs/user_embs/user_embs.yaml

python /script/get_clustered_training_data.py \
    -c /configs/user_embs/clustering.yaml

python /script/run_openai_inference.py \
    -c /configs/reranker/reranker.yaml \
    -k /secure/openai.txt

python /script/run_training.py \
    -c /configs/reranker/reranker.yaml \
    -v 0 --debug train 

python /script/run_training.py \
    -c /configs/reranker/reranker.yaml \
    -v 0 --debug train_eval

python /script/run_eval_predict_bbllm.py \
    -c /configs/reranker/reranker.yaml \
    -k /secure/openai.txt \
    -t train_eval -v 0

python /script/run_training.py \
    -c /configs/reranker/reranker.yaml \
    -v 0 --debug fit 

python /script/run_training.py \
    -c /configs/reranker/reranker.yaml \
    -v 0 --debug fit_eval

python /script/run_eval_predict_bbllm.py \
    -c /configs/reranker/reranker.yaml \
    -k /secure/openai.txt \
    -t fit_eval -v 0

python /script/get_adapter_training_data.py \
    -rc /configs/reranker/reranker.yaml \
    -ac /configs/adapter/adapter.yaml \
    --version 0

python /script/run_openai_inference.py \
    -c /configs/adapter/adapter.yaml \
    -k /secure/openai.txt

python /script/run_training.py \
    -c /configs/adapter/adapter.yaml \
    -v 0 --debug train 

python /script/run_training.py \
    -c /configs/adapter/adapter.yaml \
    -v 0 --debug train_eval 

python /script/run_training.py \
    -c /configs/adapter/adapter.yaml \
    -v 0 --debug fit 

python /script/run_training.py \
    -c /configs/adapter/adapter.yaml \
    -v 0 --debug fit_eval 

python /script/run_eval_predict_bbllm.py \
    -c /configs/adapter/adapter.yaml \
    -k /secure/openai.txt \
    -t train_eval -v 0

python /script/run_eval_predict_bbllm.py \
    -c /configs/adapter/adapter.yaml \
    -k /secure/openai.txt \
    -t fit_eval -v 0
