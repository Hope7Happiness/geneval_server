set -ex

SUBMISSION_DIR=$1

# CONDA_PY_PATH=/data/scratch-oc40/zhh24/anaconda3/envs/geneval/bin/python
CONDA_PY_PATH=/srv/mingyang/miniconda3/envs/geneval/bin/python

mkdir -p ./server_logs
rm -rf ./server_logs/results.jsonl ./server_logs/summary.json
$CONDA_PY_PATH geneval/evaluation/evaluate_images.py \
    $SUBMISSION_DIR \
    --outfile "./server_logs/results.jsonl" \
    --model-path ./pretrained
$CONDA_PY_PATH geneval/evaluation/summary_scores.py ./server_logs/results.jsonl ./server_logs/summary.json