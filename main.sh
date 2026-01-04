set -ex

SUBMISSION_DIR=$1

mkdir -p ./server_logs
rm -rf ./server_logs/results.jsonl ./server_logs/summary.json
/data/scratch-oc40/zhh24/anaconda3/envs/geneval/bin/python geneval/evaluation/evaluate_images.py \
    $SUBMISSION_DIR \
    --outfile "./server_logs/results.jsonl" \
    --model-path ./pretrained
/data/scratch-oc40/zhh24/anaconda3/envs/geneval/bin/python geneval/evaluation/summary_scores.py ./server_logs/results.jsonl ./server_logs/summary.json