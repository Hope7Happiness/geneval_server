# python geneval/generation/random_generate.py geneval/prompts/evaluation_metadata.jsonl --outdir ./random_out --batch_size 4
# python geneval/generation/diffusers_generate.py geneval/prompts/evaluation_metadata.jsonl --model "stable-diffusion-v1-5/stable-diffusion-v1-5" --outdir ./sd1.5_out --skip_grid --batch_size 4


# python geneval/evaluation/evaluate_images.py \
#     ./random_out \
#     --outfile "./random_out/results.jsonl" \
#     --model-path ./pretrained

# python geneval/evaluation/summary_scores.py "./random_out/results.jsonl"


python geneval/evaluation/evaluate_images.py \
    ./sd1.5_out \
    --outfile "./sd1.5_out/results.jsonl" \
    --model-path ./pretrained
python geneval/evaluation/summary_scores.py "./sd1.5_out/results.jsonl"