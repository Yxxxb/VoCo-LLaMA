#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python -m llava.eval.model_vqa_science \
    --model-path voco_llama_ckpt \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/voco_llama.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --voco_num 2 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/voco_llama.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/voco_llama_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/voco_llama_result.json
