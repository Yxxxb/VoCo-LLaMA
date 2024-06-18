#!/bin/bash

CUDA_VISIBLE_DEVICES=3 python -m llava.eval.model_vqa_loader \
    --model-path voco_llama_ckpt \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder /group/40033/public_datasets/coco/val2014 \
    --answers-file ./playground/data/eval/pope/answers/voco_llama.jsonl \
    --voco_num 2 \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/voco_llama.jsonl

