#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python -m llava.eval.model_vqa_loader \
    --model-path voco_llama_ckpt \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/voco_llama.jsonl \
    --voco_num 2 \
    --temperature 0 \
    --conv-mode vicuna_v1

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment voco_llama

cd eval_tool

python calculation.py --results_dir answers/voco_llama

