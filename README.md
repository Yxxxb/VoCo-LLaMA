# VoCo-LLaMA: Towards Vision Compression with Large Language Models

[Xubing Ye](https://github.com/Yxxxb), [Yukang Gan](https://scholar.google.com/citations?user=8rltp9AAAAAJ&hl=zh-CN), [Xiaoke Huang](https://xk-huang.github.io/), [Yixiao Ge](https://geyixiao.com/), [Ying Shan](https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en), [Yansong Tang](https://andytang15.github.io)

<p align="left">
  <a href='https://arxiv.org/abs/2406.12275'>
  <img src='https://img.shields.io/badge/Arxiv-2404.19759-A42C25?style=flat&logo=arXiv&logoColor=A42C25'></a> 
  <a href='https://arxiv.org/pdf/2406.12275'>
  <img src='https://img.shields.io/badge/Paper-PDF-purple?style=flat&logo=arXiv&logoColor=yellow'></a> 
  <a href='https://yxxxb.github.io/VoCo-LLaMA-page/'>
  <img src='https://img.shields.io/badge/Project-Page-%23df5b46?style=flat&logo=Google%20chrome&logoColor=%23df5b46'></a> 
</p>

## TL;DR

We propose VoCo-LLaMA, the first approach to compress vision tokens using LLMs. By fully utilizing the LLMs' understanding paradigm of vision tokens, our method can compress hundreds of vision tokens into a single VoCo token, while minimizing visual information loss.

VoCo-LLaMA demonstrates the ability to understand video through continuous training using time-series compressed token sequences of video frames.

VoCo-LLaMA presents a promising way to unlock the full potential of VLMs' contextual window.

![image](https://github.com/Yxxxb/VoCo-LLaMA/blob/main/IMG/structure.jpg)

## News

- [x] **[2024/06/17]** Upload paper and release vision compression code.

## Preparation

### Install

1. Clone this repository and navigate to VoCo-LLaMA folder

```bash
git clone https://github.com/Yxxxb/VoCo-LLaMA.git
cd VoCo-LLaMA
```

2. Install Package

```Shell
conda create -n voco_llama python=3.10 -y
conda activate voco_llama
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install additional packages for training cases

```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
cp /group/40034/xubingye/VoCo-LLaMA/llava/model/language_model/cache_py/modeling_attn_mask_utils.py /data/miniconda3/envs/llava/lib/python3.10/site-packages/transformers/modeling_attn_mask_utils.py
```

### Data and Pre-trained weights

VoCo-LLaMA training requires only visual instruction fine-tuning. Please download the aligned LLaVA checkpoints ([base LLM and projection layers](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md)). Please download the annotation of the LLaVA instruction tuning data [llava_v1_5_mix665k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json), and download the images from constituting datasets:

- COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip)
- GQA: [images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)
- OCR-VQA: [download script](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing), we save all files as `.jpg`
- TextVQA: [train_val_images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)
- VisualGenome: [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)

After downloading all of them, organize the data as follows in `./playground/data`,

```
├── coco
│   └── train2017
├── gqa
│   └── images
├── ocr_vqa
│   └── images
├── textvqa
│   └── train_images
└── vg
    ├── VG_100K
    └── VG_100K_2
```

## Train

VoCo-LLaMA is trained on 8 A100 GPUs with 40GB memory. To train on fewer GPUs, you can reduce the `per_device_train_batch_size` and increase the `gradient_accumulation_steps` accordingly. Always keep the global batch size the same: `per_device_train_batch_size` x `gradient_accumulation_steps` x `num_gpus`. 

Train VoCo-LLaMA with vision instruction tuning by running following command:

```
bash scripts/finetune.sh
```

## Evaluation

There are evaluations about visual understanding we follow the relevant settings in LLaVA. Please refer to the LLaVA official [repository](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md) for details of data setup and testing.

## Acknowledgement

- [LLaVA](https://github.com/haotian-liu/LLaVA): the codebase we built upon.
- [Vicuna](https://github.com/lm-sys/FastChat): our base model Vicuna-7B that has the amazing language capabilities!



