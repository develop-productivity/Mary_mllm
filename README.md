# Introduction

Marry_mllm 旨在提供一个小型多模态大语言模型，并提供一个简单可用的预训练、SFT以及DPO训练代码。
本项目模型结构，训练细节、原理等均来自以下两个项目的启发：
- [omni-vision](https://nexa.ai/blogs/omni-vision)
- [完全从零开始实现DPO算法，不依赖trl库，已经实现预训练、SFT、DPO全流程，公式对照代码，让你完全搞懂](https://www.bilibili.com/video/BV1keqaY8EY4/?spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=9de8f05b70f1b7f369db9e085e0a9eb5)

## 模型结构

<img src="asserts/framework.png" width="80%">

# QucikStart
## Environment

Personal Software and hardware environment configuration
```
GPU: RTX3090x4
CPU: Intel(R) Xeon(R) Silver 4214R CPU @ 2.40GHz
python: 3.10
pytorch: torch2.1.2+cu121
```

You should clone this project and create a python env.
```
git clone https://github.com/develop-productivity/Marry_mllm.git
cd Marry_mllm
conda create -n env_name python=3.10
pip install -r requirments.txt

```

## Test

## Train

### datasets preparation
Datasets:
* pre-train datasets: [LLaVA-CC3M-Pretrain-595K](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K)

* sft datasets: [LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K). We use LLaVA-Instruct-150K and LLaVA-Instruct-80K

* pretrain and sft images: [minimind_v](https://huggingface.co/datasets/jingyaogong/minimind-v_dataset/tree/main)

* dpo datasets:


### the pre-trained model checkpoint
* vision model: [siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384)
* language model: [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)

You should manguage you the file structure as follows
```

```

### run the scripts


# TODO

- [ ] Support LoRA finetine
- [ ] Suport kbit training
- [ ]  Support deepspeed config
- [ ] 
