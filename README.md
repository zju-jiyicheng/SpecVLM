<div align="center">
<h1>
    <img src="assets/logo.png" height="100px" align="top"/>  
    <br>  
    <em>SpecVLM</em>  
    <br>  
</h1>

<p align="center">
<a href="https://www.arxiv.org/abs/2508.16201">
  <img src="https://img.shields.io/badge/arXiv-2508.16201-b31b1b.svg"></a> 
<a href="https://openreview.net/forum?id=mWElG6fKEN">
  <img src="https://img.shields.io/badge/EMNLP-2025-red.svg"></a> 
<a href="https://opensource.org/licenses/Apache-2.0">
  <img src="https://img.shields.io/badge/License-Apache_2.0-green.svg"></a> 
<a href="https://github.com/zju-jiyicheng/SpecVLM/pulls">
    <img src="https://img.shields.io/badge/Contributions-welcome-blue.svg?style=flat"></a>
</p>

<h2 style="color: #6a5acd; font-weight: bold;">
🚀 2.68× Decoding Speedup with 90% Token Reduction ⬇️
</h2>

</div>

Yicheng Ji*, Jun Zhang*, Heming Xia, Jinpeng Chen, Lidan Shou, Gang Chen, Huan Li (* equal contribution)

## 📣 Overview
<p align="center">
  <img src="assets/teaser.png" alt="Framework" width="100%"/>
</p>

## 🎉 News
- 2025.08: Our paper "[SpecVLM: Enhancing Speculative Decoding of Video LLMs via Verifier-Guided Token Pruning]" has been accepted to EMNLP 2025 Main. [->Link](https://arxiv.org/abs/2508.16201)
- 2026.02: Our paper "[ParallelVLM: Lossless Video-LLM Acceleration with Visual Alignment Aware Parallel Speculative Decoding]" has been accepted to CVPR 2026. [->Link](https://arxiv.org/abs/2603.19610)
- 2026.04: Our paper "[See the Forest for the Trees: Loosely Speculative Decoding via Visual-Semantic Guidance for Efficient Inference of Video LLMs]" has been accepted to ACL 2026 Main. [->Link](https://arxiv.org/abs/2604.05650)

## 📌  Method
<p align="center">
  <img src="assets/overview.png" alt="SpecVLM Framework" width="80%"/>
</p>


## 📖 Abstract
Video large language models (Vid-LLMs) have shown strong capabilities in understanding video content. However, their reliance on dense video token representations introduces substantial memory and computational overhead in both prefilling and decoding. To mitigate the information loss of recent video token reduction methods and accelerate the decoding stage of Vid-LLMs losslessly, we introduce SpecVLM, a training-free speculative decoding (SD) framework tailored for Vid-LLMs that incorporates staged video token pruning. Building on our novel finding that the draft model’s speculation exhibits low sensitivity to video token pruning, SpecVLM prunes up to 90% of video tokens to enable efficient speculation without sacrificing accuracy. To achieve this, we perform a two-stage pruning process: Stage I selects highly informative tokens guided by attention signals from the verifier (target model), while Stage II prunes the remaining redundant ones in a spatially uniform manner. Extensive experiments on four video understanding benchmarks demonstrate the effectiveness and robustness of SpecVLM, which achieves up to 2.68× decoding speedup for LLaVA-OneVision-72B and 2.11× speedup for Qwen2.5-VL-32B.  

---

## ⚙️ Environment Setup

Install the required dependencies:
```bash
conda create -n specvlm python==3.10 -y
conda activate specvlm
pip install torch torchvision
pip install -r requirements.txt
```


## 🛠 Download Models & Datasets
- For LLaVA-OneVision models: [->Link](https://huggingface.co/llava-hf)
- For Qwen2.5-VL models: [->Link](https://huggingface.co/Qwen)
- For VideoDetailCaption dataset: [->Link](https://huggingface.co/datasets/lmms-lab/VideoDetailCaption)


## 🚀 Quick Evaluation
Run the demo script to quickly evaluate SpecVLM:
```bash
# Qwen2.5-VL
sh run_qwenvl.sh
# LLaVA-OneVision
sh run_llava.sh
```
Please also moderate the model path, data path, pruning ratio, and frame number in the .sh file.

After runing the script, the output result and efficiency metric will be stored in the save_path.

## Note
- Our method primarily targets resource-constrained long-video scenarios, where GPU memory bandwidth constitutes the main bottleneck during inference. Users are advised to set the input length according to GPU capacity. Theoretically, as frame number grows, SpecVLM achieves higher acceleration ratios.
- In principle, our approach is lossless, with only minimal impact introduced parallel decoding settings. Given the insensitivity of draft models to token pruning, we also recommend uniform pruning as a compatibility-friendly alternative.

## Citation
If you find our work useful or relevant to your research, please kindly cite our papers:
```bibtex
@inproceedings{ji2025specvlm,
  title={Specvlm: Enhancing speculative decoding of video llms via verifier-guided token pruning},
  author={Ji, Yicheng and Zhang, Jun and Xia, Heming and Chen, Jinpeng and Shou, Lidan and Chen, Gang and Li, Huan},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  pages={7216--7230},
  year={2025}
}

@misc{ji2026foresttreeslooselyspeculative,
  title={See the Forest for the Trees: Loosely Speculative Decoding via Visual-Semantic Guidance for Efficient Inference of Video LLMs},
  author={Yicheng Ji and Jun Zhang and Jinpeng Chen and Cong Wang and Lidan Shou and Gang Chen and Huan Li},
  year={2026},
  eprint={2604.05650},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2604.05650}
}

@article{kong2026parallelvlm,
  title={ParallelVLM: Lossless Video-LLM Acceleration with Visual Alignment Aware Parallel Speculative Decoding},
  author={Kong, Quan and Shen, Yuhao and Ji, Yicheng and Li, Huan and Wang, Cong},
  journal={arXiv preprint arXiv:2603.19610},
  year={2026},
  url={https://arxiv.org/abs/2603.19610}
}
```
