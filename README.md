# Beyond KV Caching: Shared Attention for Efficient LLMs
[[paper](TBD)]

## Abstract
The efficiency of large language models (LLMs) remains a critical challenge, particularly in contexts where computational resources are limited. Traditional attention mechanisms in these models, while powerful, require significant computational and memory resources due to the necessity of recalculating and storing attention weights across different layers. This paper introduces a novel Shared Attention (SA) mechanism, designed to enhance the efficiency of LLMs by directly sharing computed attention weights across multiple layers. Unlike previous methods that focus on sharing intermediate Key-Value (KV) caches, our approach utilizes the isotropic tendencies of attention distributions observed in advanced LLMs post-pretraining to reduce both the computational flops and the size of the KV cache required during inference. We empirically demonstrate that implementing SA across various LLMs results in minimal accuracy loss on standard benchmarks. Our findings suggest that SA not only conserves computational resources but also maintains robust model performance, thereby facilitating the deployment of more efficient LLMs in resource-constrained environments.

## Usage

### Environment Setup

```bash
conda create -n shareAtt python=3.8
conda activate shareAtt

pip install torch torchvision torchaudio
pip install transformers==4.33.0 accelerate datasets scipy sentencepiece
```
### Prepare Weights

Download the Llama-2-7B-hf weights (.safetensor files) into the `models/Llama2-7b-hf` folder.

### Direct Application of Shared Attention

To apply Shared Attention, modify `modeling_llama.py` in `models/Llama2-7b-hf` at line 262.
For instance, for SA from layers 27 to 30 (excluding layer_idx from the list):
```python
self.share_attn = self.layer_idx not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 31]
```
### Reproduction of Evaluations

Install `lm-evaluation-harness` from [EleutherAI's repository](https://github.com/EleutherAI/lm-evaluation-harness).

Replace the `modeling_llama.py` file in the transformers library with the modified file in `models/Llama2-7b-hf`.

Run the evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 lm_eval --model hf --model_args pretrained=./models/Llama2-7b-hf/ --tasks mmlu,glue,gsm8k,hellaswag --batch_size auto --output_path ./eval_out/llama2-7b-23_26  --use_cache ./eval_cache/llama2-7b-23_26
```

### Fine-tuning

Set up Accelerate with DeepSpeed:

```bash
accelerate config
```

Download Llama-3-8b and modify corresponding files.

Download Alpaca instruct dataset `alpaca_data_cleaned.json` from [gururise's repository](https://github.com/gururise/AlpacaDataCleaned).

Train the model:
```bash
ACCELERATE_USE_DEEPSPEED=true CUDA_VISIBLE_DEVICES="0,1" accelerate launch alpaca_finetuning.py
```

## Citation

If you find our works useful or relevant to your project and research, please kindly cite our paper:

```bibtex
@article{liao2024shareAtt,
        title={Beyond KV Caching: Shared Attention for Efficient LLMs},
        author={Bingli Liao and Danilo Vasconcellos Vargas},
        journal={arXiv},
        year={2024}
        }
```
