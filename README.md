# AdaOWQ: Adaptive Outlier Allocation Protection for Three-Tier Weight Quantization in Large Language Models

## This is a submitted paper to the PeerJ Computer Science journal for review. When reviewers request the full code, it will be updated.

## For quantization
1. CUDA_VISIBLE_DEVICES=4 python main.py  /home/xujie-intern/.cache/huggingface/hub/llama-2-13b-hf/snapshots/v1/  wikitext2 --wbits 3 --target_bit 3.01  --expansion_factor 1.5  --tier1_ratio 0.3  --tier2_bits 8  --act-order  --true-sequential --seed 10000  --fake  --save  /data-model/infer-r1/xujie/my_code/owq/log/llama2-13-3.01-v10.pth 2>&1 | tee /data-model/infer-r1/xujie/my_code/owq/log/213b-3.01-v10.log


## For evaluation on zero-shot
2. CUDA_VISIBLE_DEVICES=5  python zeroshot.py   --model hf-causal-owq   --model_args pretrained=/home/xujie-intern/.cache/huggingface/hub/llama2-7b/snapshots/,load=/data-model/infer-r1/xujie/my_code/owq/log/llama2-7b-v3_fake.pth   --batch_size 4   --tasks  winogrande   --no_cache   --num_fewshot 0


## Custom medical datasets
1. quant_sample_per_task_20.with_task.jsonl


[1] Reference: https://github.com/xvyaward/owq/blob/main/README.md

## Citation
If you use this code or method in your research, please cite our manuscript:
[AdaOWQ], Under Review at PeerJ, 2026.
