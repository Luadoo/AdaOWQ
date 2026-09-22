# 1. Title: Adaptive Outlier Allocation Protection for Three-Tier Weight Quantization in Large Language Models

## This is a submitted paper to the PeerJ Computer Science journal for peer review.

# 2. Description:


# 3. Dataset Information:
## Custom medical datasets
1. quant_sample_per_task_20.with_task.jsonl

# 4. Code Information:

## For quantization (Usage Instructions)
1. CUDA_VISIBLE_DEVICES=4 python main.py  /home/xujie-intern/.cache/huggingface/hub/llama-2-13b-hf/snapshots/v1/  wikitext2 --wbits 3 --target_bit 3.01  --expansion_factor 1.5  --tier1_ratio 0.3  --tier2_bits 8  --act-order  --true-sequential --seed 10000  --fake  --save  /data-model/infer-r1/xujie/my_code/owq/log/llama2-13-3.01-v10.pth 2>&1 | tee /data-model/infer-r1/xujie/my_code/owq/log/213b-3.01-v10.log


## For evaluation on zero-shot (Usage Instructions)
2. CUDA_VISIBLE_DEVICES=5  python zeroshot.py   --model hf-causal-owq   --model_args pretrained=/home/xujie-intern/.cache/huggingface/hub/llama2-7b/snapshots/,load=/data-model/infer-r1/xujie/my_code/owq/log/llama2-7b-v3_fake.pth   --batch_size 4   --tasks  winogrande   --no_cache   --num_fewshot 0

# 5. Requirements:
We used Python libraries, and for that version, we have uploaded a requirements file for checking.

# 6. Methodology:
Not applicable; we depend on the baseline OWQ [1] paper and do not use any data processing or modeling.


# 7. Reference
[1] OWQ papers: https://github.com/xvyaward/owq/blob/main/README.md

## Citation
If you use this code or method in your research, please cite our manuscript:
[AdaOWQ], Under Review at PeerJ Computer Science, 2026.
