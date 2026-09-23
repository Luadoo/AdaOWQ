# 1. Title: Adaptive Outlier Allocation Protection for Three-Tier Weight Quantization in Large Language Models

## This is a submitted paper to the PeerJ Computer Science journal for peer review.

# 2. Description:
AdaOWQ is an adaptive, three-tier mixed-precision post-training quantization (PTQ) framework for Large Language Models. Building upon OWQ, AdaOWQ incorporates activation energy with Hessian sensitivity to dynamically allocate outlier protection budgets across layers, capturing secondary critical channels with an efficient FP16/INT8/INT3-4 precision hierarchy.

# 3. Dataset Information: (Custom medical datasets)
* /your file location/utils/custom_data.json.gz (This dataset is from Unisound AI Technology Co., Ltd., Beijing, China, for end-device inference and to protect private user data information)

# 4. Code Information:

### (1)Environment setup
* conda create -n myllm python=3.10 -y
* conda activate myllm
*  print(torch.__version__)  2.13.0+cu130

<img width="590" height="140" alt="image" src="https://github.com/user-attachments/assets/0d09be45-f94e-43e2-b90b-e672819ed464" />




### (2)Install all the dependencies
pip install -r requirements.txt

### (3)Install CUDA kernel (3/4bit_W x FP16_A)
* cd owq/kernel
* python setup_cuda.py install
* In our paper's limitations, we clarify that although AdaOWQ reduces memory footprint by physically packing Tier-2 columns into 8-bit integers, executing mixed-precision matrix multiplications (FP16, INT8, and INT3/INT4) in a single layer still requires specialized CUDA/NPU runtime kernels to achieve linear wall-clock latency speedups on real-world edge hardware. This is left for future work.

## For quantization (Usage Instructions)
1. CUDA_VISIBLE_DEVICES=4 python main.py  /home/xujie-intern/.cache/huggingface/hub/llama-2-13b-hf/snapshots/v1/  c4 --wbits 3 --target_bit 3.01  --expansion_factor 1.5  --tier1_ratio 0.3  --tier2_bits 8  --act-order  --true-sequential --seed 10000  --fake  --save  /data-model/infer-r1/xujie/my_code/owq/log/llama2-13-3.01-v10.pth 2>&1 | tee /data-model/infer-r1/xujie/my_code/owq/log/213b-3.01-v10.log


## For evaluation on zero-shot (Usage Instructions)
2. CUDA_VISIBLE_DEVICES=5  python zeroshot.py   --model hf-causal-owq   --model_args pretrained=/home/xujie-intern/.cache/huggingface/hub/llama2-13b-hf/snapshots/v1,load=/data-model/infer-r1/xujie/my_code/owq/log/llama2-13-3.01-v10_fake.pth   --batch_size 4   --tasks  winogrande   --no_cache   --num_fewshot 0

# 5. Requirements:
We used Python libraries, and for that version, we have uploaded a requirements file for checking.

# 6. Methodology:
Not applicable; we depend on the baseline OWQ [1] paper and do not use any data processing or modeling.


# 7. Reference
[1] OWQ papers: https://github.com/xvyaward/owq/blob/main/README.md

## Citation
If you use this code or method in your research, please cite our manuscript:
[AdaOWQ], Under Review at PeerJ Computer Science, 2026.


## License
This work is licensed under the MIT License.
Code provided for peer review of manuscript [Paper ID: #149231].
