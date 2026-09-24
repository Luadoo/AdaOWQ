# 1. Title: Adaptive Outlier Allocation Protection for Three-Tier Weight Quantization in Large Language Models

## This is a submitted paper to the PeerJ Computer Science journal for peer review.

# 2. Description:
AdaOWQ is an adaptive, three-tier mixed-precision post-training quantization (PTQ) framework for Large Language Models. Building upon OWQ, AdaOWQ incorporates energy activation with Hessian sensitivity to dynamically allocate outlier protection budgets across layers, capturing secondary critical channels with an efficient FP16/INT8/INT3-4 precision hierarchy.

# 3. Dataset Information: (Custom medical datasets)
* /your file location/utils/custom_data.json.gz (This dataset is from Unisound AI Technology Co., Ltd., Beijing, China, for end-device inference and to protect private user data information)

# 4. Code Information:

### (1) Environment setup





### (2) Install all the dependencies
pip install -r requirements.txt

### (3) Install CUDA kernel (3/4bit_W x FP16_A)
* cd owq/kernel
* python setup_cuda.py install
* In our paper's limitations, we clarify that although AdaOWQ reduces memory footprint by physically packing Tier-2 columns into 8-bit integers, executing mixed-precision matrix multiplications (FP16, INT8, and INT3/INT4) in a single layer still requires specialized CUDA/NPU runtime kernels to achieve linear wall-clock latency speedups on real-world edge hardware. This is left for future work.

## For quantization (Usage Instructions)
1. 


## For evaluation on zero-shot (Usage Instructions)
2. 

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
