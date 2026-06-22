# GLM5 on Ascend NPU
本recipe是基于GLM5模型在NPU上进行RLHF后训练的样例，基于GRPO与规则奖励，使用dapo-math17k数据集进行训练，使用数据集mmlu、gpqa、aime2024、aime2025、BFCL进行评测，同样适用于GLM5.1模型


# 环境配套

| **组件**  | **配套版本** | **备注**                     |
| ----------------- | -------------------- | ------------------------------------ |
| CANN            | 8.5.0/9.0.0 | |
| python          | 3.11               |                                    |
| pytorch         | 2.9.0              |                                    |
| vllm            | v0.18.0	 | |
| vllm-ascend     | releases/v0.18.0| |
| verl            | release/v0.8.0 |   |
| transformers    | v5.3.0             | |
| Megatron        | core0.16.0 | |
| Megatron-Bridge | v0.3.1 | |
| MindSpeed       | core_0.16.0 |   |


# 激活CANN

```
source /CANN/900B160/ascend-toolkit/set_env.sh
source /CANN/900B160/nnal/atb/set_env.sh
```

# 环境安装

## 安装transformers

```bash
git clone https://github.com/huggingface/transformers.git -b v5.3.0
cd transformers
pip install -e .
cd ..
```

## 安装vllm

```bash
git clone https://github.com/vllm-project/vllm.git -b v0.18.0
cd vllm
git apply ../verl-ascend-recipe/GLM5/patch/vllm.patch
pip3 install -r requirements/common.txt
pip3 install -r requirements/build.txt
pip install torch==2.9.0
pip install torch_npu==2.9.0.post2
VLLM_TARGET_DEVICE=empty pip install -v -e . 
cd ..
```

## 安装vllm_ascend

```bash
git clone https://github.com/vllm-project/vllm-ascend.git -b releases/v0.18.0
cd vllm-ascend
git apply ../verl-ascend-recipe/GLM5/patch/vllm_ascend.patch
pip install -r requirements-dev.txt
pip install -v -e .
cd ..
# 装triton_ascend
pip install triton-ascend==3.2.1 --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple --trusted-host triton-ascend.osinfra.cn
```

## 安装verl

```bash
git clone https://github.com/volcengine/verl.git -b release/v0.8.0
cd verl
git apply ../verl-ascend-recipe/GLM5/patch/verl.patch
pip install -r requirements-npu.txt
pip install -v -e .
cd ..
```

## 安装MindSpeed

```bash
git clone https://gitcode.com/Ascend/MindSpeed.git -b core_r0.16.0
cd MindSpeed
git apply ../verl-ascend-recipe/GLM5/patch/mindspeed.patch
pip install -e .
cd ..
```

## 安装Megatron-LM

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git -b core_v0.16.0
cp -r Megatron-LM/megatron verl/
cd ..
```

## 安装Megatron-Bridge

```bash
git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git -b v0.3.1
cd Megatron-Bridge
git apply ../verl-ascend-recipe/GLM5/patch/megatron-bridge.patch
cd ..
cp -r Megatron-Bridge/src/megatron/bridge verl/megatron
```

# 训练启动


```bash
cd verl
# 修改ray_start.sh中对应的网卡、主节点IP、权重、数据集地址
bash ../verl-ascend-recipe/GLM5/scripts/ray_start.sh
```