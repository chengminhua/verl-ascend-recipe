# True On-Policy（训推一致性 RL）

在昇腾 NPU 上，通过 **Megatron（MindSpeed）训练 + vLLM-Ascend Rollout** 实现 GRPO / GSPO / DAPO 强化学习，并在训练侧与推理侧同时启用 **训推一致性（True On-Policy）** 对齐训推差异。

---

## 版本配套

| 组件 | 版本 | 备注                     |
| --- | --- |------------------------|
| CANN | 9.0.0.B160（CANN900B160） |                        |
| Python | 3.11 |                        |
| PyTorch / torch_npu | 2.9.0 | 随 PTA B120             |
| vLLM | 0.18.0 |                        |
| vLLM-Ascend | v0.18.0 + [PR #10375](https://github.com/vllm-project/vllm-ascend/pull/10375) | FA3 batch-invariant 适配 |
| Megatron-LM | `3bec9aa97dda898d16ff5a89bac0ed2b6682b172` |                        |
| MindSpeed | `core_r0.16.0` |               |
| verl | `release/v0.8.0` + [PR #6678](https://github.com/verl-project/verl/pull/6678) | PP 适配                  |
| triton_ascend | 3.2.1 | CANN 9.0 需此版本          |

---

## 目录结构

```
true_on_policy/
├── README.md                 # 本文档
├── REQUIRED_VERL.txt         # verl 版本钉扎
├── patch/
├── ├── batch_invariant_ops.py
│   └── npu_true_on_policy_patch.py
└── scripts/
    ├── grpo/   run_grpo_qwen3_4b_megatron_npu.sh  run_grpo_qwen3_30b_megatron_npu.sh
    ├── gspo/   run_gspo_qwen3_4b_megatron_npu.sh  run_gspo_qwen3_30b_megatron_npu.sh
    └── dapo/   run_dapo_qwen3_4b_megatron_npu.sh  run_dapo_qwen3_30b_megatron_npu.sh
```

---

## 环境搭建

以下步骤在 Linux + Ascend NPU 机器上操作。安装 vLLM-Ascend、flash-attention-npu、batch_invariant ops 前需 **先 `source` CANN 环境**。

### 1. Python 虚拟环境

```bash
conda create -n verl_main python=3.11 -y
conda activate verl_main
```

### 2. 推理栈（vLLM + vLLM-Ascend）

```bash
pip install vllm==0.18.0

git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git checkout releases/v0.18.0
git fetch origin pull/10375/head:pr-10375
git merge pr-10375 --no-edit
pip install -r requirements.txt
export COMPILE_CUSTOM_KERNELS=1
pip install -v -e .
cd ..

pip install triton-ascend==3.2.1 \
  --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple \
  --trusted-host triton-ascend.osinfra.cn
pip install setuptools==80.9.0
```

### 3. 训练栈（verl + MindSpeed + Megatron-LM）

```bash
# verl（含 PP 适配 PR）
git clone https://github.com/verl-project/verl.git
cd verl
git checkout release/v0.8.0
git fetch origin pull/6678/head:pr-6678
git merge pr-6678 --no-edit
git submodule update --init --recursive recipe   # DAPO 等算法 recipe，见下文
pip install -v -e .
cd ..

# 将本 recipe 放入 verl 树（训推 patch + 启动脚本）
git clone https://github.com/verl-project/verl-ascend-recipe.git
mkdir -p verl/verl_ascend_recipe
cp -r verl-ascend-recipe/true_on_policy verl/verl_ascend_recipe/

# MindSpeed
git clone https://gitcode.com/Ascend/MindSpeed.git
cd MindSpeed
git checkout core_r0.16.0
pip install -r requirements.txt
pip install -v -e .
cd ..

# Megatron-LM
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout 3bec9aa97dda898d16ff5a89bac0ed2b6682b172
pip install -v -e .
cd ..
```

### 4. 其他 Python 依赖

```bash
pip install uvloop==0.21.0 torch==2.9.0 torch_npu==2.9.0 torchaudio==2.9.0 \
  torchdata==0.11.0 torchvision==0.24.0 ray==2.53.0 mbridge==0.15.1
pip install "nvidia-modelopt[torch]>=0.37.0" transformers==5.3.0 \
  flash-linear-attention==0.5.0 setuptools==80.9.0 qwen_vl_utils mathruler viztracer
pip uninstall -y torch_c_dlpack_ext || true
```

### 5. Flash Attention NPU（FA3）

```bash
# 安装前 source CANN
git clone https://github.com/MinghuasLab/flash-attention-npu.git
cd flash-attention-npu
git checkout trainlnferConsist
git fetch origin pull/1/head:pr-1
git merge pr-1 --no-edit
git submodule update --init --recursive
python setup.py install
cd ..
```

### 6. batch_invariant ops（训推一致性训练侧算子）

按需下载对应平台的 `.run` 包并安装（安装前 source CANN）：

| 平台         | 下载 |
|------------| --- |
| A2 aarch64 | https://ascend-cann-open.obs.cn-north-4.myhuaweicloud.com/ops-batchinvariant/beta/20260611/cann-ops-batch_invariant-910b-1.0.0-linux.aarch64.run |
| A2 x86_64  | https://ascend-cann-open.obs.cn-north-4.myhuaweicloud.com/ops-batchinvariant/beta/20260611/cann-ops-batch_invariant-910b-1.0.0-linux.x86_64.run |
| A3 aarch64 | https://ascend-cann-open.obs.cn-north-4.myhuaweicloud.com/ops-batchinvariant/beta/20260611/cann-ops-batch_invariant-A3-1.0.0-linux.aarch64.run |
| A3 x86_64  | https://ascend-cann-open.obs.cn-north-4.myhuaweicloud.com/ops-batchinvariant/beta/20260611/cann-ops-batch_invariant-A3-1.0.0-linux.x86_64.run |

```bash
chmod +x cann-ops-batch_invariant-*.run
./cann-ops-batch_invariant-*.run
```

Python 扩展包（`batch_invariant_ops` whl）：

1. 下载并解压 [batch_invariant-torch_ops_extension-1.0.0.zip](https://ascend-cann-open.obs.cn-north-4.myhuaweicloud.com/ops-batchinvariant/beta/20260611/batch_invariant-torch_ops_extension-1.0.0.zip)
2. 进入 `torch_ops_extension/batch_invariant_ops`，执行 `bash build_and_install.sh`
3. 安装 `dist/batch_invariant_ops-1.0.0-*.whl`

---

## 数据准备

脚本通过环境变量指定数据路径，请将 parquet 文件放到对应目录（或覆盖 `TRAIN_FILE` / `VAL_FILE`）：

| 算法 | 默认训练集 | 默认验证集 |
| --- | --- | --- |
| GRPO / GSPO（4B） | `$HOME/data/gsm8k/train.parquet` | `$HOME/data/gsm8k/test.parquet` |
| GRPO / GSPO（30B） | `$HOME/data/dapo-math-17k.parquet` | 同训练集 |
| DAPO | `$HOME/data/dapo-math-17k.parquet` | `$HOME/data/aime-2024.parquet` |

DAPO 数据可参考上游 `recipe/dapo/prepare_dapo_data.sh`（需先初始化 `recipe` submodule，见下节）。

---

## DAPO 额外准备（recipe submodule）

GRPO / GSPO 与 DAPO 使用 **不同的 Python 训练入口**：

| 算法 | Python 入口 | 代码位置 |
| --- | --- | --- |
| GRPO / GSPO | `verl.trainer.main_ppo` | verl 核心库 |
| DAPO | `recipe.dapo.main_dapo` | verl 的 `recipe/` submodule |

verl 将 DAPO 等算法 recipe 迁移至独立仓库 [verl-recipe](https://github.com/verl-project/verl-recipe)，以 git submodule 挂载在 verl 仓库的 `recipe/` 路径。**跑 DAPO 脚本前必须初始化该 submodule**：

```bash
cd verl   # verl 仓库根目录
git submodule update --init --recursive recipe
```

完成后 DAPO 实现位于 `recipe/dapo/`（`main_dapo.py`、`dapo_ray_trainer.py`、`config/dapo_megatron_trainer.yaml` 等）。本目录 `scripts/dapo/*.sh` 的启动命令为：

```bash
python3 -m recipe.dapo.main_dapo --config-name=dapo_megatron_trainer ...
```

---

## 快速开始

### 1. 激活环境

```bash
conda activate verl_main
source ${ASCEND_HOME}/set_env.sh          # CANN
# source ${ASCEND_HOME}/../nnal/atb/set_env.sh   # 若使用 ATB，按实际路径调整
cd verl                                   # verl 仓库根目录
```

### 2. 配置模型与数据（示例）

```bash
export MODEL_PATH=/path/to/Qwen3-4B              # 或 HuggingFace 模型 ID
export TRAIN_FILE=/path/to/train.parquet
export VAL_FILE=/path/to/test.parquet
export NGPUS_PER_NODE=16                         # 单机 NPU 数
```

### 3. 运行脚本

在 **verl 仓库根目录** 执行（脚本会在 `./logs/` 下写日志）：

```bash
# GRPO — Qwen3-4B dense
bash verl_ascend_recipe/true_on_policy/scripts/grpo/run_grpo_qwen3_4b_megatron_npu.sh

# GRPO — Qwen3-30B-A3B MoE
bash verl_ascend_recipe/true_on_policy/scripts/grpo/run_grpo_qwen3_30b_megatron_npu.sh

# GSPO — Qwen3-4B / 30B
bash verl_ascend_recipe/true_on_policy/scripts/gspo/run_gspo_qwen3_4b_megatron_npu.sh
bash verl_ascend_recipe/true_on_policy/scripts/gspo/run_gspo_qwen3_30b_megatron_npu.sh

# DAPO — Qwen3-4B / 30B（需先 `git submodule update --init --recursive recipe`）
bash verl_ascend_recipe/true_on_policy/scripts/dapo/run_dapo_qwen3_4b_megatron_npu.sh
bash verl_ascend_recipe/true_on_policy/scripts/dapo/run_dapo_qwen3_30b_megatron_npu.sh
```

各算法训练入口：

| 脚本目录 | Python 模块 |
| --- | --- |
| `scripts/grpo/`、`scripts/gspo/` | `verl.trainer.main_ppo` |
| `scripts/dapo/` | `recipe.dapo.main_dapo` |

### 4. 训推一致性开关

所有脚本默认 **开启** 训推一致性（`ENABLE_TRUE_ON_POLICY=1`）。关闭后可做 baseline 对比：

```bash
ENABLE_TRUE_ON_POLICY=0 bash verl_ascend_recipe/true_on_policy/scripts/grpo/4b.sh
```

开启时会同时生效：

| 层级 | 机制 |
| --- | --- |
| Rollout（vLLM） | `VLLM_BATCH_INVARIANT=1` + `VERL_USE_EXTERNAL_MODULES` 加载 `npu_patch` |
| 训练（Megatron） | MindSpeed `batch_invariant_mode` / `use_batch_invariant_ops` 等 Hydra 参数 |

---

## 参考

- 精度 / 训推一致性排查：[transfer_to_npu_guide.md](../../docs/ascend_tutorial/dev_guide/model_dev/transfer_to_npu_guide.md)