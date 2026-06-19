# 题目二：基于 LeRobot 的 ACT 跨环境泛化实验

本工程实现作业题目二：在 CALVIN 数据集上训练 ACT 策略，比较单环境 B 训练与 A/B/C 联合训练在未见环境 D 上的零样本 Action L1。

## 目录说明

- `task2_act/`：助教 LeRobot 数据读取、训练、评测与绘图代码。
- `scripts/bootstrap_autodl.sh`：Ubuntu 22.04 服务器环境安装脚本。
- `scripts/run_full_experiment.sh`：完整实验入口。
- `configs/experiment.yaml`：本次实验固定配置。
- `tests/`：轻量自测，使用合成 CALVIN 小数据。

## 服务器环境

目标环境：Linux、Python 3.12、CUDA GPU。完整实验在 NVIDIA V100 上完成。

```bash
bash scripts/bootstrap_autodl.sh
```

本工程使用助教提供的划分版数据集：

https://huggingface.co/datasets/xiaoma26/calvin-lerobot/tree/main

下载后目录应包含：

```text
calvin-lerobot/
  splitA/
  splitB/
  splitC/
  splitD/
```

## 一键运行

```bash
cd task2_act_server_package
bash run_on_server.sh /data/calvin-lerobot /data/cv_hw3_task2
```

脚本会自动选择环境方式：检测到 `conda` 时创建 `cv_hw3_act`，检测不到时创建当前目录下的 `.venv`。

如果 `/data/calvin-lerobot` 里还没有数据，脚本会自动从 Hugging Face 下载完整的 `splitA/B/C/D`。下载慢时可以先设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

等价的手动方式：

```bash
python -m task2_act.download_ta_dataset --output-root /data/calvin-lerobot
export TA_DATA_ROOT=/data/calvin-lerobot
export WORK_ROOT=/data/cv_hw3_task2
bash scripts/run_full_experiment.sh
```

默认实验设置：

- B 模型：环境 B，120000 帧。
- ABC 模型：环境 A/B/C 各 40000 帧，总计 120000 帧。
- D 评测集：环境 D，40000 帧。
- ACT：`chunk_size=50`，`n_action_steps=10`，batch size 8，100000 steps。
- 主指标：D 环境 Action L1。
- 训练曲线：本地 `train_metrics.csv`，同时使用 WandB offline 记录。
- 存储占用：全量 `splitA/B/C/D` 约 69.9GB；加上环境、缓存和训练输出，建议预留 120GB 以上。

## 单步运行

下载助教数据：

```bash
python -m task2_act.download_ta_dataset --output-root /data/calvin-lerobot
```

训练 B 模型：

```bash
python -m task2_act.train_act \
  --ta-split-root /data/calvin-lerobot/splitB \
  --target-frames 120000 \
  --output-dir /data/cv_hw3_task2/outputs/act_b \
  --wandb-project cv_hw3_act \
  --wandb-mode offline
```

训练 ABC 模型：

```bash
python -m task2_act.train_act \
  --ta-split-root /data/calvin-lerobot/splitA \
  --ta-split-root /data/calvin-lerobot/splitB \
  --ta-split-root /data/calvin-lerobot/splitC \
  --frames-per-split 40000 \
  --output-dir /data/cv_hw3_task2/outputs/act_abc \
  --wandb-project cv_hw3_act \
  --wandb-mode offline
```

评测：

```bash
python -m task2_act.evaluate_action_l1 \
  --ta-eval-root /data/calvin-lerobot/splitD \
  --target-frames 40000 \
  --model b=/data/cv_hw3_task2/outputs/act_b/checkpoints/step_040000 \
  --model abc=/data/cv_hw3_task2/outputs/act_abc/checkpoints/step_040000 \
  --output-dir /data/cv_hw3_task2/eval_d
```

评测全部五个检查点并复现报告中的检查点曲线：

```bash
export TA_DATA_ROOT=/data/calvin-lerobot
export WORK_ROOT=/data/cv_hw3_task2
bash scripts/evaluate_checkpoints.sh
python report/generate_figures.py
```

## 本地轻量自测

当前 Windows 本地没有 GPU，可以用合成数据检查解析与指标逻辑：

```bash
python -m pip install pytest
python -m task2_act.make_mock_calvin --output-root tmp/mock_calvin --frames-per-scene 12
python -m pytest -q
```

完整训练需要在服务器环境执行。

## 实验结果

所有检查点均在未参与训练的 CALVIN D 上评测。Action L1 越低越好。

| Training step | B only | A+B+C |
|---:|---:|---:|
| 20,000 | 0.201016 | 0.181526 |
| **40,000** | **0.199815** | **0.180052** |
| 60,000 | 0.200922 | 0.185826 |
| 80,000 | 0.200498 | 0.185518 |
| 100,000 | 0.204507 | 0.188230 |

两组模型的最低 D 环境 Action L1 均出现在 40,000 step。A+B+C 联合训练相对 B-only 将误差降低 9.89%。完整数值、训练曲线和评测曲线位于 `results/` 与 `report/figures/`。

## 模型权重

最终提交使用以下两个检查点：

- `act_b/checkpoints/step_040000`
- `act_abc/checkpoints/step_040000`

权重下载地址：`<MODEL_WEIGHTS_URL>`

下载并解压后，将两个目录放置为：

```text
weights/
  act_b_step_040000/
    config.json
    model.safetensors
    policy_preprocessor.json
    policy_preprocessor_step_3_normalizer_processor.safetensors
    policy_postprocessor.json
    policy_postprocessor_step_0_unnormalizer_processor.safetensors
  act_abc_step_040000/
    config.json
    model.safetensors
    policy_preprocessor.json
    policy_preprocessor_step_3_normalizer_processor.safetensors
    policy_postprocessor.json
    policy_postprocessor_step_0_unnormalizer_processor.safetensors
```

评测下载后的权重：

```bash
python -m task2_act.evaluate_action_l1 \
  --ta-eval-root /data/calvin-lerobot/splitD \
  --target-frames 40000 \
  --model b=weights/act_b_step_040000 \
  --model abc=weights/act_abc_step_040000 \
  --output-dir /data/cv_hw3_task2/eval_downloaded_weights
```

