# HauhauCS Qwen3.8-27B AWQ-MTP

本目录的启动脚本独立加载 Hauhau 量化权重，使用 `--quantization compressed-tensors`，不再转发到 Base。模型来自 twolven 发布的 vLLM 适配仓库，而非 GGUF 原始仓库。

## 项目文件

| 文件 | 用途 |
|---|---|
| `download-hauhau-mirror.py` | 固定版本镜像下载、断点续传和完整性校验 |
| `metadata/mirror-tree.json` | 上游该版本的全部 14 个文件及大小、哈希 |
| `start-hauhau-vllm.sh` | 本模型的 vLLM 启动入口，支持 `--dry-run` |
| `environment.yml` | Conda 环境定义，引用本目录 `requirements.txt` |
| `requirements.txt` | 与现有可运行环境一致的直接 Python 依赖 |
| `environment.lock.yml` | 当前 Linux x86_64 环境的完整 Conda / pip 版本快照 |
| `setup-env.sh` | 从任意目录创建本项目环境 |
| `.env.example` | 模型、GPU、端口、容量参数示例 |
| `verify-hauhau-service.py` | 模型名、中文、流式、工具调用、图像接口验证 |
| `hauhau-vllm.service` | 本机 systemd 配置示例 |

## 1. 创建 Conda 环境

以下命令在仓库根目录执行：

```bash
bash qwen38-27b-Uncensored-HauhauCS/setup-env.sh
conda activate qwen38-hauhau-vllm
python -m pip check
```

`CONDA_BIN=/path/to/conda` 可指定 Conda；脚本只创建环境，已有同名环境时由 Conda 拒绝覆盖。若本机同名环境已存在，先激活并检查依赖，或手动从项目目录使用 `conda env create -n 新名称 -f environment.yml`。

基础环境为 Python 3.12；直接依赖固定 vLLM 0.27.1、PyTorch 2.13.0、torchvision 0.28.0、torchaudio 2.11.0、Transformers 5.15.1、huggingface-hub 1.28.0、compressed-tensors 0.17.0、flashinfer-python 0.6.16.post3、Pillow 12.3.0。这组版本来自现有服务环境，`pip check` 通过，torch 运行时为 CUDA 13.0。

现有 vLLM wheel 与 PyPI 同名发行文件 SHA-256 一致：`98e9fc2a1ed8549a733c9d1b242e2002b82367da9e29e37801761438cb3a2670`，没有依赖仅本机存在的私有修改 wheel。安装仍需要匹配的 NVIDIA 驱动及 GPU；环境文件不安装宿主机驱动。

`environment.lock.yml` 保留现有环境的全部版本供审计或同平台恢复，不含机器前缀、令牌和 `file://` 依赖。它不是跨平台、带工件哈希的求解锁；不能将其等同于在所有机器上重建成功。需要按快照创建时，进入本项目目录运行 `conda env create -n 新名称 -f environment.lock.yml`。本次未新建环境或重新安装运行中的服务依赖。

环境创建命令应在项目目录执行，或使用 `setup-env.sh`，以正确解析相对路径 `requirements.txt`。网络受限时可在调用前设置 `PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple`；不在仓库保存全局镜像配置。

与 Base 相同的 Conda 基础依赖已通过 dry-run 求解，当前运行环境满足全部直接依赖；这不等于已在新机器完成全量安装和 GPU 启动。

## 2. 下载完整镜像

- 模型仓库：[twolven/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-AWQ-MTP](https://hf-mirror.com/twolven/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-AWQ-MTP)。
- 固定版本：`1b909afa8b1bc0eeff0d32420e26207b974f33a6`。
- 默认目标目录：仓库根目录下的 `models/HauhauCS-Qwen3.8-27B-Aggressive-AWQ-MTP/`。
- 默认下载源：`https://hf-mirror.com`，可通过 `--endpoint` 或 `HF_ENDPOINT` 改写。

```bash
python qwen38-27b-Uncensored-HauhauCS/download-hauhau-mirror.py --dry-run
python qwen38-27b-Uncensored-HauhauCS/download-hauhau-mirror.py
python qwen38-27b-Uncensored-HauhauCS/download-hauhau-mirror.py --verify-only
```

下载脚本逐项下载清单中的全部文件，不只下载权重；缺失文件交由 `huggingface_hub` 下载并恢复其缓存中的未完成传输。LFS 文件按 SHA-256 校验，普通文件按 Git blob SHA-1 校验。已有文件会先校验，大小相同也不能跳过哈希检查。脚本支持 `--model-dir`、`--results-dir`、`--workers`。

若已有文件被修改或损坏，脚本返回非零并记录失败项，不静默覆盖。确需恢复上游版本时使用 `--force-download`；不要对服务正在使用的权重目录执行覆盖下载。旧自定义下载器的 `.segments/` 不是 Hugging Face Hub 缓存，不能续用其分段。

校验结果写入 `results/hauhau-vllm/download-verification.json`；全部通过后另写 `verified-files.json` 和 `revision.txt`。`--dry-run` 和 `--verify-only` 不访问网络。验证脚本的结果输出目录与下载记录分开文件保存。

## 3. 配置与启动

```bash
cp qwen38-27b-Uncensored-HauhauCS/.env.example qwen38-27b-Uncensored-HauhauCS/.env
# 编辑 .env，设置本机 VLLM_BIN、CUDA_VISIBLE_DEVICES、TENSOR_PARALLEL_SIZE 等。
bash qwen38-27b-Uncensored-HauhauCS/start-hauhau-vllm.sh --dry-run
bash qwen38-27b-Uncensored-HauhauCS/start-hauhau-vllm.sh
```

默认 GPU `5,6` / TP=2、端口 `18090`、模型名 `qwen3.8-27b-hauhau-aggressive`。上下文 262144、BF16 计算、FP8 KV cache、最大 2 个序列、批处理 token 上限 8192、MTP 3 tokens，默认关闭思考。修改 GPU 数量时同步调整 TP；根据实际显存调整上下文和显存利用率。

两套默认 GPU 存在重叠，默认配置用于分别运行。若需同时运行，要先分配互不重叠且空闲的 GPU，并按显存调整配置。端口区分不代表显存已隔离。

启动器优先选择 `VLLM_BIN`，然后是激活 Conda 环境中的 vLLM，再回退至本机既有 `/data/miniconda3/envs/qwen38-hauhau-vllm/bin/vllm`，最后查询 `PATH`。systemd 不执行交互式 Conda 激活，应在项目 `.env` 中显式设置完整 `VLLM_BIN` 路径。

配置优先级：脚本默认值 < 进程环境变量 < 项目 `.env`；命令行附加参数原样传给 vLLM。`HAUHAU_ENV_FILE=/path/to/file` 可选择其他配置。Hauhau 不读取仓库根目录 `.env`，避免误用 Base 的模型名、端口和 GPU 配置。此前该入口转发 BF16；需要官方模型时请改用 Base 启动脚本。 `.env` 使用 Bash 语法，只加载可信配置。下载脚本不自动加载 `.env`；自定义 `MODEL_DIR` 时将同一路径传给下载脚本的 `--model-dir`。

## 4. 验证与 systemd

```bash
curl -fsS http://127.0.0.1:18090/health
python qwen38-27b-Uncensored-HauhauCS/verify-hauhau-service.py --base-url http://127.0.0.1:18090
```

验证会发送少量实际请求，不执行模型生成的工具调用；支持通过 `--model`、`--max-model-len` 和 `--results-dir` 指定被测配置。

服务文件按本机 `/data/ai/chenzhangyue/huggingface-models` 路径编写，其他机器先修改路径和 `.env`，确认 GPU 空闲后再安装：

```bash
sudo cp qwen38-27b-Uncensored-HauhauCS/hauhau-vllm.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now hauhau-vllm.service
journalctl -u hauhau-vllm.service -f
```

已有服务应用新启动参数需要 `systemctl restart`。仅修改仓库脚本不会自动启动另一个模型。本次整理未启动 Hauhau，也未重启 BF16。

[返回仓库说明](../README.md)
