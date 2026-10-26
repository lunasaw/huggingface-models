# Hugging Face 模型部署脚本

按模型保留现有目录结构，集中管理本地 vLLM 启动、模型下载、接口验证和性能测试。仓库只包含代码、配置示例和文档；`models/` 权重、日志、验证结果和基准产物保留在本机，不提交 Git。

## 目录

```text
.
├── qwen38-27b-Base/                 # 官方 BF16 模型
│   ├── start-qwen38-bf16-vllm.sh    # 正式启动入口
│   ├── qwen38-bf16-vllm.service     # 当前机器的 systemd 配置
│   ├── download-qwen38-mirror.py    # 固定版本下载与哈希校验
│   ├── verify-qwen38-service.py     # 中文、流式、工具调用、图片检查
│   └── metadata/mirror-tree.json   # 下载必需的固定版本文件清单
├── qwen38-27b-Uncensored-HauhauCS/  # 原 Hauhau 入口，兼容转发到 BF16
├── benchmarks/hauhau-local-20261024/
│   ├── run_matrix.py              # 完整压测矩阵
│   ├── run_matrix_safe.py         # 较低负载的矩阵
│   └── build_report.py            # 汇总 JSON，生成 CSV、图表、HTML
├── docs/                          # 部署记录、历史概念说明
├── tests/                         # 不依赖 GPU 的回归检查
├── .env.example                   # 启动配置示例
├── requirements-tools.txt         # 验证和报告工具依赖
├── models/                        # 本地模型（忽略）
├── logs/                          # 历史日志（忽略）
└── results/                       # 下载、验证、基准输出（忽略）
```

目录名中的连字符保持不变，Python 脚本直接执行，不要求作为 Python 包安装。

## 环境

工具脚本使用 Python 3.11+，下载还需要 `curl`。本机已验证的服务环境为 `/data/miniconda3/envs/qwen38-hauhau-vllm`，vLLM 0.27.1、PyTorch 2.13.0；迁移到其他机器时应准备与 GPU / CUDA 匹配的 vLLM 环境。工具依赖可以单独安装：

```bash
python3 -m pip install -r requirements-tools.txt
```

启动脚本优先使用显式配置的 `VLLM_BIN`，否则使用上述本机环境；该路径不存在时查找 `PATH` 中的 `vllm`。

## 下载与启动

以下命令在仓库根目录执行，脚本本身也支持从其他工作目录调用。

```bash
cp .env.example .env
# 按机器配置编辑 .env，尤其是 VLLM_BIN、CUDA_VISIBLE_DEVICES 和 TENSOR_PARALLEL_SIZE。

# 查看下载计划；不会联网或写模型文件。
python3 qwen38-27b-Base/download-qwen38-mirror.py --dry-run

# 固定版本下载、断点续传、逐文件哈希校验。
python3 qwen38-27b-Base/download-qwen38-mirror.py

# 仅校验已有文件；会读取完整权重。
python3 qwen38-27b-Base/download-qwen38-mirror.py --verify-only

# 检查启动命令，然后前台启动。
bash qwen38-27b-Base/start-qwen38-bf16-vllm.sh --dry-run
bash qwen38-27b-Base/start-qwen38-bf16-vllm.sh
```

默认使用 GPU `4,5,6,7` / TP=4、端口 `18089`、BF16 权重、FP8 KV cache、262144 上下文、最大 2 个序列、MTP 3 tokens。主模型名为 `Qwen3.8 27b`，兼容别名为 `qwen3.8-27b`。参数保留现有部署设置，不保证适用于其他显存容量。

配置加载顺序为：脚本默认值、进程环境变量、`.env`；追加的 vLLM 命令行参数最后传入。可用 `QWEN_ENV_FILE=/path/to/config` 指定配置文件。`.env` 按 Bash 语法加载，只应放置可信本地配置。模型路径可设为 `MODEL_DIR`，下载脚本使用对应的 `--model-dir`。

`qwen38-27b-Uncensored-HauhauCS/start-hauhau-vllm.sh` 保留此前的兼容行为：启动官方 BF16 服务。该入口不加载磁盘上的历史 AWQ 权重。

## systemd

服务文件对应本机路径 `/data/ai/chenzhangyue/huggingface-models`，迁移时修改 `WorkingDirectory` 和 `ExecStart`。日志写入 journal，不依赖预先创建 `logs/`。

```bash
sudo cp qwen38-27b-Base/qwen38-bf16-vllm.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now qwen38-bf16-vllm.service
journalctl -u qwen38-bf16-vllm.service -f
```

已有服务正在运行时，`daemon-reload` 只更新配置；需要应用新启动参数时再执行 `sudo systemctl restart qwen38-bf16-vllm.service`。

## 接口验证

```bash
curl -fsS http://127.0.0.1:18089/health
python3 qwen38-27b-Base/verify-qwen38-service.py \
  --base-url http://127.0.0.1:18089
```

验证脚本会发送少量真实请求，检查两个模型名、中文回复、默认关闭思考、流式、工具调用和红色图片识别；不会执行模型提出的工具调用。结果默认写入 `results/qwen38-bf16-deploy/api-verification.json`，可通过 `--results-dir` 改变目录。模型名称和上下文长度也可通过命令行配置。

## 基准与报告

历史矩阵默认指向 Hauhau 模型别名；服务切换为官方 BF16 后需显式传入当前模型和 tokenizer。运行前会检查服务健康和模型名。默认输出至 `results/hauhau-benchmark-时间戳/`，不会覆盖原历史数据。

```bash
python3 benchmarks/hauhau-local-20261024/run_matrix_safe.py \
  --model qwen3.8-27b \
  --tokenizer models/Qwen3.8-27B \
  --vllm-bin /data/miniconda3/envs/qwen38-hauhau-vllm/bin/vllm \
  --output-dir results/bf16-benchmark \
  --dry-run

# 确认计划后去掉 --dry-run 执行压测，再生成报告。
python3 benchmarks/hauhau-local-20261024/build_report.py \
  --input-dir results/bf16-benchmark
```

`run_matrix.py` 是完整压力矩阵；`run_matrix_safe.py` 对长上下文使用较低并发，但仍会产生实际负载。失败请求和未执行用例会分别计入统计。

## 本地检查

```bash
python3 -m unittest discover -s tests -v
bash -n qwen38-27b-Base/start-qwen38-bf16-vllm.sh
bash -n qwen38-27b-Uncensored-HauhauCS/start-hauhau-vllm.sh
```

回归检查不下载权重、不启动 vLLM、不向运行中的服务发请求。

## 部署资料

- [官方 BF16 部署记录](docs/qwen38-bf16-deployment.md)
- [Hauhau 历史部署概念说明](docs/hauhau-vllm-concepts.md)

模型文件的许可由各自上游仓库决定，本仓库不分发模型权重。
