# Qwen3.8-27B 官方 BF16 部署记录

本文保留历史部署过程；当前可执行入口与操作步骤见[根目录 README](../README.md)。本文提及的验证 JSON 和日志只保存在本机 `results/qwen38-bf16-deploy/` 与 `logs/` 中。

## 模型文件

- 仓库：<https://huggingface.co/Qwen/Qwen3.8-27B>
- 下载镜像：<https://hf-mirror.com>
- 固定版本：`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- 路径：`/data/ai/chenzhangyue/huggingface-models/models/Qwen3.8-27B`
- 32 个仓库文件完整下载；18 个 safetensors 权重文件共 55,563,006,776 字节。
- 权重为 BF16，未应用 AWQ 或其他权重量化。
- `verified-files.json` 记录逐文件校验结果：LFS 文件核对 SHA-256，其余文件核对 Git blob SHA-1。`qwen38-27b-Base/metadata/mirror-tree.json` 保存对应版本的远端文件元数据。
- 模型仓库文件保持原样；重复惩罚通过 vLLM 启动参数覆盖。

## 原参数迁移

目标地址：`http://127.0.0.1:18089/v1`。

按用户要求，主模型名称为 `Qwen3.8 27b`（包含空格，大小写如上）。为兼容仍发送旧名称的客户端，同时接受 `qwen3.8-27b`；两者指向同一份官方 BF16 权重。

保留配置：BF16 计算、FP8 KV 缓存、最大上下文 262144、显存利用率 0.95、最大并发序列 2、批处理 token 上限 8192、前缀缓存、分块预填充、PIECEWISE CUDA graph、MTP 3 tokens、qwen3 推理解析、qwen3_coder 工具调用解析。

默认采样仍为 temperature=1.0、top_p=0.95、top_k=20、repetition_penalty=1.05；默认 enable_thinking=false、reasoning_effort=medium。客户端显式传入的设置可以覆盖服务默认值。

## 首次启动结果（2026-10-25）

原配置 GPU 5、6 / TP=2 的权重加载成功，每卡模型占用约 26.09 GiB，但 256K 上下文的 KV 缓存校验失败：需要 4.64 GiB，实际可用 1.87 GiB，vLLM 估计最大上下文为 91200。服务未就绪，未启用开机启动或失败重启。

日志：`logs/qwen38-bf16-vllm-18089.log`。

用户已于 2026-10-25 授权使用四张空闲卡。启动前确认 GPU 3、4、5、6 均无计算进程，改为这些卡 / TP=4，保留 256K 和其余配置。15:04 启动四卡服务，15:09 通过 vLLM KV 缓存检查并开始监听 18089。

15:15 发生 CUDA OOM，日志记录 TP0 所用 GPU（物理 GPU 3）还有其他进程占用约 788 MiB。改名操作开始时，服务已退出，GPU 3 仍有其他进程。为使用四张空闲卡，改用 GPU 4、5、6、7，TP=4 及推理参数保持不变。原 `Restart=on-failure` 未恢复该次故障，因为 API 主进程退出码为 0；现改为 `Restart=always`，显式 `systemctl stop` 仍会停止服务。

15:48 新名称 `Qwen3.8 27b` 的服务恢复健康，`/v1/models` 只返回该名称和 262144 上下文；使用此名称发送中文聊天请求返回 HTTP 200。验证记录：`rename-verification.json`。

16:37 恢复 `qwen3.8-27b` 作为兼容别名，修复客户端继续发送旧名称时的 HTTP 404。主名称仍为 `Qwen3.8 27b`，两个名称的实际聊天请求均返回 HTTP 200。验证记录：`alias-verification.json`。

## 启动及验证文件

- `qwen38-27b-Base/start-qwen38-bf16-vllm.sh`：新模型启动命令。
- `qwen38-27b-Base/qwen38-bf16-vllm.service`：systemd 服务配置；已启用开机启动和失败自动恢复。
- `qwen38-27b-Base/verify-qwen38-service.py`：使用真实 API 请求检查模型名、中文回复、关闭思考、流式输出、工具调用和图片输入。
- `qwen38-27b-Base/download-qwen38-mirror.py`：镜像下载与文件哈希校验脚本。

旧量化模型文件暂留磁盘，不作为新服务权重。首次四卡 API 验证结果保存在 `api-verification.json`，包含中文默认回复、当时的兼容别名、流式输出、工具调用和图片识别。
