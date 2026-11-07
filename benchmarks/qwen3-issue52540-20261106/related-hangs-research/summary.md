# 与当前服务相关的潜在卡死问题

检索日期：2026-11-06（America/New_York）

当前本地环境：vLLM 0.26.0、FlashInfer 0.6.14、RTX 5090/SM120、Qwen3.8-27B-NVFP4。当前服务已禁用 `FlashInferFP8ScaledMMLinearKernel`，但本地启动日志显示 FlashInfer top-k/top-p sampler 仍启用。

更新：用户确认该服务用于多图业务，并将接入 OpenViking。此前“当前是文本服务，因此多图问题不适用”的判断错误，应撤回。此前 30 分钟验证仅覆盖纯文本、temperature=0 负载。新的部署与多模态核对见 [OpenViking 多图风险核对](openviking/assessment.md)。

## 优先关注

| 问题 | 可能表现 | 对当前环境的相关性 |
|---|---|---|
| [FlashInfer #3610](https://github.com/flashinfer-ai/flashinfer/issues/3610) / [修复 PR #3615](https://github.com/flashinfer-ai/flashinfer/pull/3615) | SM120/SM121 上 top-k/top-p sampler 永久自旋；GPU 高利用率、低功耗、0 token/s、健康检查仍 200 | 高。当前 GPU 是 RTX 5090，且 vLLM 0.26.0 环境默认启用 sampler；与 #52540 的 FP8 GEMM 路径独立 |
| [FlashInfer #6237](https://github.com/flashinfer-ai/flashinfer/issues/6237) / [修复 PR #6239](https://github.com/flashinfer-ai/flashinfer/pull/6239) | cuDNN graph 被 LRU 淘汰后，CUDA Graph 仍使用已释放的 plan；可能 cuGraphLaunch 崩溃，也可能卡在 cuDNN kernel | 高。当前 FlashInfer 0.6.14、cuDNN 9.19；CUTLASS 绕过可避开这条 FP8 cuDNN 路径。修复已进入 FlashInfer 0.7.2.post1 |
| [vLLM #52247](https://github.com/vllm-project/vllm/issues/52247) | 任意不返回的 GPU kernel 会让 `get_output().synchronize()` 无限等待；`/health` 仍 200，SIGTERM 可能无效，只能重启 | 高，属于所有 GPU 卡死的放大器，而非单一触发器 |

## 条件性相关

- [vLLM #61239](https://github.com/vllm-project/vllm/issues/61239)：FlashInfer metadata workspace 与 async scheduling 竞争，可能非法访存或 GPU hang。主要针对 v0.31.0 的改动路径；升级到新版本并启用 async scheduling 时需要重新核对。
- [vLLM #58422](https://github.com/vllm-project/vllm/issues/58422)：Qwen3.8 GDN + MTP4 + NVFP4 KV + SM120 的静默 wedge。当前服务没有 MTP，KV 是 FP8，因此不是当前配置的直接复现条件。
- [vLLM #40707](https://github.com/vllm-project/vllm/issues/40707) / [候选修复 PR #40757](https://github.com/vllm-project/vllm/pull/40757)：Mamba 对齐、chunked prefill、多张大图导致 scheduler 零进展。用户确认存在多图业务，因此应优先验证；关键是单次模型请求是否包含多张大图，以及编码缓存与块对齐条件。#40709 已关闭但未合并，#40757 仍开放，不能当作现版已修复。
- [vLLM #37729](https://github.com/vllm-project/vllm/issues/37729)：较早的 GDN/FlashInfer/CUDA Graph 卡死系列；其中 FlashInfer ≤0.6.13 的 GDN prefill 卡死已在 0.6.14 修复，但它说明 GDN 路径仍需单独关注。

## 建议

1. 补做多图请求回归：同一请求的图片数量、视觉 token 数、图间文本、共同前缀、并发和长短请求混合都需要覆盖。生产端不直接投递已知可能卡死的构造输入。
2. 在 RTX 5090/SM120 服务中单独验证 `VLLM_USE_FLASHINFER_SAMPLER=0`；这与当前 FP8 线性层绕过配置是两项独立设置。#3615 已合并的内容仅为 reset race 修复，不包括其历史描述中的全部共驻留修复，不能把旧 issue 的“关闭”当作所有采样卡死已解决。
3. 继续保留 `VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel`。升级时核对整体依赖兼容性和目标修复，并重新跑多模态及采样回归；包含 #6239 不等于覆盖所有卡死原因。
4. 监控请求进度、prefill/decode 进度、运行/等待请求数与真实推理探针，不只检查 `/health`。阈值需要容纳正常长图预填充、排队和冷启动；不能仅因 generation 计数暂时不增长就自动重启。
