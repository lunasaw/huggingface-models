# OpenViking 多图调用风险核对

报告更新时间：2026-11-06 04:49 EST（America/New_York）。原始日志保留宿主机时间与偏移。

## 修正后的结论

用户确认 Qwen 服务存在多图业务，并会接入 OpenViking。因此 vLLM #40707 的多图调度停滞应列为优先验证项。之前“文本服务不适用”的表述错误：此前仅执行了文本压测，不能推断业务只有文本。

目前证据足以确认模型架构、部署设置及本地对齐函数具有相关特征，尚不足以认定 OpenViking 或现网已触发该卡死。

## 已核实事实

- 用户给出的 [OpenViking Studio](http://service-manager-vd-openviking.daily.idcvdian.com/studio/home) 可访问；`/health` 返回 healthy，版本 0.4.13，auth_mode=api_key。
- `prod/service-manager-vd-openviking` 的 4 个 Pod 在查询时全部 Running/Ready，重启次数均为 0。该状态不能证明所有后台任务都有进展。
- OpenViking 的 `/api/v1/observer/models`、`/api/v1/observer/queue` 返回 401，因此没有获取实际 VLM 地址、超时、重试和并发配置。未绕过认证，也未从首页推断其请求格式。
- 现网 `service-manager-vllm-qwen3-0`：vLLM 0.26.0，TP=1，speculative_config=None；max-num-seqs=4，max-num-batched-tokens=8192；chunked prefill、prefix caching、CUDA Graph 开启。
- 该 Pod 启动日志记录：图像 encoder cache 预算 16384 tokens，按 1 张最大特征尺寸图片做 profile。此处的 1 张是 profile 设置，不能解释为接口最多允许 1 张图。
- FP8 线性层使用 CutlassFP8ScaledMMLinearKernel；GDN prefill 使用 Triton/FLA；FlashInfer top-k/top-p sampler 仍启用。
- 现网未设置 `--generation-config vllm`，日志确认模型默认采样参数 temperature=1.0、top_k=20、top_p=0.95；客户端显式参数可以覆盖它们。
- vLLM 的 readiness 和 liveness 均调用 `/health`；没有从这份 Pod spec 看到真实推理进度探针。
- 本地指定模型的 config 为 `Qwen3_5ForConditionalGeneration`，视觉 patch_size=16、spatial_merge_size=2。模型名称中的 Qwen3.8 不改变其实际配置所指示的 Qwen3.5 架构实现。

## 多图调度问题如何触发

[vLLM #40707](https://github.com/vllm-project/vllm/issues/40707) 描述的组合是：混合 GDN/Mamba 模型、缓存块对齐、分块预填充及多张大图。当图像编码缓存只能容纳其中一张大图，调度器必须先消费第一张图才能装入第二张；若图间剩余可推进 token 少于对齐块大小，而对齐把本次推进量变为 0，请求可能永远无法前进。

表现可以是显存仍占用、GPU 计算利用率低或为 0、请求无输出，区别于 GPU kernel 自旋时的高利用率。单卡和较低 max-num-seqs 不能独立排除单请求触发。

上游案例中的 3024×4032 图片、11844 个视觉 token、16-token block 是特定模型和参数下的示例；不能不经实际 processor 和运行配置核对就当作当前服务的精确阈值。

本地仅提取已安装 vLLM 的 `_mamba_block_aligned_split` 函数做 CPU 调用：构造 block_size=16、已处理 11872、可处理 4、总长 23721，得到 0；可处理 16/32 时分别返回 16/32。该观察确认了“可调度量可能被对齐到 0”，不代表已运行完整 scheduler、多模态 encoder、GPU 模型，或已复现现网故障。当前生产实际 block_size、mamba cache mode 和图片 token 序列尚未完整核实。

截至本次查询，#40707 开放；候选修复 [#40757](https://github.com/vllm-project/vllm/pull/40757) 仍开放、未合并；另一候选 #40709 已关闭但未合并。不应简单照抄未验证补丁，因为 Mamba checkpoint 与缓存正确性也需要回归。

## OpenViking 调用方式决定覆盖范围

- 若一个 VLM HTTP 请求含多张大图，本问题直接相关，应在隔离实例复测。
- 若 OpenViking 将图像拆成多次单图调用，上游这个“同请求多图互等”的精确条件不同，但并发、队列、超时重试及其他卡死路径仍需评估。
- “上传了多图/PDF”本身不等于“发给 vLLM 的同一个请求有多张图”。目前没有读取到实际出站 payload 或认证配置，不能替用户假定。

## 现有缓解措施的边界

`VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel` 针对 FP8 GEMM，不能修复 scheduler 零进展。关闭 FlashInfer sampler 也不能修复该多图问题。此前纯文本、temperature=0、输出 48 tokens 的 30 分钟通过结果，同样不能代表多图或随机采样业务通过。

后续隔离验证应覆盖：单次 1/2/4 张图、视觉 token 预算边界、短图间文本、重复前缀、单请求与并发请求、真实采样参数。若业务允许，可评估逐图摘要后文本聚合或限制每次调用的总视觉 token，代价是改变跨图推理与信息保留，不能自动套用。

观测应同时看任务年龄、排队、prefill/decode 进度、模型请求超时和完成计数。客户端重试应有上限，并避免把已卡住的请求继续堆到同一实例。长图预填充本就可能暂时不生成 token，因此不要只凭 generation 计数一段时间未变化就判死。

本轮仅做只读查询和 CPU 函数验证；未修改或重启两个生产服务，也未向生产提交可能触发卡死的图片请求。证据保存在本目录的 deployment-evidence.json、各 Pod startup.json、local-alignment-evidence.json。
