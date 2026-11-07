# vLLM #52540 复现前置检查

后续本机 GPU 1 实验已完成：默认内核捕获 cuDNN GPU 地址异常，禁用后的对照组 30 分钟、5645 次请求无错误。详见 [完整实验结果](RESULTS.md)。下文保留实验前的生产检查与初始设计，实际执行参数以 RESULTS.md 为准。

采集时间：2026-11-06T14:10:34.806789+08:00（Asia/Shanghai）。

## 结论

模型和硬件属于值得验证的组合，但当前生产配置已绕过目标故障路径。全部 16 个 Pod 都设置了 VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel，启动日志均确认选择 CutlassFP8ScaledMMLinearKernel，且 CUDA Graph 已捕获。未进行持续压测，也未修改、重启任何生产实例。

## 已核实配置

- Service：prod/service-manager-vllm-qwen3，ClusterIP 10.68.102.113:8081。当前机器不能解析短域名，已使用查到的 ClusterIP 访问。
- 模型：qwen3.8-27b-nvfp4；路径 /models/Qwen3.8-27B-NVFP4；quantization=modelopt_mixed。
- vLLM：0.26.0；FlashInfer 启动日志版本：0.6.14。cuDNN 精确版本尚未读取。
- GPU：GPU inventory 查询确认 16 次分配、按节点/UUID 去重 16 张 RTX 5090，16/16 Pod Ready；启动日志确认 sm120。设备配额并非实时利用率。
- TP=1；max-num-seqs=4；max-num-batched-tokens=8192；CUDA Graph 捕获尺寸 1、2、4。
- max-model-len=184320；KV cache dtype=fp8_e4m3；prefix caching 与 chunked prefill 开启。
- Pod 0 启动日志：FP8 线性层使用 CutlassFP8ScaledMMLinearKernel；NVFP4 层使用 FlashInferCutlassNvFp4LinearKernel；FULL_AND_PIECEWISE；KV cache 5.89 GiB / 187430 tokens。
- 所有副本当前重启次数为 0；这不能证明过去其他已删除实例没有发生过故障。

## 已执行的接口验证

仅发送一次合成短请求，要求回复 OK。HTTP 200，耗时 0.347 秒，返回 OK。该请求验证接口当前可用，不构成故障复现或长期稳定性结论。

## 可执行的对照实验设计

1. 使用允许 CUDA 故障和重启的独立 SM120 测试实例，模型与镜像对齐现网。测试实例使用独立标签，不加入生产 Service 的 selector；测试直接访问该实例。GPU 和测试实例由用户指定。
2. A 组保留现有禁用配置，B 组仅在独立实例中移除 FlashInferFP8ScaledMMLinearKernel 的禁用项。先保持其他参数相同，包括 max-num-seqs=4 和捕获尺寸 1、2、4。每组使用独立进程，记录实际依赖版本。
3. B 组启动后必须核对 FlashInferFP8ScaledMMLinearKernel 选择，并检查自动调优缓存中是否实际选用 CudnnFp8GemmRunner。仅选用 FlashInfer 外层内核，不足以证明进入目标 cuDNN 路径；若未进入，记录为未覆盖目标路径。
4. 两组使用相同合成请求集，输入长度约 2500–3500 tokens，输出上限 48 tokens，关闭 thinking，并改变输入开头以避免大量 prefix cache 命中。先做低并发连通性检查，再以并发 4 持续运行 60 分钟；记录实际 prompt/completion token 数。
5. 如需接近原报告的并发 24 条件，先评估独立 32 GB 显卡显存，再同时调整两组 max-num-seqs 和 CUDA Graph 捕获尺寸后重新做对照。对当前 max-num-seqs=4 的实例发送 24 个请求，不能当作原报告中同一引擎 24 条并行序列的等价复现。
6. 同步记录每个请求的状态、时延、成功计数、超时，以及引擎日志、GPU 状态、Xid。首次 CUDA illegal/misaligned 错误或明确失去进展后停止投递。超时本身只能作为待排查异常，不能直接断定复现了本 issue。
7. 要归因到此 issue，需保留故障内核或工作区证据，并验证 A 组在相同条件下的表现。固定时长未失败，只能写“本轮未复现”。

重新启用内核需要改变测试服务的启动环境并重启进程，不能通过 OpenAI 请求参数完成。生产入口连接 16 个副本，负载会分散，也会掩盖单实例故障，因此不作为故障复现入口。

## 后端清单

| Pod | Phase / Ready | Pod IP | 宿主机 IP | 节点 | FP8 CUTLASS 已确认 |
|---|---|---|---|---|---|
| service-manager-vllm-qwen3-0 | Running / 1/1 | 10.69.45.162 | 10.26.96.156 | idc-bj08-5090-59 | True |
| service-manager-vllm-qwen3-1 | Running / 1/1 | 10.69.36.103 | 10.26.80.174 | idc-bj08-5090-21 | True |
| service-manager-vllm-qwen3-2 | Running / 1/1 | 10.69.51.135 | 10.26.48.160 | idc-bj08-5090-82 | True |
| service-manager-vllm-qwen3-3 | Running / 1/1 | 10.69.40.207 | 10.26.48.140 | idc-bj08-5090-39 | True |
| service-manager-vllm-qwen3-4 | Running / 1/1 | 10.69.50.26 | 10.26.64.151 | idc-bj08-5090-76 | True |
| service-manager-vllm-qwen3-5 | Running / 1/1 | 10.69.51.73 | 10.26.48.161 | idc-bj08-5090-81 | True |
| service-manager-vllm-qwen3-6 | Running / 1/1 | 10.69.42.68 | 10.26.48.150 | idc-bj08-5090-45 | True |
| service-manager-vllm-qwen3-7 | Running / 1/1 | 10.69.48.35 | 10.26.64.149 | idc-bj08-5090-66 | True |
| service-manager-vllm-qwen3-8 | Running / 1/1 | 10.69.38.144 | 10.26.96.140 | idc-bj08-5090-30 | True |
| service-manager-vllm-qwen3-9 | Running / 1/1 | 10.69.45.78 | 10.26.96.154 | idc-bj08-5090-57 | True |
| service-manager-vllm-qwen3-10 | Running / 1/1 | 10.69.45.35 | 10.26.96.151 | idc-bj08-5090-56 | True |
| service-manager-vllm-qwen3-11 | Running / 1/1 | 10.69.45.31 | 10.26.96.151 | idc-bj08-5090-56 | True |
| service-manager-vllm-qwen3-12 | Running / 1/1 | 10.69.45.57 | 10.26.96.151 | idc-bj08-5090-56 | True |
| service-manager-vllm-qwen3-13 | Running / 1/1 | 10.69.45.45 | 10.26.96.151 | idc-bj08-5090-56 | True |
| service-manager-vllm-qwen3-14 | Running / 1/1 | 10.69.45.59 | 10.26.96.151 | idc-bj08-5090-56 | True |
| service-manager-vllm-qwen3-15 | Running / 1/1 | 10.69.35.227 | 10.26.80.169 | idc-bj08-5090-19 | True |

Service：ClusterIP；8081 → 8081/TCP；selector app=service-manager-vllm-qwen3，选中上表 16 个 Pod。Ready 状态与 selector 关联不能替代 EndpointSlice 的实时就绪确认。

原始配置与逐副本筛选日志见同目录 preflight.json。

参考：[原始根因评论](https://github.com/vllm-project/vllm/issues/52540#issuecomment-5310659901)。
