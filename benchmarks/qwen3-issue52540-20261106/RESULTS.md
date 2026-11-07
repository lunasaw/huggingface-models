# vLLM #52540 本机复现实验

结论：已在用户指定模型和 GPU 1 上复现 cuDNN FP8 GEMM 路径的 GPU 地址异常。禁用 FlashInfer FP8 线性内核的对照组完成 30 分钟发压，5645 个请求全部成功。所有测试进程已停止，GPU 1 已释放。

| 组别 | 负载 | 结果 |
|---|---|---|
| B1：默认内核 | 先 24 并发，再 16 并发；同一进程 | 24 并发阶段成功 2160 次；切换后 CUDA Graph replay 主机 SIGSEGV |
| B2：默认内核，全新进程 | 直接 16 并发 | 约 26 分 34 秒发生 cuDNN GPU 地址异常 / Xid 13；3232 次成功、16 次 HTTP 500 |
| A：禁用 FlashInfer FP8 内核 | 直接 16 并发，持续发压 30 分钟 | 5645 次成功、0 错误；结束后健康检查 200，无本轮 GPU 1 Xid |

这支持保留现网已有的 `VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel`。本次复现确认了故障路径和异常类型；没有逐项证明上游讨论的所有底层机制，也没有证明长期运行必然稳定。

## Issue 的含义

上游评论将问题定位到混合量化模型中 FP8 层所走的 FlashInfer/cuDNN GEMM 路径。模型名称虽然包含 NVFP4，但检查点中仍有 per-tensor FP8 层；这些层会触发另一套内核选择，因而不能根据模型名称判断是否受影响。

上游观察到共享工作区扩容后地址变化，使先前捕获的 CUDA Graph 保留旧地址；工作区中用于同步的区域未清零还会造成卡死。即使处理这两点，上游仍观察到 cuDNN 内部地址异常，怀疑与捕获和普通执行混用时的参数暂存有关，该具体机制尚有推断成分。配置绕过的作用是让 FP8 线性层改用 CUTLASS，避开整个 FlashInfer FP8 候选选择路径；NVFP4 层仍可使用原来的内核。

下文记录本机的独立观测；上游的内部机制分析不能自动视为本机已逐项证实。

## 环境与范围

用户授权使用当前机器的空闲 GPU 1。实验绑定 UUID `GPU-a697cda4-5468-8365-b094-805565a294a4`，RTX 5090 / SM120；独立 API 只监听 `127.0.0.1:28841`。GPU 2 上原有服务保持运行。生产 Kubernetes 服务未改配置、未重启、未做持续压测。

模型：`/data/ai/chenzhangyue/qwen38-27B/model/Qwen3.8-27B-NVFP4`。检查点为 ModelOpt MIXED，193 个 NVFP4 层、208 个 per-tensor FP8 层。

环境：`/data/miniconda3/envs/qwen38-27b-vllm/bin/python`，vLLM 0.26.0、PyTorch 2.11.0+cu130、FlashInfer 0.6.14、cuDNN 9.19.0.56、驱动 610.43.02。驱动不同于上游报告，不能视为完全相同环境。源文件与已安装 wheel RECORD 校验匹配，未修改共享依赖。

服务参数：TP=1，BF16，KV FP8 e4m3，Mamba BF16，max model len=32768，max num seqs=24，max batched tokens=8192，CUDA Graph FULL_AND_PIECEWISE，捕获尺寸 1/2/4/8/16/24，开启 chunked prefill 和 prefix caching，显存比例 0.90。这些是实验参数，与生产 max-num-seqs=4 的设置有区别。

负载：合成文本，实际输入约 2565–3563 tokens，输出固定 48 tokens，temperature=0、seed=52540、thinking 关闭。请求开头使用唯一标识减少 prefix cache 命中。客户端保存状态、用量、时延和前 50 条响应哈希；没有保存完整响应正文。

## 已确认的失败

首轮 B 组 `gpu1-B-stock` 未设置 `VLLM_DISABLED_KERNELS`。启动日志确认选择 `FlashInferFP8ScaledMMLinearKernel`。自动调优缓存的 63 个 FP8 项中，60 个为 `CudnnFp8GemmRunner`，3 个为 cuBLAS。M=1/2/4/8/16 分桶均选择 cuDNN，M=32 的三个形状均选择 cuBLAS；24 路 decode 映射到 M=32 桶。

24 并发阶段于 2026-11-06 14:45:24 +08:00 开始，主动停止客户端时累计运行 1070.229 秒，完成 2160 个请求、0 个请求错误。24 个在途请求被客户端取消。此次主动停止是为了改用覆盖 cuDNN decode 的 16 并发，不是服务故障。

服务排空后未重启，15:03:50 开始 16 并发阶段。随后引擎发生主机 SIGSEGV；15:04:16 API 检测到引擎退出，16 个请求全部 HTTP 500，客户端阶段总时长 30.021 秒。栈为：

```text
libcuda.so.1
cuGraphLaunch
cudaGraphLaunch
at::cuda::CUDAGraph::replay()
```

主机 core `gpu1-B-stock/core.927147` 及 gdb 回溯已保存。没有 `.nvcudmp` GPU core，没有观察到对应 Xid 或 OOM。仅凭主机 SIGSEGV 不能证明是上游捕获的 cuDNN GPU Warp Misaligned Address、工作区迁移或 semaphore 问题；目前是覆盖目标路径后观测到的 CUDA Graph 崩溃。

证据：`gpu1-B-stock/failure-evidence.json`、`failure-excerpt.log`、`host-core-backtrace.txt`、`host-core-registers.txt`、`decode-runner-buckets.json`、两阶段 `soak-summary.json`，以及完整 `server.log`。

## 全新进程复测：cuDNN GPU 地址异常

全新 B 组 `gpu1-B-repeat-c16`：重新建立进程、独立 vLLM/自动调优缓存；仅复用编译产物。直接运行 16 并发，避免上一轮 24→16 切换这一额外因素。自动调优仍为 60 个 cuDNN、3 个 cuBLAS；M=16 的三个选中 tactic 与第一轮一致，部分 prefill tactic 有变化。

负载于 15:12:43 +08:00 开始。15:39:18，GPU 1 的 PCI `0000:4c:00` 记录 Xid 13 / Out Of Range Address；距负载开始约 1594 秒（26 分 34 秒）。15:39:52 API 检测引擎退出。客户端总观察 1630.371 秒，成功 3232 个请求，16 个 HTTP 500，实际输入 2565–3563 tokens，输出共 155136 tokens（每次 48）。失败前没有人为停止或切换负载。

本轮生成 GPU core `gpu1-B-repeat-c16/cuda-core.1300322.nvcudmp`。离线 cuda-gdb 直接定位：

```text
CUDA Exception: Warp Out of range Address
cudnn_generated_fort_native_sm120_matMul_pointwise_pointwise_knob_10_128x256x64_0_0
@!P4 STG.E.128 desc[UR36][R24.64],R36
```

因此，本机已复现上游所讨论的 cuDNN GEMM 路径 GPU 地址异常及引擎退出，证据不再仅限于 HTTP 500 或主机段错误。尚未对工作区地址变化、补充 semaphore 区域和参数 staging 分别插桩，不能声称已经区分或独立证实上游的全部三个内部机制。

证据：该目录下 `failure-evidence.json`、`failure-excerpt.log`、`cuda-core-analysis.txt`、`cuda-fault-instruction.txt`、`kernel-gpu1-xid.log`、`soak-summary.json` 与原始 GPU / host core。

两次 B 组的前 50 条响应哈希仅 12/50 一致。这些是自由文本生成且批次形状不同，不是数值一致性的 golden 测试，不能据此推断输出错误，也不能声明精确输出一致。

## 对照组结果

A 组 `gpu1-A-cutlass`：相同模型、服务参数、请求生成器，只设置 `VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel`；启动日志已确认回退至 `CutlassFP8ScaledMMLinearKernel`。由于全新 B 进程直接 16 并发也已捕获明确 GPU 故障，对照组采用相同的直接 16 并发。

A 组独立自动调优缓存有 42 个 FP4 项、0 个 FP8 项；结合启动日志可确认绕过配置已实际生效。证据见 `gpu1-A-cutlass/kernel-selection-evidence.json`。

15:47:48 开始发压，持续提交请求 1800 秒，最后一个请求于 1801.768 秒完成；客户端在 1810.374 秒记录最终结果。共 5645 个请求全部成功、0 个错误、0 个取消、0 个在途请求，输入 17368974 tokens、输出 270960 tokens。中位时延 5.094 秒，P95 5.3435 秒。结束后健康检查返回 200；从本组启动到最终检查的 GPU 1 内核日志中没有 Xid，也没有 GPU/host core 或目标 CUDA 错误日志。

证据：`gpu1-A-cutlass/success-evidence.json`、`soak-summary.json`、`kernel-window.log`、`server.log`。这是一轮 30 分钟的有限观察，不能外推为长期无故障保证。

## 配置与性能可比性

`ab-config-comparison.json` 记录两组服务命令完全一致。环境差异为禁用项和各自独立的缓存、core 输出目录；两组复用相同 FlashInfer 编译产物，但未共用 FP8 自动调优缓存。

共同完成的前 3000 个请求，逐条输入和输出 token 数相同，累计输入 9232557、输出 144000 tokens。A 组完成该集合用时约 958.736 秒，B 组约 1483.996 秒；中位时延分别 5.0868 和 7.8924 秒。这说明绕过配置在本轮合成负载上也更快；它不是对所有生产负载的性能保证。详见 `ab-request-comparison.json`。这些统计不验证生成内容的正确性。

## 清理与证据保存

所有实验的 API、压测和 GPU 采样进程已退出；127.0.0.1:28841 已关闭。GPU 1 显存恢复为 2 MiB、无计算进程，GPU 2 原有 API 与引擎进程仍在运行。未进行 GPU reset。核实记录见 `cleanup.json`。

两轮故障的 host core 和 B2 的 GPU core 原件已保留；文件大小与 SHA-256 见 `core-artifacts.json`。离线故障内核与指令的精简证据见 `gpu1-B-repeat-c16/cuda-core-summary.txt`。

## 脚本与复跑

`launch_local.py` 启动独立测试进程；`start_soak.py` 启动有时限的客户端和 GPU 采样；`inspect_local.py` 检查健康、进度、错误和 runner 缓存；`stop_local.py` 校验进程身份后只停止本实验进程。默认 GPU UUID、模型和 Python 环境已固定为本次授权范围。每次启动需使用一个新的 run name。

```bash
python launch_local.py B-flashinfer --run-name another-B
python inspect_local.py another-B
# 等待 health=200 后：
python start_soak.py another-B --concurrency 16 --duration 1800
python inspect_local.py another-B
python stop_local.py another-B

python launch_local.py A-cutlass --run-name another-A
# 等待 health=200 后同样运行 start_soak.py，完成后 stop_local.py。
```

判定时需要区分“覆盖了路径”“观测到服务崩溃”“证实与上游同一底层根因”。本次已有 cuDNN GPU core 直接证据，但尚无各项内部机制的独立插桩验证。

上游：[根因与绕过配置](https://github.com/vllm-project/vllm/issues/52540#issuecomment-5310659901)。生产入口的前置核实见 `README.md` 和 `preflight.json`。
