# HauhauCS Qwen vLLM 历史部署概念说明

> 本文记录 2026-10-24 的 AWQ 部署和概念，以下“当前”均指当时的配置。服务现已切换为官方 BF16；端口、GPU、上下文长度以[根目录 README](../README.md)和正式启动脚本为准。Hauhau 目录已恢复独立 AWQ-MTP 启动入口，配置见该项目 README。文中基准结果和日志仅在本机保留。

本文说明当前这套服务中出现的模型格式、量化、推理、GPU 并行、缓存、MTP 和性能指标。内容以本机实际启动脚本、模型目录配置和基准结果为准。

相关文件：

- [Hauhau 启动脚本](../qwen38-27b-Uncensored-HauhauCS/start-hauhau-vllm.sh)
- 并发 1 基准：`results/hauhau-vllm/random-1024in-256out-c1.json`（本机文件）
- 并发 4 基准：`results/hauhau-vllm/random-1024in-256out-c4.json`（本机文件）
- vLLM 日志：`logs/hauhau-vllm-20261024-1908.log`（本机文件）

---

## 1. 先看整体：一次请求是怎样跑完的

当前服务可以抽象成下面这条链路：

```text
客户端
  │ HTTP / OpenAI-compatible API
  ▼
127.0.0.1:28901
  │
  ▼
vLLM API Server
  │ 1. tokenizer：文字 → token ID
  │ 2. chat template：messages → 模型输入格式
  │ 3. prefill：一次处理完整提示词
  │ 4. decode：逐步生成输出 token
  │ 5. MTP：先提出最多 3 个候选 token，再由主模型验证
  ▼
两张 GPU（物理 GPU 5、6，TP=2）
  │
  ├─ 4-bit 压缩权重（compressed-tensors / AWQ 系列格式）
  ├─ BF16 计算
  ├─ FP8 KV cache
  └─ CUDA / Triton / FlashInfer 等 GPU kernel
  ▼
生成 token → tokenizer 解码 → HTTP 响应
```

这几个词描述的是不同层次的问题：

| 层次 | 主要问题 | 当前使用的概念 |
|---|---|---|
| 模型内容 | 模型学到了什么、参数有多少 | Qwen、HauhauCS、27B、Aggressive |
| 文件格式 | 权重怎样存放在磁盘上 | GGUF、Safetensors |
| 权重量化 | 权重用多少位保存和计算 | AWQ、INT4、group size 128 |
| 推理引擎 | 怎样高效地运行模型和提供接口 | vLLM |
| GPU 并行 | 一份模型如何放到多张卡 | Tensor Parallelism，TP=2 |
| 运行时精度 | 普通计算和缓存用什么数值类型 | BF16、FP8 KV cache |
| 解码加速 | 怎样减少逐 token 生成的开销 | MTP / speculative decoding |
| 服务协议 | 客户端怎样调用 | OpenAI-compatible API |
| 性能测试 | 怎样描述速度和延迟 | TTFT、TPOT、P50、P95、吞吐 |

---

## 2. 模型、参数和 token

### 2.1 LLM、模型权重和参数

LLM 是 Large Language Model（大语言模型）。它本质上是一组经过训练的数值参数，也叫模型权重。收到文字后，模型通过矩阵运算计算“下一个 token 各自有多大概率”，再根据采样参数选出下一个 token，重复这个过程。

**27B** 中的 B 是 billion，表示约 270 亿个参数。27B 是模型规模，不等于需要 27 GB 显存，也不等于输出质量一定更好。显存占用还取决于：

- 权重是否量化；
- 激活值使用什么精度；
- KV cache 需要保存多少上下文；
- 是否使用多模态模块、MTP 模块和 CUDA Graph；
- 是否把模型拆到多张 GPU 上。

当前目录名是 `HauhauCS-Qwen3.8-27B-Aggressive-AWQ-MTP`，服务别名是 `qwen3.8-27b-hauhau-aggressive`。这些是发布者或部署者使用的名称。模型的 `config.json` 中，vLLM 实际解析到的架构是 `qwen3_5` / `Qwen3_5ForConditionalGeneration`，因此：

- `3.8` 是当前目录和 API 使用的模型名称的一部分；
- `qwen3_5` 是配置文件声明的实现架构；
- 运行时以配置文件和 vLLM 注册的架构为准，不能只根据目录名推断内部实现。

### 2.2 Token 是什么

模型不直接处理“字”“词”或字符串，而是处理 token。Tokenizer 把文字切成 token，再把 token 映射为整数 ID。

例如，英文单词可能是一个 token，也可能被拆成多个 token；中文一个字可能是一个 token，也可能和相邻字符合并或拆分。因而“1024 tokens”不是“1024 个汉字”，而是 tokenizer 产生的 1024 个离散单元。

一次请求中有三类长度：

| 名称 | 含义 |
|---|---|
| 输入 token / prompt tokens | 用户消息和模板展开后的 token 数量 |
| 输出 token / completion tokens | 模型新生成的 token 数量 |
| 总上下文长度 | 输入 token、输出 token 以及必要控制 token 的总长度 |

当前服务用 `--max-model-len 32768` 把单个请求允许的总上下文上限设为 32768。它不是“最多输出 32768 token”。例如输入已经占用 30000 token 时，输出空间还会受到剩余上下文长度限制。

模型配置原本声明的 `max_position_embeddings` 是 262144，但本次服务主动限制到 32768。这样做通常是为了控制 KV cache 占用、启动时间和稳定性。

### 2.3 Chat template

聊天接口收到的是类似下面的结构：

```json
{
  "messages": [
    {"role": "user", "content": "你好"}
  ]
}
```

模型不能直接理解这个 JSON。Chat template 会把角色、消息、特殊 token 等内容拼成该模型训练时使用的文本格式，再交给 tokenizer。不同模型的模板可能不同，所以不能随意把一个模型的模板复制给另一个模型。

本服务通过：

```text
--default-chat-template-kwargs '{"enable_thinking":false,"reasoning_effort":"medium"}'
```

给模板传默认参数。`enable_thinking=false` 表示关闭模板层面的思考模式；它不是删除模型能力，而是要求该次聊天不要按思考模式组织输出。

---

## 3. GGUF、Safetensors、AWQ 和 compressed-tensors

这四个词经常一起出现，但它们不是同一种东西。

### 3.1 GGUF：模型文件容器和运行时生态

GGUF 是一种用于保存模型张量和元数据的文件格式，常见于 llama.cpp / GGML 生态。它可以把以下内容放在一个或多个文件中：

- 模型权重；
- 张量名称和形状；
- tokenizer 信息；
- 上下文长度等配置；
- 量化类型和其他元数据。

GGUF 是**文件格式**，不是某一种固定的量化算法。一个 GGUF 文件可能使用不同的量化类型，也可以保存未量化或不同精度的张量。

本次部署的原始 HauhauCS 仓库是 GGUF 格式，而当前使用的 vLLM 版本不能直接加载这个通用 GGUF 仓库。因此没有让 vLLM 直接读取原始 GGUF，而是使用了一个转换后的 vLLM/AWQ 适配版本。

常见误区：

| 说法 | 是否准确 | 说明 |
|---|---|---|
| “GGUF 就是 4-bit” | 不准确 | GGUF 是容器格式，可以承载多种精度和量化类型 |
| “只要把文件后缀改成 safetensors 就能加载” | 不准确 | 文件内部张量布局、配置和运行时支持必须匹配 |
| “vLLM 一定能加载所有 Hugging Face 模型” | 不准确 | 需要同时支持架构、权重格式、量化方法和当前版本的实现 |

### 3.2 Safetensors：安全的张量存储格式

当前适配版目录中有：

- `model.safetensors`，约 18 GB；
- `model-mtp.safetensors`，约 811 MB。

Safetensors 主要解决“如何把张量安全、快速地存到磁盘并加载”的问题。它保存张量的二进制数据和索引元数据，不依赖执行任意 Python 代码来恢复对象，因此比某些可执行反序列化格式更适合发布模型权重。

Safetensors 也只是**存储格式**。看到 `.safetensors` 不能单独判断模型是 FP16、BF16、INT8 还是 INT4；必须结合 `config.json` 和量化元数据判断。

### 3.3 AWQ：Activation-aware Weight Quantization

AWQ 的全称通常写作 Activation-aware Weight Quantization，中文可译为“感知激活值的权重量化”。它的核心目标是：

1. 观察校准数据经过模型时哪些通道或权重更重要；
2. 对大多数权重进行低比特量化；
3. 对重要部分采取保护、缩放或更合适的量化处理；
4. 在显著减少显存和带宽的同时，尽量保留原模型质量。

当前模型名称中带有 `AWQ`，表示它属于 AWQ 量化适配路线。它的 `config.json` 则声明了 vLLM 实际使用的量化后端：

```json
{
  "quant_method": "compressed-tensors",
  "format": "pack-quantized",
  "weights": {
    "num_bits": 4,
    "group_size": 128,
    "strategy": "group"
  }
}
```

启动脚本也使用了：

```text
--quantization compressed-tensors
```

因此，本次服务不是通过 `--quantization awq` 这个参数启动的。更准确的表述是：**模型发布名称和权重量化来源属于 AWQ 系列，而当前文件和 vLLM 加载器使用 compressed-tensors 的打包格式。** 量化算法、磁盘格式和 vLLM 后端是三个相关但不同的概念。

### 3.4 INT4、group size 128、scale 和 zero point

当前量化权重的主要信息可以这样读：

| 配置 | 含义 |
|---|---|
| `num_bits: 4` | 量化权重主要使用 4 bit 表示 |
| `group_size: 128` | 通常每 128 个权重共享一组量化参数 |
| `strategy: group` | 使用分组量化，而不是整层只使用一组参数 |
| `symmetric: false` | 允许非对称量化，通常需要 zero point |
| `zp_dtype: torch.int8` | zero point 使用 int8 存储 |
| `format: pack-quantized` | 低比特值以打包形式存储，以节省空间和提高 kernel 读取效率 |

量化可以粗略表示为：

```text
量化值 q ≈ round(weight / scale) + zero_point
恢复值 weight ≈ (q - zero_point) × scale
```

实际实现会按照后端 kernel 的格式进行更复杂的打包、缩放和矩阵乘法。4 bit 不是说每次运算都只能使用 4 bit；常见做法是权重以 4 bit 保存和读取，乘法时转成 BF16、FP16 或其他适合 GPU 的格式计算。

### 3.5 不是所有模块都一定是 INT4

量化配置的 `ignore` 列表包含视觉模块、部分线性注意力 / Mamba 模块和 MTP 相关模块。这表示部分模块被排除在相同的权重量化规则之外，可能保留 BF16、FP16、FP32 或使用专门的表示。

因此，下面这句话不准确：

> “这是一个所有张量都是 4-bit 的模型。”

更准确的说法是：

> “这是以 4-bit 分组权重量化为主、部分模块保持其他精度，并由 compressed-tensors 加载的模型。”

### 3.6 四种精度不要混为一谈

当前配置同时涉及多种数值表示：

| 部位 | 当前主要设置 | 作用 |
|---|---|---|
| 量化后的线性层权重 | INT4 打包 | 减少模型权重显存和读取带宽 |
| 未被量化的权重/普通激活计算 | BF16 | 在速度、范围和精度之间折中 |
| 注意力 KV cache | FP8，日志中为 `float8_e4m3fn` | 减少长上下文缓存占用 |
| 某些模型状态或算子 | 由模型配置和 kernel 决定，部分可能使用 FP32 | 保持数值稳定性 |

所以 `--dtype bfloat16` 并不会把已经量化的 INT4 权重“解量化回 BF16”，它主要指定普通计算和未量化部分使用的计算类型。

---

## 4. vLLM、prefill、decode 和 KV cache

### 4.1 vLLM 是什么

vLLM 是一个大模型推理引擎和 HTTP 服务框架。它负责：

- 加载模型和 tokenizer；
- 把模型映射到 GPU；
- 管理请求队列；
- 将多个请求动态合并执行；
- 管理 KV cache；
- 调用高效的 CUDA、Triton、FlashAttention / FlashInfer 等 kernel；
- 提供 OpenAI-compatible API。

它不是一个新的模型，也不会自动改变模型的训练知识。它主要改变“模型怎样被高效地运行”。

### 4.2 Prefill：处理输入

Prefill 是第一次处理 prompt 的阶段。例如输入有 1024 token，模型会集中处理这 1024 个 token，并建立后续生成需要的中间状态。

Prefill 通常更接近大矩阵批量计算，受输入长度和批处理大小影响明显。TTFT（Time To First Token，首 token 时间）包括排队、调度、输入处理以及产生首个输出 token 的时间，因此输入长或并发高时通常会变大。

### 4.3 Decode：逐 token 生成

Decode 是输出阶段。每次生成一个或一组候选 token，追加到序列，再生成下一步。普通自回归解码具有串行依赖：第 N 个 token 生成后，才能知道第 N+1 个 token。

TPOT（Time Per Output Token）表示首 token 之后，连续输出 token 之间的平均时间。TPOT 越小，逐字流式输出看起来越快。

### 4.4 KV cache 是什么

Transformer 注意力每生成新 token，都需要用到之前 token 的 Key 和 Value。如果每次都从头重新计算，生成会很慢。KV cache 把已经计算过的 Key / Value 保存起来，后续只计算新 token。

KV cache 不是模型权重，也不是对话数据库。它通常只在请求存在期间保存某个序列的中间状态；请求结束后，相关空间可以复用。

它的占用量大致随以下因素增加：

- 上下文 token 数；
- 并发请求数；
- 模型层数和注意力维度；
- cache 使用的数值类型；
- 模型是全注意力、线性注意力还是混合架构。

当前日志显示：

```text
GPU KV cache size: 593,464 tokens
Maximum concurrency for 32,768 tokens per request: 18.11x
```

这里的 `18.11x` 是根据可用 KV cache 和 32768 的单请求长度计算出的理论容量指标，不表示服务会同时接受 18 个请求。启动脚本的 `--max-num-seqs 4` 会把调度中的序列数限制在 4，因此实际并发上限还受它限制。

### 4.5 FP8 KV cache

启动脚本使用：

```text
--kv-cache-dtype fp8
```

日志显示实际使用的是 `float8_e4m3fn`。FP8 是 8-bit 浮点格式，内存比 BF16 / FP16 小一半，适合保存大量 KV cache。它不是把整个模型都变成 FP8：

- 权重仍以量化权重和其他精度混合存在；
- 普通计算仍主要按 BF16 配置运行；
- 只有 KV cache 采用 FP8 存储路径。

优点是可以容纳更长上下文或更多并发；代价是缓存数值精度降低，实际效果依赖 scale、硬件和模型。对本次 1024 输入、256 输出的基准来说，KV cache 压力并不大；在更长上下文或更多并发时它更有价值。

### 4.6 Prefix caching

启动脚本使用：

```text
--enable-prefix-caching
```

如果多个请求的开头 token 完全相同，vLLM 可以复用这段前缀的计算结果和缓存，减少重复 prefill。例如多个请求都以同一份系统提示词、长文档或工具定义开头时很有帮助。

它要求前缀在 token 层面匹配，不是“文字看起来相似”就能命中。当前基准使用随机输入，且每条请求的输入基本不同，所以日志中的 prefix cache hit rate 为 0%，这是预期现象。

### 4.7 Chunked prefill

启动脚本使用：

```text
--enable-chunked-prefill
```

Chunked prefill 会把很长的输入分成若干块处理，让长 prompt 的计算可以和其他请求的 decode 交错调度。这样通常能改善多用户场景下的响应性和长尾，但会引入调度复杂度，具体收益依赖 prompt 长度、并发和 token budget。

---

## 5. MTP 和 speculative decoding

### 5.1 普通解码为什么慢

普通自回归生成每一步通常只由主模型产生一个 token：

```text
主模型 → token 1
主模型 → token 2
主模型 → token 3
```

每个 token 都要经过完整或大部分模型计算，虽然 GPU 计算很快，但每一步都有调度、内存访问和模型执行开销。

### 5.2 Speculative decoding 的基本思路

Speculative decoding（推测解码）让一个较快的 draft / proposal 模块先提出候选 token，再让主模型一次检查这些候选：

```text
草稿模块提出：token A、B、C
主模型验证：A ✓、B ✓、C ✗
最终接受：A、B，然后由主模型决定后续 token
```

如果候选通常正确，就能在一次主模型调度中前进多个 token，从而降低有效的每 token 成本。如果候选经常错误，验证开销可能抵消收益，甚至变慢。

### 5.3 MTP 是什么

MTP 是 Multi-Token Prediction（多 token 预测）。当前模型目录中额外有 `model-mtp.safetensors`，启动脚本指定：

```json
{"method":"mtp","num_speculative_tokens":3}
```

这表示每轮最多提出 3 个 speculative token。它不是“每次无条件输出 3 个 token”，而是“尝试提出 3 个候选，主模型验证后只接受通过的部分”。最终生成仍遵守主模型、采样参数和 EOS 规则。

### 5.4 MTP 接受率怎样理解

基准中的 MTP 接受率约为：

- 并发 1：53.96%；
- 并发 4：54.72%。

它表示草稿模块提出的候选 token 中，有多大比例被主模型验证接受。它不是：

- 模型回答正确的概率；
- 用户满意率；
- 生成质量分数；
- “每 3 个 token 一定接受 3 个”的保证。

不同工具可能按候选位置、草稿 token 总数或平均接受长度统计，因此比较不同基准时要确认统计口径。日志还记录了 `Mean acceptance length` 和每个候选位置的 acceptance rate，它们与汇总接受率不是完全相同的指标。

### 5.5 MTP 的实际取舍

| 情况 | 可能结果 |
|---|---|
| 主模型和 MTP 预测高度一致 | 有效生成速度提升 |
| 候选经常被拒绝 | 加入验证开销，收益较小或变差 |
| `num_speculative_tokens` 较大 | 潜在吞吐更高，但验证计算、显存和调度压力也更大 |
| 并发高、显存紧张 | speculative slots 会占用额外资源 |

日志中有提醒：设置多个 speculative token 会多次执行 MTP 层，可能降低接受率；同时 scheduler 的 `max_num_scheduled_tokens` 受到 speculative 设置影响。这是性能调优时需要关注的地方。

---

## 6. Tensor Parallelism：为什么两张 GPU 能跑一份模型

### 6.1 TP=2 的含义

启动脚本使用：

```text
--tensor-parallel-size 2
```

Tensor Parallelism（张量并行）会把同一层的矩阵或张量切分到两张 GPU 上。一次请求通常会同时使用两张卡，双方在层间通过 GPU 通信交换部分结果。

它不是：

- 启动两个互相独立的模型副本；
- 把请求简单地一半发给 GPU 5、一半发给 GPU 6；
- 只在显存不够时才有用。

它的主要作用是让一份较大的模型分布到多张 GPU，同时利用两张卡的计算能力。代价是每一步都会产生跨 GPU 通信，通信路径和带宽会影响吞吐及延迟。

### 6.2 CUDA_VISIBLE_DEVICES 和物理 GPU 编号

脚本中有：

```text
export CUDA_VISIBLE_DEVICES=5,6
```

这表示进程只看得到物理 GPU 5 和 6。对这个进程来说，它们通常会重新编号为逻辑 GPU 0 和 1：

```text
进程逻辑编号 0 → 物理 GPU 5
进程逻辑编号 1 → 物理 GPU 6
```

因此 vLLM 日志里出现 rank 0、rank 1，并不一定直接显示物理编号 5、6。排查时要结合 `CUDA_VISIBLE_DEVICES` 和 `nvidia-smi` 看映射关系。

### 6.3 NCCL、P2P 和 IB

NCCL 是 NVIDIA 用于 GPU 集合通信的库。当前脚本设置：

```text
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
```

含义是：

| 环境变量 | 作用 |
|---|---|
| `NCCL_P2P_DISABLE=1` | 禁止 NCCL 使用 GPU 间直接 P2P 通道 |
| `NCCL_IB_DISABLE=1` | 禁止使用 InfiniBand 通道 |

这样做常用于绕开特定机器拓扑、驱动或通信稳定性问题。代价可能是 GPU 间通信速度下降。它们不是模型参数，也不会改变模型答案；它们改变的是多 GPU 通信路径。

脚本还使用：

```text
--disable-custom-all-reduce
```

这会关闭 vLLM 的自定义 all-reduce 实现，使用更通用的通信路径。优点通常是兼容性和稳定性更可控，代价可能是部分场景的通信性能降低。

### 6.4 TP 和数据并行的区别

| 对比 | Tensor Parallelism | Data Parallelism |
|---|---|---|
| 模型权重 | 多卡共同保存一份模型 | 每张卡各保存一份模型副本 |
| 单请求 | 通常需要多卡协同 | 通常由一张卡处理 |
| 主要目标 | 让大模型放得下、加速单模型计算 | 提高独立请求吞吐 |
| 当前配置 | TP=2 | 没有配置多个数据并行副本 |

---

## 7. BF16、FP16、FP8、INT4 怎么区分

### 7.1 BF16

BF16（bfloat16）是 16-bit 浮点格式。它与 FP16 都是 16 bit，但指数位布局不同，BF16 的数值范围接近 FP32，通常更容易避免溢出，代价是有效精度位较少。

当前脚本：

```text
--dtype bfloat16
```

这表示 vLLM 对支持该设置的普通模型计算使用 BF16。它不是量化开关。

### 7.2 FP16

FP16 也是 16-bit 浮点格式，有更高的尾数精度但更小的指数范围。它在很多 GPU 推理中很常见。BF16 和 FP16 不能只看“都是 16 bit”就认为完全等价。

### 7.3 FP8

FP8 是 8-bit 浮点格式。它比 BF16 / FP16 占用更少内存，但可表示范围和精度都更有限，通常需要 scale 或专门 kernel。当前服务将 FP8 主要用于 KV cache，而不是把整个模型权重都改成 FP8。

### 7.4 INT4

INT4 是 4-bit 整数表示。它常用于权重量化；相比 16-bit 浮点，原始存储位数理论上只有四分之一，但还要加上 scale、zero point、对齐和未量化模块的额外开销。

### 7.5 一张表总结

| 类型 | 位数 | 常见用途 | 当前部署中的位置 |
|---|---:|---|---|
| FP32 | 32 | 高精度计算、部分状态 | 部分模型状态或稳定性路径可能使用 |
| BF16 | 16 | 普通推理计算、未量化张量 | `--dtype bfloat16` |
| FP16 | 16 | 常见推理计算 | 当前主配置不是 FP16 |
| FP8 | 8 | KV cache 或特定计算 | `--kv-cache-dtype fp8` |
| INT8 | 8 | 量化参数或部分权重 | 量化元数据中的 zero point 使用 int8 |
| INT4 | 4 | 权重量化 | compressed-tensors 的主权重格式 |

---

## 8. 启动脚本逐项解释

当前脚本的核心内容是：

```bash
export CUDA_VISIBLE_DEVICES=5,6
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

vllm serve \
  /data/ai/chenzhangyue/huggingface-models/models/HauhauCS-Qwen3.8-27B-Aggressive-AWQ-MTP \
  --served-model-name qwen3.8-27b-hauhau-aggressive \
  --host 127.0.0.1 \
  --port 28901 \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --quantization compressed-tensors \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 4096 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --disable-custom-all-reduce \
  --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

| 参数 | 作用 | 当前值的实际含义 |
|---|---|---|
| `vllm serve PATH` | 启动 vLLM 服务并加载模型 | 从本地适配版目录加载 |
| `--served-model-name` | API 中可使用的模型名 | 请求中的 `model` 应写 `qwen3.8-27b-hauhau-aggressive` |
| `--host 127.0.0.1` | 监听地址 | 只接受本机连接，局域网其他机器不能直接访问 |
| `--port 28901` | HTTP 端口 | 服务地址为 `http://127.0.0.1:28901` |
| `--tensor-parallel-size 2` | 张量并行卡数 | 两张 GPU 协同运行一份模型 |
| `--dtype bfloat16` | 普通计算精度 | 使用 BF16 作为主要计算 dtype |
| `--quantization compressed-tensors` | 指定量化加载后端 | 按模型配置加载打包的低比特权重 |
| `--max-model-len 32768` | 单请求总上下文上限 | 输入和输出总长度最多按 32768 规划 |
| `--gpu-memory-utilization 0.90` | vLLM 目标显存占比 | 用于权重、激活、CUDA Graph 和 KV cache 的预算 |
| `--max-num-seqs 4` | 同时调度的序列数上限 | 并发 4 基准与此设置相符 |
| `--max-num-batched-tokens 4096` | 一次调度的 token 预算 | 影响长 prompt、批处理和调度延迟 |
| `--kv-cache-dtype fp8` | KV cache 精度 | 缓存主要使用 FP8 |
| `--enable-prefix-caching` | 开启公共前缀缓存 | 相同 token 前缀可以复用 prefill 结果 |
| `--enable-chunked-prefill` | 开启分块 prefill | 长 prompt 可拆块并与 decode 交错 |
| `--disable-custom-all-reduce` | 关闭 vLLM 自定义 all-reduce | 使用更通用的多 GPU 通信路径 |
| `--compilation-config` | 设置编译 / CUDA Graph 行为 | 使用 PIECEWISE 模式减少重复 kernel launch 开销 |
| `--speculative-config` | 开启 speculative decoding | 使用 MTP，每轮最多提出 3 个候选 token |
| `VLLM_WORKER_MULTIPROC_METHOD=spawn` | worker 进程启动方式 | 用 spawn 启动多进程 worker，减少 CUDA fork 问题 |

### 8.1 gpu-memory-utilization 不是“把显存固定用到 90%”

`0.90` 是 vLLM 用来规划显存的目标比例，不是精确的实时占用锁定值。显存还会受 CUDA Graph、编译缓存、kernel workspace、驱动和其他进程影响。

本次日志显示 CUDA Graph 估算后，有效比例略低于直接的 0.90，并报告了约 16.94 GiB 的 KV cache。看到 GPU 没有恰好占到 90% 不代表参数没有生效。

### 8.2 max-num-batched-tokens 和 max-num-seqs 的区别

它们分别限制不同维度：

- `max-num-seqs=4`：最多同时调度多少条序列；
- `max-num-batched-tokens=4096`：一次调度中最多处理多少 token 预算。

例如 4 条请求各有很长 prompt 时，序列数没有超过 4，但 token 预算仍可能先成为瓶颈。反过来，短 prompt 可以在 token 预算允许时批量处理多条序列，但仍受序列数上限限制。

日志提示 speculative decoding 会影响 `max_num_scheduled_tokens` 的实际调度规划；如果增大 MTP 候选数，通常需要重新评估 token budget。

---

## 9. API、模型名和本地访问

### 9.1 服务地址

当前地址是：

```text
http://127.0.0.1:28901
```

`127.0.0.1` 也叫 loopback，只能从运行服务的这台机器访问。它不是 `0.0.0.0`，因此不能直接从其他主机访问。

### 9.2 健康检查

```bash
curl http://127.0.0.1:28901/health
curl http://127.0.0.1:28901/v1/models
```

服务正常时，`/health` 应返回成功状态，`/v1/models` 会列出服务别名。

### 9.3 Chat Completions 请求示例

```bash
curl http://127.0.0.1:28901/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.8-27b-hauhau-aggressive",
    "messages": [
      {"role": "user", "content": "请用一句话解释 KV cache。"}
    ],
    "temperature": 0.6,
    "top_p": 0.95,
    "max_tokens": 256,
    "stream": false
  }'
```

这里的 `model` 是 `--served-model-name` 定义的 API 别名，不一定等于磁盘目录名，也不一定等于 Hugging Face 仓库名。

### 9.4 temperature、top_p、top_k 和 repetition penalty

这些是生成阶段的采样参数，不改变模型权重：

| 参数 | 含义 | 直观效果 |
|---|---|---|
| `temperature` | 调整概率分布的尖锐程度 | 越低通常越稳定、越确定；越高通常更多样 |
| `top_p` | 只从累计概率达到 p 的候选集合采样 | 限制长尾候选 |
| `top_k` | 只从概率最高的 k 个候选采样 | 直接限制候选数量 |
| `repetition_penalty` | 惩罚已出现过的 token | 减少重复，但过高可能损伤自然性 |
| `do_sample` | 是否按概率采样 | 关闭时更接近贪心或确定性解码 |

模型的 `generation_config.json` 默认包含 `temperature=1.0`、`top_p=0.95`、`top_k=20`、`repetition_penalty=1.05`。基准请求显式使用了 `temperature=0.6`、`top_p=0.95`，因此基准行为不完全等于只使用模型默认值。

### 9.5 EOS 和 ignore-eos

EOS 是 End Of Sequence，表示模型认为回答可以结束。正常服务通常在生成 EOS 时提前停止。

基准使用 `ignore-eos`，意思是即使模型生成 EOS，也继续生成，直到达到设定的输出 token 数或其他停止条件。这个选项适合做固定长度吞吐测试，因为每条请求都尽量生成同样多的 token；但它不代表正常聊天体验，也可能产生多余内容。

---

## 10. Qwen3.5 混合架构和多模态文件

当前配置中可以看到：

- `model_type: qwen3_5`；
- `num_hidden_layers: 64`；
- 同时存在 `linear_attention` 和 `full_attention` 层；
- 配置包含视觉模型和图像 / 视频 token；
- `model.safetensors` 中还包含视觉相关权重。

### 10.1 Full attention

全注意力会让当前 token 直接关注上下文中的相关 token。标准实现的计算和缓存成本会随序列长度明显增长，因此长上下文时 KV cache 需求较大。

### 10.2 Linear attention / Mamba 风格状态

线性注意力或状态空间风格模块使用另一类递推状态表示，目标是降低长序列的计算或缓存压力。它不等于普通 Transformer 的 full attention，因此 vLLM 需要针对这些模块使用专门实现和 kernel。

日志中出现了 Mamba cache mode、`mamba_ssm_dtype` 和相关 Triton kernel，说明当前架构不是只有传统 full attention。这个事实也解释了为什么不能只按照一个普通 Qwen Transformer 的显存公式估算全部资源。

### 10.3 多模态配置不等于本次请求用了图片

模型目录包含视觉配置、图像 token 和视频处理器配置，说明权重有多模态结构。启动日志同时提示当前 vLLM 没有为该模型注册对应的多模态 processor，因此本次以 text-only mode 运行。

本次基准发送的是随机文本 token，没有使用图片或视频。因而：

- 视觉权重仍可能占用磁盘或初始化资源；
- 但基准中的输入吞吐不代表图片 / 视频推理性能；
- 不能从这次文本基准推断视觉能力或多模态延迟。

---

## 11. CUDA Graph、Torch Compile、Triton 和 FlashInfer

这些是推理加速基础设施，不是模型能力。

### 11.1 Torch Compile

vLLM 可以将部分 PyTorch 计算图编译成更适合当前输入形状和 GPU 的执行形式。编译会增加首次启动时间，但可以减少重复解释和调度开销。

### 11.2 CUDA Graph

CUDA Graph 会把一组 GPU 操作预先捕获，之后重复执行时减少 CPU 发起 kernel 的开销。当前设置：

```json
{"cudagraph_mode":"PIECEWISE"}
```

PIECEWISE 可以理解为对适合捕获的部分进行分段图优化，更适合输入形状动态变化的服务场景。

日志显示启动阶段包含编译和 CUDA Graph capture，初始化耗时约 273.87 秒，其中编译约 113.51 秒。服务启动慢不等于每个请求都慢；编译缓存存在后，后续重启可能更快，但缓存失效或换形状时仍可能重新编译。

### 11.3 Triton JIT

Triton 是用于编写 GPU kernel 的工具。JIT（Just-In-Time compilation）表示第一次遇到某种形状或路径时才编译 kernel。日志中提示若干 Triton kernel 在推理期间首次编译，并明确说明会造成 latency spike。

这会造成以下现象：

- 服务已经显示 healthy，但第一批请求仍可能较慢；
- P95 可能明显高于 P50；
- 预热相同形状的请求后，后续延迟更稳定。

### 11.4 FlashInfer

FlashInfer 是用于注意力、KV cache 和 decode 等操作的高效 GPU kernel 库。日志显示它识别到 BF16 prefill、BF16 decode 和 FP8 KV cache。它负责执行路径，不改变模型的 tokenizer、权重知识或 API 协议。

---

## 12. 性能指标逐项解释

### 12.1 请求吞吐

```text
request_throughput = 完成请求数 / 测试总时长
```

单位是 `req/s`。例如并发 1 的 20 条请求在约 72.47 秒完成，得到约 0.276 req/s。

请求吞吐适合回答“单位时间完成多少个请求”，但不能单独代表 token 生成速度，因为不同请求的输入和输出长度可能不同。

### 12.2 输出吞吐

```text
output_throughput = 生成的输出 token 总数 / 测试总时长
```

单位是 `tok/s`。这次每条请求固定输出 256 token，因此它与请求吞吐有直接关系，但在真实请求长度不同时，两者不能互相替代。

### 12.3 总 token 吞吐

总 token 吞吐通常把输入 token 和输出 token 都算进去：

```text
total_token_throughput
  = (输入 token 总数 + 输出 token 总数) / 测试总时长
```

它适合衡量引擎处理 token 的总工作量，但与用户感受到的回答速度不完全相同。

### 12.4 TTFT

TTFT 是 Time To First Token，首 token 时间。从请求发出到收到第一个输出 token 的时间，通常包含：

- HTTP 和服务端排队；
- scheduler 调度；
- prompt tokenizer / prefill；
- 首次 decode；
- 必要的多 GPU 通信。

TTFT 越低，用户越快看到回答开始出现。长 prompt 通常主要影响 TTFT。

### 12.5 TPOT

TPOT 是 Time Per Output Token，首 token 之后每个输出 token 的平均间隔。它常被理解为流式输出速度的倒数：

```text
大致 token 速度 ≈ 1 / TPOT
```

例如 13.09 ms/token 大约对应 76 token/s 的单序列生成速度；实际汇总吞吐还会受到并发、批处理和 MTP 的影响。

### 12.6 P50、P95 和长尾

把所有请求按延迟从小到大排序：

- P50 是中位数，50% 请求不超过这个值；
- P95 是 95 分位数，95% 请求不超过这个值，剩下 5% 更慢。

P50 代表典型体验，P95 更能体现排队、JIT 编译、GPU 争用和长请求造成的尾部风险。不能只看 P50 就判断服务在高峰期是否稳定。

### 12.7 本次基准结果

| 并发 | 成功 | 请求吞吐 | 输出吞吐 | P50 TTFT | P50 TPOT | P50 端到端 | MTP 接受率 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20/20 | 0.276 req/s | 70.65 tok/s | 278.8 ms | 13.09 ms | 3.61 s | 53.96% |
| 4 | 40/40 | 0.758 req/s | 194.05 tok/s | 335.4 ms | 16.04 ms | 4.43 s | 54.72% |

并发 4 时的 P95 端到端延迟约 11.81 秒，明显高于 P50，说明存在长尾。并发增加后，整体请求吞吐和输出吞吐提高，但单请求的 TTFT、TPOT 和端到端延迟也上升，这是吞吐与延迟之间的常见折中。

本次每条请求约 1024 输入 token、256 输出 token。用近似关系检查结果：

```text
端到端时间 ≈ TTFT + 输出阶段耗时
            ≈ TTFT + 255 × TPOT
```

并发 1：`0.279 + 255 × 0.01309 ≈ 3.62 秒`，与 3.61 秒接近；并发 4：`0.335 + 255 × 0.01604 ≈ 4.43 秒`，与 4.43 秒接近。这里的 255 是因为第一个输出 token 已经计入 TTFT，具体统计实现可能有小幅差异。

### 12.8 这次测试能说明什么、不能说明什么

能说明：

- 当前硬件、模型格式、vLLM 参数组合下的服务速度；
- 并发从 1 增加到 4 后的吞吐和延迟变化；
- MTP 在这组随机输入上的候选接受情况；
- 固定输入输出长度下的相对性能。

不能说明：

- MMLU、GSM8K 等任务能力；
- HauhauCS 模型在真实对话中的质量；
- 图片或视频推理能力；
- 任意输入长度和任意并发下的 P95；
- 关闭 `ignore-eos` 后的真实聊天体验。

另外，基准的 `request_rate=inf` 表示请求以 burst 方式尽快发出，而不是按固定速率到达。因此它更接近压力批次测试，不是长时间稳定流量测试。

---

## 13. 常见误解速查

| 容易混淆的概念 | 正确理解 |
|---|---|
| GGUF vs AWQ | GGUF 主要是文件容器；AWQ 主要是权重量化方法 / 路线 |
| AWQ vs compressed-tensors | AWQ 是量化来源或方法名称；compressed-tensors 是当前权重格式和 vLLM 加载后端 |
| Safetensors vs BF16 | Safetensors 是文件格式；BF16 是张量数值类型 |
| INT4 vs FP8 | INT4 主要用于权重；FP8 当前主要用于 KV cache |
| `--dtype bfloat16` vs 解量化 | BF16 指普通计算精度，不会把 INT4 权重永久变回 BF16 |
| KV cache vs 模型权重 | 权重是模型本体；KV cache 是请求运行时的临时中间状态 |
| MTP vs 多输出 | MTP 提出候选并验证，不是无条件一次输出 3 个 token |
| TP=2 vs 两个副本 | TP=2 是两张卡共同保存和执行一份模型，不是两个独立服务副本 |
| max model len vs max output tokens | 前者是输入加输出的上下文上限；后者只限制输出部分 |
| P50 vs P95 | P50 是典型中位数；P95 反映慢请求长尾 |
| 27B vs 27 GB | 参数量不等于显存占用；量化、缓存和并行方式都会改变显存需求 |
| multimodal config vs multimodal benchmark | 有视觉配置不等于这次请求使用了图片或视频 |
| model directory vs served model name | 磁盘目录是加载路径；API 请求使用 `--served-model-name` 别名 |

---

## 14. 常用运维命令

### 14.1 激活环境

```bash
conda activate /data/miniconda3/envs/qwen38-hauhau-vllm
```

当前环境中使用 vLLM 0.27.1 和 PyTorch 2.13.0+cu130。环境决定可用的 vLLM 参数、CUDA kernel、量化后端和模型架构支持。

### 14.2 启动或重启

```bash
bash /data/ai/chenzhangyue/huggingface-models/qwen38-27b-Uncensored-HauhauCS/start-hauhau-vllm.sh
```

历史启动方式使用 `tee` 保存日志。现有独立入口加载 Hauhau，前台日志输出至终端；仓库内 systemd 配置使用 journal。启动阶段可能较慢，因为需要加载权重、分配 KV cache、编译 kernel、预热并捕获 CUDA Graph。

### 14.3 检查 GPU

```bash
nvidia-smi
nvidia-smi -i 5,6
```

重点观察：

- 物理 GPU 5、6 是否有 vLLM worker；
- 两张卡显存占用是否大致均衡；
- 是否有其他进程占用显存；
- 温度和功耗是否异常。

### 14.4 检查服务和日志

```bash
curl -s http://127.0.0.1:28901/health
curl -s http://127.0.0.1:28901/v1/models
tail -f /data/ai/chenzhangyue/huggingface-models/logs/hauhau-vllm-20261024-1908.log
```

出现首批请求较慢时，先检查日志中的 Triton JIT、CUDA Graph、编译和显存信息，再判断是否真的是持续性能问题。

---

## 15. 常见问题和处理方向

### 15.1 vLLM 报错不能加载 GGUF

原因通常是当前 vLLM 版本、模型架构或 GGUF 适配路径不匹配。解决方向是：

1. 使用当前 vLLM 明确支持的 Safetensors / compressed-tensors / AWQ 适配版；
2. 或使用支持该 GGUF 的 llama.cpp 等运行时；
3. 不要只修改扩展名，必须完成真实格式转换和配置适配。

### 15.2 显存不足

可按影响从小到大检查：

1. 确认没有其他进程占用物理 GPU 5、6；
2. 降低 `--max-num-seqs`；
3. 降低 `--max-model-len`；
4. 降低 `--max-num-batched-tokens`；
5. 保留 FP8 KV cache；
6. 重新评估 CUDA Graph、MTP 候选数和 GPU memory utilization。

降低上下文长度通常最直接地减少潜在 KV cache 需求，但会限制长 prompt；降低并发则会牺牲吞吐。

### 15.3 第一个请求慢，后面变快

常见原因是：

- 模型和 tokenizer 尚未完全加载；
- Torch Compile 尚未完成；
- CUDA Graph 尚未捕获对应形状；
- Triton kernel 第一次 JIT 编译；
- CUDA cache 或 kernel workspace 尚未建立。

这属于典型 warmup / JIT 行为。基准测试最好先发送若干和正式请求形状接近的预热请求，再开始统计。

### 15.4 并发提高后 P95 很大

可能原因包括：

- 请求被 scheduler 排队；
- 长 prompt 的 prefill 阻塞短请求；
- MTP 验证和候选 slots 增加调度压力；
- GPU 间通信成为瓶颈；
- 某些输入形状触发 JIT 编译；
- 输出长度或 EOS 行为不一致。

应同时观察 P50、P95、运行中请求数、等待队列、GPU KV cache usage 和日志中的吞吐，而不能只看平均值。

### 15.5 其他机器访问不到 API

当前 `--host 127.0.0.1` 只绑定本机。若确实需要远程访问，需要在网络和安全策略允许的前提下改为合适的监听地址，并配合反向代理、访问控制和认证。仅把服务暴露到公网而没有认证不适合作为生产配置。

### 15.6 API 报模型名不存在

请求中的 `model` 必须使用：

```text
qwen3.8-27b-hauhau-aggressive
```

它来自 `--served-model-name`。如果请求写成 Hugging Face 仓库名或本地目录名，服务可能返回模型不存在。

---

## 16. 一句话记忆版

可以用下面这段话概括当前部署：

> 原始模型是 GGUF 生态中的 HauhauCS 版本；为了让当前 vLLM 使用，实际加载的是转换后的 Safetensors 适配版。它以 compressed-tensors 方式加载以 AWQ 路线为主的 4-bit 分组量化权重，普通计算使用 BF16，KV cache 使用 FP8；模型被 Tensor Parallelism 拆到物理 GPU 5、6 上运行；MTP 每轮最多提出 3 个候选 token，再由主模型验证；vLLM 通过 OpenAI-compatible API 在本机 28901 端口提供服务。
