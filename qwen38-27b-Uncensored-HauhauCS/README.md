# HauhauCS 兼容入口

保留现有目录及 `start-hauhau-vllm.sh` 文件名。该脚本将全部参数转发给 `../qwen38-27b-Base/start-qwen38-bf16-vllm.sh`，启动当前官方 BF16 部署。

历史权重位于 `../models/HauhauCS-Qwen3.8-27B-Aggressive-AWQ-MTP/`，仅保留在本机。历史基准脚本位于 `../benchmarks/hauhau-local-20261024/`。兼容入口不会恢复 AWQ 服务。

```bash
bash qwen38-27b-Uncensored-HauhauCS/start-hauhau-vllm.sh --dry-run
```

启动配置和操作步骤见[根目录说明](../README.md)。
