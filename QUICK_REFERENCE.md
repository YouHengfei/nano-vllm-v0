# Nano-vLLM 快速参考指南 (Quick Reference)

## 最快上手 (30秒入门)

```python
from nanovllm import LLM, SamplingParams

# 1. 初始化模型（第一次会下载或加载本地模型）
llm = LLM("/path/to/model")

# 2. 定义采样参数
params = SamplingParams(temperature=0.6, max_tokens=256)

# 3. 生成文本
outputs = llm.generate(["Hello, world!"], params)

# 4. 获取结果
print(outputs[0]["text"])
```

---

## 关键类和方法速查

### LLM 类

```python
class LLM:
    # 构造函数
    __init__(model, **kwargs)
    # 参数：
    #   model: str - 模型权重目录
    #   tensor_parallel_size: int = 1 - 并行GPU数
    #   max_num_batched_tokens: int = 16384 - 批量token限制
    #   max_num_seqs: int = 512 - 最多序列数
    #   gpu_memory_utilization: float = 0.9 - GPU利用率
    #   enforce_eager: bool = False - 禁用torch.compile
    
    # 生成文本（主要接口）
    generate(prompts, sampling_params, use_tqdm=True)
    # 参数：
    #   prompts: list[str] | list[list[int]] - 输入提示词
    #   sampling_params: SamplingParams | list[SamplingParams] - 采样参数
    #   use_tqdm: bool - 是否显示进度条
    # 返回：
    #   list[{"text": str, "token_ids": list[int]}]
    
    # 添加单个请求
    add_request(prompt, sampling_params)
    
    # 执行一次推理步骤
    step() -> (outputs, num_tokens)
    
    # 检查是否完成
    is_finished() -> bool
```

### SamplingParams 类

```python
class SamplingParams:
    temperature: float = 1.0
    # 采样温度
    # < 1.0: 更确定性，输出更集中
    # = 1.0: 原始概率分布
    # > 1.0: 更随机，输出更多样
    # 推荐值：0.6-0.9
    
    max_tokens: int = 64
    # 最多生成的token数（不含输入）
    # 推荐值：128-512
    
    ignore_eos: bool = False
    # 是否忽略EOS token
    # False: 遇到EOS即停止（正常）
    # True: 强制生成直到max_tokens
```

---

## 常见使用场景

### 场景1: 单个提示词，简单生成

```python
llm = LLM("model_path", enforce_eager=True)
params = SamplingParams(temperature=0.7, max_tokens=256)
result = llm.generate(["What is AI?"], params)
print(result[0]["text"])
```

### 场景2: 批量生成多个提示词

```python
llm = LLM("model_path")
params = SamplingParams(temperature=0.6, max_tokens=128)

prompts = [
    "Explain machine learning",
    "What is neural networks?",
    "Define natural language processing"
]

results = llm.generate(prompts, params)

for i, result in enumerate(results):
    print(f"Q{i+1}: {result['text']}\n")
```

### 场景3: 不同提示词用不同参数

```python
llm = LLM("model_path")

prompts = [
    "Generate a creative story",
    "What is 2 + 2?"
]

params = [
    SamplingParams(temperature=0.9, max_tokens=512),  # 创意写作
    SamplingParams(temperature=0.1, max_tokens=32),   # 事实性问题
]

results = llm.generate(prompts, params)
```

### 场景4: 手动控制推理循环

```python
llm = LLM("model_path")
params = SamplingParams(temperature=0.6, max_tokens=100)

# 添加多个请求
for prompt in ["Hello", "World"]:
    llm.add_request(prompt, params)

# 手动执行推理步骤
results = {}
while not llm.is_finished():
    outputs, num_tokens = llm.step()
    
    for seq_id, token_ids in outputs:
        results[seq_id] = token_ids
    
    print(f"Processed {num_tokens} tokens")

# 解码结果
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("model_path")
for seq_id, token_ids in results.items():
    print(f"Result {seq_id}: {tokenizer.decode(token_ids)}")
```

### 场景5: 启用多GPU张量并行

```python
import os

# 指定使用的GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# 创建2-GPU张量并行
llm = LLM(
    "model_path",
    tensor_parallel_size=2,
    max_num_batched_tokens=32768,  # 可以增加
)

params = SamplingParams(temperature=0.6, max_tokens=256)
results = llm.generate(["Hello world"], params)
```

---

## 性能优化建议

### 吞吐量优先

```python
llm = LLM(
    "model_path",
    max_num_batched_tokens=32768,    # ↑ 增加
    max_num_seqs=512,                # ↑ 增加
    gpu_memory_utilization=0.95,     # ↑ 增加
    tensor_parallel_size=2,          # ↑ 多GPU
)

# 发送更多请求
prompts = ["prompt"] * 100  # 大批量
params = SamplingParams(temperature=0.6, max_tokens=100)
llm.generate(prompts, params)
```

### 延迟优先 (快速响应)

```python
llm = LLM(
    "model_path",
    max_num_batched_tokens=16384,    # ↓ 减少
    max_num_seqs=32,                 # ↓ 减少
    gpu_memory_utilization=0.8,      # ↓ 减少
)

# 单个或少量请求
prompts = ["What is AI?"]
params = SamplingParams(temperature=0.6, max_tokens=256)
llm.generate(prompts, params)
```

### 低显存设备

```python
llm = LLM(
    "model_path",
    max_num_batched_tokens=8192,     # ↓ 减少
    max_num_seqs=128,                # ↓ 减少
    gpu_memory_utilization=0.7,      # ↓ 减少
    enforce_eager=True,              # 禁用编译
)
```

---

## 调试技巧

### 打印配置信息

```python
from nanovllm.config import Config

config = Config(
    "model_path",
    tensor_parallel_size=1,
    max_num_batched_tokens=16384,
)

print(f"Model: {config.model}")
print(f"Max model len: {config.max_model_len}")
print(f"Max num seqs: {config.max_num_seqs}")
print(f"KV cache blocks: {config.num_kvcache_blocks}")
```

### 禁用进度条

```python
params = SamplingParams(temperature=0.6, max_tokens=256)
results = llm.generate(
    prompts,
    params,
    use_tqdm=False  # 不显示进度条
)
```

### 跟踪内存使用

```python
import torch
import gc

# 清理缓存
gc.collect()
torch.cuda.empty_cache()

# 获取当前显存使用
total = torch.cuda.get_device_properties(0).total_memory
reserved = torch.cuda.memory_reserved(0)
allocated = torch.cuda.memory_allocated(0)

print(f"Total: {total / 1e9:.2f}GB")
print(f"Reserved: {reserved / 1e9:.2f}GB")
print(f"Allocated: {allocated / 1e9:.2f}GB")
print(f"Free: {(total - allocated) / 1e9:.2f}GB")
```

---

## 错误处理

### CUDA内存溢出 (OOM)

```python
# ❌ 错误（显存不足）
llm = LLM("model_path", max_num_batched_tokens=65536)

# ✅ 正确（减小配置）
llm = LLM("model_path", max_num_batched_tokens=8192)

# ✅ 或者启用张量并行分担
llm = LLM("model_path", tensor_parallel_size=2)
```

### 模型加载失败

```python
# ❌ 错误（路径不存在）
llm = LLM("invalid/path")

# ✅ 正确（使用有效路径）
llm = LLM("~/huggingface/Qwen3-0.6B")

# ✅ 或者使用相对路径
llm = LLM("./models/Qwen3-0.6B")
```

### 温度参数错误

```python
# ❌ 错误（不支持贪心采样）
params = SamplingParams(temperature=0.0)  # ValueError!

# ✅ 正确（使用微小值）
params = SamplingParams(temperature=0.01)

# ✅ 或者设置较低的温度
params = SamplingParams(temperature=0.1)
```

---

## 模型下载

### 使用Hugging Face CLI

```bash
# 安装CLI工具
pip install huggingface-hub

# 下载模型
huggingface-cli download \
    Qwen/Qwen3-0.6B \
    --local-dir ~/huggingface/Qwen3-0.6B \
    --local-dir-use-symlinks False
```

### 在代码中自动下载

```python
from transformers import AutoModel, AutoTokenizer

# 首次会自动下载到~/.cache/huggingface
model_id = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
```

---

## 性能基准对比

| 引擎 | 吞吐量(tok/s) | 显存(RTX4070) | 代码行数 |
|-----|----------|-----------|--------|
| vLLM | 1361.84 | 接近满载 | ~10K |
| **Nano-vLLM** | **1434.13** ✨ | 优化更好 | **~1200** |

---

## 进阶主题

### 自定义调度策略

修改 `nanovllm/engine/scheduler.py` 中的 `schedule()` 方法

### 添加新的采样算法

修改 `nanovllm/layers/sampler.py`

### 支持新的模型架构

在 `nanovllm/models/` 添加新的模型类

### 使用自定义Layer融合

在 `nanovllm/layers/` 中优化kernel实现

---

## 文档导航

- 📘 [完整项目指南](PROJECT_GUIDE.md) - 详细的项目说明
- 🏗️ [架构深度解析](ARCHITECTURE_DEEP_DIVE.md) - 内部实现细节
- 💡 [代码注释](nanovllm/) - 源代码中的详细注释
- 🔗 [官方GitHub](https://github.com/GeeeekExplorer/nano-vllm)

---

## 常见问题 (FAQ)

**Q: Nano-vLLM支持哪些模型？**
A: 支持任何通过Hugging Face托管的模型(Qwen, Llama, etc)

**Q: 可以用CPU推理吗？**
A: 不行，当前只支持GPU (CUDA)

**Q: 支持量化吗？**
A: 当前版本不支持，可以手动集成

**Q: 如何评估生成质量？**
A: 使用BLEU, ROUGE等指标，或人工评估

**Q: 生成结果的可复现性如何？**
A: 相同的温度和模型权重应该产生不同的结果（因为温度>0）
  要让结果可复现，需要固定random seed和temperature

**Q: 性能为什么不如期望？**
A: 检查GPU使用率、batch大小、显存配置等参数

---

**祝您使用愉快！** 🚀
