# Nano-vLLM 项目完整指南

## 项目概述

**Nano-vLLM** 是一个轻量级的大型语言模型(LLM)推理引擎实现，从零开始构建，用Python编写，仅需约1,200行代码。该项目旨在提供与vLLM相媲美的推理性能，同时保持代码的可读性和易维护性。

### 核心特性

1. **高速离线推理** - 推理速度与官方vLLM相当甚至更优
2. **清晰的代码实现** - 使用纯Python实现，易于理解和学习
3. **优化算法支持**:
   - KV缓存管理和前缀缓存
   - 张量并行(Tensor Parallelism)
   - CUDA图优化
   - Torch动态编译

---

## 项目架构

```
nano-vllm/
├── __init__.py           # 导出主要接口(LLM, SamplingParams)
├── config.py            # 全局配置管理类
├── llm.py               # LLM推理类(继承自LLMEngine)
├── sampling_params.py   # 采样参数配置
│
├── engine/              # 核心推理引擎
│   ├── llm_engine.py        # 主引擎类，协调各个组件
│   ├── model_runner.py      # 模型运行器，执行实际推理
│   ├── scheduler.py         # 调度器，负责序列调度和内存管理
│   ├── sequence.py          # 单个请求序列的状态表示
│   └── block_manager.py     # KV缓存块管理器
│
├── layers/              # LLM神经网络层实现
│   ├── attention.py         # 多头注意力机制(支持Flash Attention)
│   ├── linear.py            # 线性变换层
│   ├── layernorm.py         # 层归一化
│   ├── activation.py        # 激活函数(GELU等)
│   ├── embed_head.py        # 嵌入和输出头
│   ├── rotary_embedding.py  # RoPE位置编码
│   └── sampler.py           # Token采样层
│
├── models/              # 具体模型实现
│   └── qwen3.py            # Qwen3模型定义
│
└── utils/               # 工具函数
    ├── loader.py            # 模型权重加载器
    └── context.py           # 上下文管理器
```

---

## 核心组件详解

### 1. 配置管理 (config.py)

`Config`类使用dataclass定义了所有推理参数：

- **model**: 模型权重路径
- **max_num_batched_tokens**: 单个批次最大token数(默认16384)
- **max_num_seqs**: 最多并发序列数(默认512)
- **max_model_len**: 最大上下文长度(默认4096)
- **tensor_parallel_size**: 张量并行大小(1-8)
- **gpu_memory_utilization**: GPU内存利用率(默认0.9)
- **enforce_eager**: 是否禁用torch编译
- **kvcache_block_size**: KV缓存块大小(256的倍数)

### 2. LLM引擎 (llm_engine.py)

主要职责：
- **初始化**: 加载模型、分词器、初始化多进程(如果需要张量并行)
- **请求管理**: 添加新请求序列
- **推理循环**: 调度序列、执行模型推理、处理结果
- **生成接口**: 提供与vLLM相同的`generate()`接口

关键方法：
```python
add_request()   # 添加单个请求到调度器
step()          # 执行一个推理步骤
generate()      # 批量生成文本(高层接口)
```

### 3. 序列管理 (sequence.py)

`Sequence`类代表单个用户请求的处理状态：

状态机：
```
WAITING → RUNNING → FINISHED
   ↑        ↓
   └────────┘ (抢占机制)
```

属性：
- **token_ids**: 完整的token序列(提示词 + 生成的token)
- **status**: 当前序列状态
- **block_table**: 该序列在KV缓存中的块位置
- **sampling_params**: 采样参数(temperature, max_tokens)

### 4. 调度器 (scheduler.py)

负责：
- **序列调度**: 从等待队列中选择序列进行prefill或decode
- **内存管理**: 协调BlockManager进行KV缓存分配
- **抢占策略**: 当内存不足时进行低优先级序列的抢占

工作流程：
1. **Prefill阶段**: 处理新序列的提示词，生成KV缓存
2. **Decode阶段**: 生成后续token，增量更新KV缓存

### 5. 模型运行器 (model_runner.py)

执行实际的模型推理：
- 管理GPU内存和KV缓存块
- 前向传播计算
- 处理张量并行通信(多卡时)

### 6. KV缓存管理 (block_manager.py)

高效管理KV缓存(Key-Value缓存)：
- **块化存储**: 将缓存分成固定大小的块
- **内存动态分配**: 根据序列长度动态分配/回收块
- **前缀共享**: 支持多序列间的缓存复用(前缀缓存优化)

---

## 工作流程详解

### 基础推理流程

```
1. 用户调用 llm.generate(prompts, sampling_params)
   ↓
2. LLMEngine将每个提示词转换为Sequence，添加到调度器
   ↓
3. 推理循环(while not finished):
   ├─ Scheduler.schedule()
   │  ├─ Prefill阶段: 处理新序列的提示词
   │  └─ Decode阶段: 为运行中的序列生成下一个token
   │
   ├─ ModelRunner执行实际推理
   │  ├─ 前向传播: 计算logits
   │  ├─ Sampler: 采样下一个token
   │  └─ KV缓存更新
   │
   └─ Scheduler.postprocess()
      ├─ 验证EOS或达到max_tokens
      └─ 更新序列状态
   ↓
4. 所有序列完成，返回生成结果
```

### 推理性能优化

1. **批量推理**: 
   - 多个序列在同一批中处理
   - max_num_batched_tokens限制显存占用

2. **KV缓存优化**:
   - 块化存储减少碎片
   - 前缀缓存支持不同长度输入的缓存复用

3. **Prefill vs Decode分离**:
   - Prefill: 并行处理提示词，生成初始KV缓存
   - Decode: 逐token生成，充分利用批量计算

4. **张量并行**:
   - 多个GPU上分布式推理
   - 进程间使用共享内存通信

---

## 使用示例

### 基础使用

```python
from nanovllm import LLM, SamplingParams

# 1. 初始化模型
llm = LLM(
    model_path="/path/to/Qwen3-0.6B",
    enforce_eager=True,              # 禁用torch编译(调试时)
    tensor_parallel_size=1           # 单GPU
)

# 2. 定义采样参数
sampling_params = SamplingParams(
    temperature=0.6,                 # 控制随机性
    max_tokens=256                   # 最多生成256个token
)

# 3. 生成文本
prompts = [
    "Hello, how are you?",
    "What is 2+2?"
]
outputs = llm.generate(prompts, sampling_params)

# 4. 获取结果
for output in outputs:
    print(output["text"])            # 生成的文本
    print(output["token_ids"])       # 对应的token ID
```

### 多序列处理

```python
# 不同序列使用不同的采样参数
sampling_params_list = [
    SamplingParams(temperature=0.6, max_tokens=128),
    SamplingParams(temperature=0.9, max_tokens=256),
]
outputs = llm.generate(prompts, sampling_params_list)
```

### 高级配置

```python
llm = LLM(
    model_path="/path/to/model",
    max_num_batched_tokens=32768,    # 增加显存使用
    max_num_seqs=256,                # 并发序列数
    gpu_memory_utilization=0.95,     # 提高内存利用率
    tensor_parallel_size=2,          # 双GPU张量并行
)
```

---

## 性能基准

在RTX 4070 Laptop (8GB VRAM)上的测试结果：

| 指标 | 数值 |
|------|------|
| 模型 | Qwen3-0.6B |
| 总请求数 | 256 sequences |
| 输入长度 | 100-1024 tokens (随机) |
| 输出长度 | 100-1024 tokens (随机) |
| | |
| **vLLM** | |
| 总token数 | 133,966 |
| 总时间 | 98.37s |
| 吞吐量 | 1,361.84 tokens/s |
| | |
| **Nano-vLLM** | |
| 总token数 | 133,966 |
| 总时间 | 93.41s |
| **吞吐量** | **1,434.13 tokens/s** ✨ |

**结论**: Nano-vLLM以~5%的性能优势胜过vLLM

---

## 关键设计选择

### 1. 多进程架构
- 使用进程池而非线程，避免GIL限制
- 每个参与张量并行的GPU一个进程
- 进程间通过共享内存通信

### 2. 块化KV缓存
- 固定块大小(256)便于内存管理
- 支持序列间的缓存块复用(前缀缓存)
- 减少内存碎片

### 3. 调度策略
- 优先级: Prefill > Decode
- Prefill和Decode分离处理
- 动态抢占实现资源公平分配

### 4. 采样参数约束
- 不支持贪心采样(temperature必须 > 1e-10)
- 强制随机性，避免确定性问题

---

## 文件依赖关系

```
__init__.py
├── llm.py
│   └── engine/llm_engine.py
│       ├── config.py
│       ├── sampling_params.py
│       ├── engine/sequence.py
│       ├── engine/scheduler.py
│       │   └── engine/block_manager.py
│       └── engine/model_runner.py
│           ├── models/qwen3.py
│           ├── layers/*
│           └── utils/loader.py
│
└── sampling_params.py
```

---

## 扩展点

### 支持新模型
1. 在`nanovllm/models/`中创建新模型类
2. 定义模型架构(embed, decoder layers等)
3. 在`config.py`中添加模型选择逻辑

### 添加新优化
1. **层融合**: 在`layers/`中优化kernel
2. **量化**: 集成int8/int4量化
3. **推测解码**: 快速模型生成后续token预测

### 自定义调度策略
- 修改`scheduler.py`中的`schedule()`方法
- 实现基于优先级/延迟的调度策略

---

## 依赖项

```
torch>=2.4.0           # PyTorch框架
triton>=3.0.0          # Triton编译器(自定义kernel)
transformers>=4.51.0   # Hugging Face模型加载
flash-attn             # 优化的注意力算子
xxhash                 # 快速哈希(可选)
```

---

## 总结

Nano-vLLM展示了如何用不到1200行代码实现与生产级系统相媲美的性能。其模块化设计使其成为学习LLM推理系统的理想参考实现，同时提供了足够的性能用于实际应用。

主要学习价值：
- ✅ 理解LLM推理系统的整体架构
- ✅ 学习序列调度和内存管理策略
- ✅ 掌握GPU推理优化技巧
- ✅ 参考生产级别的代码设计
