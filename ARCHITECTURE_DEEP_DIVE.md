# Nano-vLLM - 架构实现细节

## 核心数据流 (Data Flow)

```
用户输入 (prompts + sampling_params)
    ↓
LLMEngine.generate()
    ├─ tokenizer.encode() → token IDs
    ├─ add_request() → create Sequence → Scheduler.waiting queue
    │
    └─ 推理循环 (while not finished):
        ├─ Scheduler.schedule()
        │  ├─ Phase 1: Prefill - 处理新序列的输入提示词
        │  │  ├─ 从waiting队列选择序列
        │  │  ├─ BlockManager.allocate() → 分配KV缓存块
        │  │  ├─ 状态: WAITING → RUNNING
        │  │  └─ 返回 (seqs, is_prefill=True)
        │  │
        │  └─ Phase 2: Decode - 为已有KV缓存的序列生成下一token
        │     ├─ 从running队列选择序列
        │     ├─ 查询BlockManager是否可扩展缓存
        │     ├─ 若无法扩展则preempt低优先级序列
        │     └─ 返回 (seqs, is_prefill=False)
        │
        ├─ ModelRunner.call("run", seqs, is_prefill)
        │  ├─ Forward pass: 模型前向计算logits
        │  ├─ Sampler: 根据温度采样下一token
        │  └─ 返回 token_ids (for each seq)
        │
        ├─ Scheduler.postprocess(seqs, token_ids)
        │  ├─ seq.append_token(token_id)
        │  ├─ 检查是否完成 (EOS or max_tokens)
        │  ├─ 若完成: RUNNING → FINISHED
        │  ├─ BlockManager.deallocate() → 释放缓存
        │  └─ 从running队列移出
        │
        └─ 收集已完成序列的结果
            ├─ (seq_id, completion_token_ids)
            └─ 累积到outputs字典
    
    └─ 返序列结果后处理
        ├─ tokenizer.decode() → 文本
        └─ 返回 [{"text": ..., "token_ids": ...}]
```

---

## Prefill vs Decode 的区别

### Prefill 阶段
```
输入: "Hello, world!" (5 tokens)
操作: 一次性处理所有5个token，生成对应的KV缓存
计算复杂度: O(5²) ≈ 并行
显存需求: 5个token的KV值
输出: KV缓存 + 最后一个logits

特点: 
- 计算密集(compute-bound)
- 吞吐量高
- 延迟高(要处理完整提示词)
```

### Decode 阶段
```
输入: 上次的最后token (1 token)
操作: 使用缓存的KV值，计算下一个token
计算复杂度: O(1 × seq_len) ≈ 内存重(memory-bound)
显存需求: 1个新token的KV值
输出: 下一个token ID

特点:
- 访存密集(memory-bound)  
- 吞吐量低
- 延迟低(只计算1个token)
```

**为什么分离？**
- 不同的优化策略
- Prefill需要最大化并行度(多token)
- Decode需要提高缓存利用率(批量序列)
- 分离使调度更灵活

---

## 调度器 (Scheduler) 的核心算法

### 状态图
```
                    ┌──────────────┐
                    │   WAITING    │
                    └──────┬───────┘
                           │ schedule() 选中
                           ↓
                    ┌──────────────┐
              preempt│   RUNNING    │ append_token
              (内存不足)└──────┬───────┘
                           │ 
                  完成(EOS/max_tokens)
                           ↓
                    ┌──────────────┐
                    │   FINISHED   │
                    └──────────────┘
```

### Prefill调度逻辑

```python
for seq in waiting_queue:  # FIFO
    # 检查两个约束
    if can_fit_in_batch(seq) and can_allocate_kvcache(seq):
        # 满足条件，选中
        allocate_kvcache(seq)
        seq.status = RUNNING
        scheduled_seqs.append(seq)
    else:
        break  # 无法再添加，停止
```

**关键约束：**
1. `num_batched_tokens <= max_num_batched_tokens` (显存限制)
2. `num_seqs <= max_num_seqs` (序列数限制)  
3. `block_manager.can_allocate(seq)` (是否有空闲块)

**优先级：** FIFO (先进先出)

### Decode调度逻辑

```python
for seq in running_queue:
    while not block_manager.can_append(seq):
        # 内存不足，需要抢占
        if running_queue:
            preempt(running_queue.pop())  # 抢占末尾(低优先级)
        else:
            preempt(seq)  # 抢占自己，放回等待队列
            break
    else:
        # 成功为seq分配了内存
        block_manager.may_append(seq)
        scheduled_seqs.append(seq)
```

**抢占策略：** 抢占队列末尾的序列(相对较新的)

---

## KV缓存管理 (Block Manager)

### 块化策略

```
KV缓存不是连续存储，而是分成固定大小的块：

block_size = 256 tokens

Sequence A (300 tokens):
┌──256 tokens──┐┌──44 tokens───┐
└─ Block 0 ────┘└── Block 1 ────┘

Sequence B (100 tokens):
┌──100 tokens──┐
└─ Block 2 ────┘

Block Table:
Seq A: [0, 1]      (占用块0和块1)
Seq B: [2]         (占用块2)
```

**优势：**
1. **内存碎片少** - 固定块大小便于重用
2. **分配快** - 简单的块分配算法
3. **灵活性** - 块可被不同序列共享(前缀缓存)
4. **动态增长** - 易于支持增量KV缓存

### 分配流程

**Prefill时：**
```python
def allocate(seq):
    # 计算所需块数
    num_blocks_needed = ceil(len(seq) / block_size)
    
    # 从block manager的空闲块中分配
    blocks = block_manager.allocate_blocks(num_blocks_needed)
    
    # 记录到seq的block_table
    seq.block_table = blocks
    
    # 标记这些块为已用
    for block in blocks:
        block_manager.mark_used(block)
```

**Decode时：**
```python
def may_append(seq):
    # 检查最后一个块是否还有空间
    if seq.last_block_num_tokens < block_size:
        # 有空间，可以继续在现有块中追加
        pass
    else:
        # 后续append_token时会需要新块
        # block_manager会自动分配
        pass
```

**Deallocate时：** (序列完成或被抢占)
```python
def deallocate(seq):
    # 释放该序列的所有块
    for block_id in seq.block_table:
        block_manager.free_block(block_id)
    seq.block_table = []
    seq.num_cached_tokens = 0
```

---

## 多进程架构 (张量并行)

### 进程架构

```
主进程 (Rank 0)                    Worker 进程 (Rank 1-N)
────────────────────────────────────────────────────

LLMEngine
  │
  ├─ Tokenizer
  │
  ├─ Scheduler
  │
  ├─ ModelRunner (Rank 0)         ModelRunner (Rank 1)
  │   │                               │
  │   ├─ call("run", ...)         ├─ call("run", ...)
  │   │   │                       │   │
  │   │   └─ 转发请求到────────→→→ 处理请求
  │   │       event通知           (GPU 1)
  │   │
  │   └─ 前向计算              └─ AllReduce通信
  │       (GPU 0)              返回结果

  │
  └─ 合并结果 ←←←← 从Worker收集
```

### 进程间通信 (IPC)

```python
# 主进程
class ModelRunner:
    def call(self, fn_name: str, *args, **kwargs):
        # 准备数据包
        data = serialize(*args, **kwargs)
        
        # 发送给所有worker进程
        for queue in worker_queues:
            queue.put((fn_name, data))
        
        # 等待所有worker完成
        for event in events:
            event.wait()
            event.clear()
        
        # 收集结果
        results = ...
        return results

# Worker进程
def worker_main(config, rank, event):
    while True:
        fn_name, data = queue.get()
        
        if fn_name == "run":
            # 执行模型前向
            result = model_runner(data)
            
            # 与其他进程通信(AllReduce等)
            dist.all_reduce(result)
            
            # 发送完成信号
            event.set()
        elif fn_name == "exit":
            break
```

---

## 性能优化技术

### 1. 批处理 (Batching)

多个不同长度的序列在同一批中处理：
```
Batch 1 (Prefill):
  Seq A (Length 256): [token, token, ..., token]
  Seq B (Length 128): [token, token, ..., PAD, PAD, ...]
  (Seq B被padding到Seq A的长度)
  导致计算量 = (256 + 128) tokens

vs 无批处理:
  Seq A: length 256
  Seq B: length 128
  总计算量 = 256 + 128 (相同，但无法利用GPU并行)

→ 批处理让GPU充分利用
```

### 2. 增量KV缓存

避免重复计算已有的KV值：
```
第1步:
  输入: "Hello world" (12 tokens)
  计算: 全部12个tokens的KV值
  
第2步:
  已有KV: 12 tokens
  新输入: 1 token (下一个)
  计算: 仅1个新token的KV值 ✓

vs 无增量缓存:
  每步都重新计算所有tokens的KV值 ✗
```

### 3. 张量并行

在多张GPU上分布式计算：
```
单GPU (RTX 4070 8GB):
  model_size = 600M parameters
  kvcache_size = 可能不足

双GPU 张量并行:
  GPU 0: 300M参数 + KV缓存1
  GPU 1: 300M参数 + KV缓存2
  
  前向传播时:
  Linear屈(权重分割):
    GPU 0: output = input @ W0
    GPU 1: output = input @ W1
    AllReduce: 合并结果
```

### 4. 采样温度效应

```
Temperature = 0.6 (低):
  原始logits: [2.0, 1.0, 0.5]
  softmax结果: [0.66, 0.24, 0.10]
  after /0.6: [0.87, 0.09, 0.04] ← 更集中

Temperature = 1.0 (正常):
  原始logits: [2.0, 1.0, 0.5]
  softmax结果: [0.66, 0.24, 0.10]

Temperature = 1.5 (高):
  原始logits: [2.0, 1.0, 0.5]
  after /1.5: [0.57, 0.31, 0.12] ← 更均衡
  
→ 调整温度改变token概率分布
```

---

## 错误处理和边界情况

### 如果显存不足

```
Scenario: num_batched_tokens 超过GPU显存

场景1: Prefill阶段
─────────────────
第N个序列要加入，但加上它会超过token限制
→ 不添加，返回前N-1个序列
→ 下一轮继续
✓ 优雅降级，保证始终有进度

场景2: Decode阶段
──────────────
所有running序列都无法继续decode
→ 抢占最后一个(未来可能最快完成的)  
→ 释放它的缓存
→ 该序列回到waiting队列
✓ 通过抢占保证forward progress

场景3: 单个序列太长
──────────────────
甚至单个序列都无法分配KV缓存
→ 配置其实不合理
→ 应该增加max_num_kvcache_blocks或减少max_model_len
⚠️ 需要用户调整配置
```

### 如果EOS出现早期

```
max_tokens = 256
模型在第128 token时生成EOS

默认行为 (ignore_eos=False):
  生成停止
  返回128个completion tokens  ✓

强制继续 (ignore_eos=True):
  忽略EOS，继续到第256个token
  返回256个completion tokens  ✓
```

---

## 关键性能指标

### Throughput (吞吐量)

**定义：** tokens per second (tok/s)

```
吞吐量 (Prefill阶段):
  = Prefill token数 / Prefill时间
  例如: 1000 tokens / 1s = 1000 tok/s
  (高吞吐: 利用GPU并行性好)

吞吐量 (Decode阶段):
  = Decode token数 / Decode时间
  例如: 256 seqs × 1 tok / 0.5s = 512 tok/s
  (低吞吐: 内存带宽限制)
```

### Latency (延迟)

```
首token延迟 (Time To First Token, TTFT):
  = Prefill完成时间
  用户体验关键
  越小越好

全序列延迟 (Tail Latency):
  = 整个序列生成完成时间
```

### 内存利用率

```
GPU显存 = 模型参数 + KV缓存 + 计算中间变量

总计 ≈ 
  model_params * 2 bytes (fp16)
  + kvcache_size * 2 * batch_size * seq_len
  + 计算buffer
```

---

## 常见配置问题和解决方案

### 问题1: 显存溢出 (OOM)

```
症状: CUDA out of memory

原因检查:
1. 模型太大 → 减少tensor_parallel_size或用量化
2. KV缓存配置过大 → 减少gpu_memory_utilization
3. max_num_batched_tokens太大 → 减小
4. max_num_seqs太大 → 减小

解决:
config = Config(
    "model_path",
    max_num_batched_tokens=8192,      # 从16384减小
    max_num_seqs=256,                 # 从512减小
    gpu_memory_utilization=0.8,       # 从0.9减小
)
```

### 问题2: 低吞吐量

```
症状: 生成速度慢，tok/s过低

排查:
1. 批大小太小 → 增加max_num_seqs或max_num_batched_tokens
2. 张量并行未正确启用 → 检查CUDA_VISIBLE_DEVICES
3. 频繁抢占 → block_manager缺块，增加num_kvcache_blocks

优化:
config = Config(
    "model_path",
    max_num_batched_tokens=32768,     # 增大
    max_num_seqs=512,                 # 增大
    tensor_parallel_size=2,           # 启用多GPU
)
```

### 问题3: 不均衡的prefill/decode

```
症状: Prefill非常快，但Decode缓慢

原因:
- Decode是内存密集型，受显存带宽限制
- 这是正常的!

但如果drop太大:
1. 批处理太小 → 增加max_num_seqs
2. 序列太长 → KV访问模式不友好 → 无法优化

优化方向:
- Flash Attention (降低访存)
- 使用更快的GPU
- 调整batch大小
```

---

## 学习路径

**初级：** 理解整体流程
1. 读PROJECT_GUIDE.md的"基础推理流程"
2. 运行example.py观察输出
3. 理解主要类的职责

**中级：** 深入算法实现
1. 理解Scheduler的schedule()方法
2. 阅读BlockManager的实现
3. poke model_runner的forward逻辑

**高级：** 优化和扩展
1. 修改调度策略
2. 实现新的采样算法
3. 添加新的优化(如张量并行改进)

开心学习！🚀
