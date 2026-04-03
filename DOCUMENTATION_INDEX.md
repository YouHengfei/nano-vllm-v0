# 📚 Nano-vLLM 文档导航索引

此文件帮助您快速找到所需的文档和信息。

---

## 🎯 按使用场景导航

### "我想快速运行代码"
→ [QUICK_REFERENCE.md](QUICK_REFERENCE.md#最快上手-30秒入门)
- 30 秒快速开始
- 最简单的示例代码  
- 常见配置

### "我想理解项目架构"
→ [PROJECT_GUIDE.md](PROJECT_GUIDE.md#项目架构)
- 项目结构图
- 6 个核心组件详解
- 工作流程

### "我想深入理解内部实现"
→ [ARCHITECTURE_DEEP_DIVE.md](ARCHITECTURE_DEEP_DIVE.md)
- 详细的数据流
- 算法伪代码
- 性能优化原理

### "我遇到了问题"
→ [QUICK_REFERENCE.md#错误处理](QUICK_REFERENCE.md#错误处理)
- OOM 错误
- 模型加载失败
- 参数验证错误

### "我想查看 API"
→ [QUICK_REFERENCE.md#关键类和方法速查](QUICK_REFERENCE.md#关键类和方法速查)
- LLM 类接口
- SamplingParams 参数
- 方法签名和返回值

---

## 📖 按学习阶段导航

### 初级（第 1 天）
```
1. README.md                           ← 项目简介
2. QUICK_REFERENCE.md#最快上手        ← 快速开始
3. 运行 example.py                    ← 实际操作
4. QUICK_REFERENCE.md#关键类和方法速查  ← API 参考
```

### 中级（第 2-3 天）
```
1. PROJECT_GUIDE.md                        ← 完整指南
2. PROJECT_GUIDE.md#核心组件详解          ← 各组件说明
3. nanovllm/*.py 代码注释                 ← 源代码浏览
4. ARCHITECTURE_DEEP_DIVE.md#核心数据流   ← 数据流图
```

### 高级（第 4 天+）
```
1. ARCHITECTURE_DEEP_DIVE.md            ← 深度解析
2. ARCHITECTURE_DEEP_DIVE.md#调度器     ← 调度算法
3. nanovllm/engine/scheduler.py 源码    ← 详细代码
4. QUICK_REFERENCE.md#场景4             ← 手动循环控制
```

---

## 🔍 按代码文件导航

### nanovllm/__init__.py
📖 文档：[QUICK_REFERENCE.md#最快上手](QUICK_REFERENCE.md#最快上手-30秒入门)
🔗 说明：模块导入和公共接口

### nanovllm/llm.py  
📖 文档：[PROJECT_GUIDE.md#llm引擎](PROJECT_GUIDE.md#llm引擎)
✨ 代码注释：50+ 行详细说明
🔗 说明：LLM 推理类

### nanovllm/sampling_params.py
📖 文档：[PROJECT_GUIDE.md#工作流程](#推理循环)
✨ 代码注释：50+ 行详细说明
🔗 说明：采样参数配置

### nanovllm/config.py
📖 文档：[PROJECT_GUIDE.md#配置管理](PROJECT_GUIDE.md#配置管理)
✨ 代码注释：200+ 行详细说明
🔗 说明：全局配置参数

### nanovllm/engine/llm_engine.py
📖 文档：[PROJECT_GUIDE.md#llm引擎](PROJECT_GUIDE.md#llm引擎)
📖 深度：[ARCHITECTURE_DEEP_DIVE.md#核心数据流](ARCHITECTURE_DEEP_DIVE.md#核心数据流)
✨ 代码注释：300+ 行详细说明
🔗 说明：核心推理引擎

### nanovllm/engine/sequence.py
📖 文档：[PROJECT_GUIDE.md#序列管理](PROJECT_GUIDE.md#序列管理)
✨ 代码注释：300+ 行详细说明
🔗 说明：序列数据结构和生命周期

### nanovllm/engine/scheduler.py
📖 文档：[PROJECT_GUIDE.md#调度器](PROJECT_GUIDE.md#调度器)
📖 深度：[ARCHITECTURE_DEEP_DIVE.md#调度器-scheduler](ARCHITECTURE_DEEP_DIVE.md#kv缓存管理-block-manager)
✨ 代码注释：400+ 行详细说明
🔗 说明：序列调度和内存管理

### nanovllm/engine/block_manager.py
📖 文档：[PROJECT_GUIDE.md#kv缓存管理](PROJECT_GUIDE.md#kv缓存管理)
📖 深度：[ARCHITECTURE_DEEP_DIVE.md#kv缓存管理](ARCHITECTURE_DEEP_DIVE.md#kv缓存管理-block-manager)
🔗 说明：KV 缓存块管理

### nanovllm/engine/model_runner.py
📖 文档：[PROJECT_GUIDE.md#模型运行器](PROJECT_GUIDE.md#模型运行器)
🔗 说明：模型前向计算执行

---

## 🎓 按概念导航

### Prefill vs Decode
📖 解释：[ARCHITECTURE_DEEP_DIVE.md#prefill-vs-decode-的区别](ARCHITECTURE_DEEP_DIVE.md#prefill-vs-decode-的区别)
🔗 代码：`scheduler.py` 中的 `schedule()` 方法

### KV 缓存
📖 基础：[PROJECT_GUIDE.md#kv缓存管理](PROJECT_GUIDE.md#kv缓存管理)
📖 深度：[ARCHITECTURE_DEEP_DIVE.md#kv缓存管理](ARCHITECTURE_DEEP_DIVE.md#kv缓存管理-block-manager)
🔗 代码：`sequence.py` 中的块相关属性

### 调度策略
📖 基础：[PROJECT_GUIDE.md#调度器](PROJECT_GUIDE.md#调度器)
📖 深度：[ARCHITECTURE_DEEP_DIVE.md#调度器-scheduler](ARCHITECTURE_DEEP_DIVE.md#调度器-scheduler-的核心算法)
🔗 代码：`scheduler.py` 中的 `schedule()` 和 `preempt()` 方法

### 多进程架构
📖 解释：[ARCHITECTURE_DEEP_DIVE.md#多进程架构-张量并行](ARCHITECTURE_DEEP_DIVE.md#多进程架构-张量并行)
🔗 代码：`llm_engine.py` 中的 `__init__()` 方法

### 性能优化
📖 技术：[ARCHITECTURE_DEEP_DIVE.md#性能优化技术](ARCHITECTURE_DEEP_DIVE.md#性能优化技术)
📖 配置：[QUICK_REFERENCE.md#性能优化建议](QUICK_REFERENCE.md#性能优化建议)
🔗 参数：`config.py`

---

## 💻 按任务导航

### 任务：快速开始运行模型
```
1. 下载模型 - 参考 QUICK_REFERENCE.md#模型下载
2. 编写代码 - 参考 QUICK_REFERENCE.md#最快上手
3. 运行代码 - python your_script.py
4. 查看结果 - result["text"]
```

### 任务：批量生成结果
```
1. 准备 prompts 列表
2. 创建 SamplingParams - 参考 QUICK_REFERENCE.md#samplingparams-类
3. 调用 llm.generate() - 参考 QUICK_REFERENCE.md#场景2
4. 处理结果 - for result in outputs
```

### 任务：优化性能
```
1. 测量基线性能 - 参考 PROJECT_GUIDE.md#性能基准
2. 选择优化方向 - 参考 QUICK_REFERENCE.md#性能优化建议
3. 修改配置 - 参考 config.py 源码
4. 重新测试 - 使用 bench.py
```

### 任务：调试问题
```
1. 确定问题类型 - 参考 QUICK_REFERENCE.md#错误处理
2. 查找解决方案 - 参考 QUICK_REFERENCE.md#常见问题-faq
3. 修改配置或代码
4. 重新运行测试
```

### 任务：理解一个概念
```
1. 在本索引中搜索概念名称
2. 跳转到相应文档
3. 查看相关代码和注释
4. 尝试实验或修改
```

---

## 📊 文档内容对应表

| 文档文件 | 主要内容 | 读者 | 难度 |
|---------|--------|------|------|
| README.md | 项目简介、安装、基本用法 | 初学者 | ⭐ |
| QUICK_REFERENCE.md | API、场景代码、常见问题 | 初/中级 | ⭐⭐ |
| PROJECT_GUIDE.md | 完整指南、各组件详解 | 中级 | ⭐⭐⭐ |
| ARCHITECTURE_DEEP_DIVE.md | 算法、性能优化、内部实现 | 中/高级 | ⭐⭐⭐⭐ |
| DOCUMENTATION_SUMMARY.md | 文档总结、学习路径 | 所有人 | ⭐ |
| 代码注释 (*.py) | 详细的代码实现说明 | 开发者 | ⭐⭐⭐⭐⭐ |

---

## 🔗 主要文档链接

### 快速查看
- [从这里开始 - QUICK_REFERENCE.md](QUICK_REFERENCE.md)
- [项目完整指南 - PROJECT_GUIDE.md](PROJECT_GUIDE.md)
- [架构深度解析 - ARCHITECTURE_DEEP_DIVE.md](ARCHITECTURE_DEEP_DIVE.md)
- [文档总结 - DOCUMENTATION_SUMMARY.md](DOCUMENTATION_SUMMARY.md)

### 代码查看
- [nanovllm/__init__.py](nanovllm/__init__.py) - 包导出
- [nanovllm/config.py](nanovllm/config.py) - 配置详解
- [nanovllm/sampling_params.py](nanovllm/sampling_params.py) - 采样参数
- [nanovllm/engine/sequence.py](nanovllm/engine/sequence.py) - 序列结构
- [nanovllm/engine/scheduler.py](nanovllm/engine/scheduler.py) - 调度策略
- [nanovllm/engine/llm_engine.py](nanovllm/engine/llm_engine.py) - 引擎实现

### 示例代码
- [example.py](example.py) - 基础使用示例
- [bench.py](bench.py) - 性能基准测试

---

## ❓ 如果我不知道要找什么...

### "我想要 X，应该看什么？"

| 需求 | 文档位置 |
|-----|--------|
| 快速开始 | QUICK_REFERENCE.md |
| API 文档 | QUICK_REFERENCE.md#关键类和方法速查 |
| 代码示例 | QUICK_REFERENCE.md#常见使用场景 |
| 项目架构 | PROJECT_GUIDE.md#项目架构 |
| 工作原理 | ARCHITECTURE_DEEP_DIVE.md#核心数据流 |
| 性能优化 | QUICK_REFERENCE.md#性能优化建议 |
| 错误解决 | QUICK_REFERENCE.md#错误处理 |
| 调度算法 | ARCHITECTURE_DEEP_DIVE.md#调度器 |
| KV 缓存 | ARCHITECTURE_DEEP_DIVE.md#kv缓存管理 |
| 多进程 | ARCHITECTURE_DEEP_DIVE.md#多进程架构 |

---

## 🎯 推荐学习顺序

```
Day 1: 快速开始
├─ 5 min:  README.md
├─ 10 min: QUICK_REFERENCE.md#最快上手
├─ 15 min: 运行 example.py
└─ 10 min: QUICK_REFERENCE.md#关键类和方法速查

Day 2: 理解架构
├─ 30 min: PROJECT_GUIDE.md
├─ 20 min: PROJECT_GUIDE.md#工作流程详解
└─ 15 min: 阅读源代码注释

Day 3: 深入细节
├─ 40 min: ARCHITECTURE_DEEP_DIVE.md#核心数据流
├─ 30 min: ARCHITECTURE_DEEP_DIVE.md#调度器
├─ 20 min: 详细研究 scheduler.py
└─ 30 min: QUICK_REFERENCE.md#场景4

Day 4: 优化实战
├─ 30 min: QUICK_REFERENCE.md#性能优化建议
├─ 30 min: 修改配置进行测试
├─ 30 min: 运行 bench.py 进行基准测试
└─ 30 min: 实验和调优
```

---

## 📞 文档使用建议

✅ **首次使用**: 按照推荐学习顺序学习
✅ **查询 API**: 使用 QUICK_REFERENCE.md 的速查表
✅ **遇到问题**: 查看 QUICK_REFERENCE.md 的故障排除
✅ **深入理解**: 阅读 ARCHITECTURE_DEEP_DIVE.md
✅ **代码修改**: 参考相应文件的详细注释
✅ **性能优化**: 参考 QUICK_REFERENCE.md 的优化建议

---

## 🎓 这些文档的特点

- 📚 **完整** - 从入门到精通的全覆盖
- 🎯 **清晰** - 清晰的结构和导航
- 💡 **实用** - 包含大量代码示例
- 📝 **详细** - 3000+ 行的详细说明
- 🌍 **中文** - 完全中文，易于理解
- 🔗 **互联** - 各文档之间有完整的交叉引用

---

**祝您学习顺利！** 🚀

如有任何问题，请首先在本索引中查找相关链接。
