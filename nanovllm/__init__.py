"""
Nano-vLLM: 轻量级LLM推理引擎

这个包提供了两个主要的公共接口供用户使用：

1. LLM - 推理引擎主类
   初始化模型并提供generate()接口进行文本生成
   
2. SamplingParams - 采样参数配置类
   定义生成过程中的参数(温度、最大token数等)

使用示例：
─────────
from nanovllm import LLM, SamplingParams

# 初始化
llm = LLM(model="/path/to/model", tensor_parallel_size=1)

# 配置采样参数
params = SamplingParams(temperature=0.8, max_tokens=512)

# 生成文本
results = llm.generate(
    prompts=["Hello, world!", "What is machine learning?"],
    sampling_params=params,
)

# 获取输出
for result in results:
    print(result["text"])


内部模块结构：
────────────
- llm.py → LLM类(公共接口)
- sampling_params.py → SamplingParams类(采样参数)
- config.py → Config类(内部配置)
- engine/
  - llm_engine.py → 核心推理引擎
  - scheduler.py → 序列调度器
  - sequence.py → 单个序列的表示
  - block_manager.py → KV缓存管理
  - model_runner.py → 模型执行器
- layers/ → 神经网络层实现
- models/ → 具体模型实现(如Qwen)
- utils/ → 工具函数
"""

from nanovllm.llm import LLM
from nanovllm.sampling_params import SamplingParams

__all__ = ["LLM", "SamplingParams"]
