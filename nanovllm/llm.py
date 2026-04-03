from nanovllm.engine.llm_engine import LLMEngine


class LLM(LLMEngine):
    """
    LLM推理的公共接口类。
    
    这是一个简单的包装类，直接继承自LLMEngine。
    目的是为用户提供友好的命名（LLM而不是LLMEngine），
    使其更符合直观的使用方式。
    
    用法示例：
    ──────────
    from nanovllm import LLM, SamplingParams
    
    # 初始化模型
    llm = LLM("/path/to/model")
    
    # 定义采样参数
    params = SamplingParams(temperature=0.6, max_tokens=256)
    
    # 生成文本
    outputs = llm.generate(
        ["Hello, world!", "What is AI?"],
        params
    )
    
    # 处理输出
    for output in outputs:
        print(output["text"])
    
    
    设计模式：
    ────────
    这是一个典型的"外观模式(Facade)"，通过简单的继承
    为底层复杂的LLMEngine提供一个简化的接口。
    
    内部实现完全来自LLMEngine：
    - __init__() → LLMEngine.__init__()
    - generate() → LLMEngine.generate()
    - add_request() → LLMEngine.add_request()
    - step() → LLMEngine.step()
    等等
    
    
    与vLLM的兼容性：
    ────────────
    Nano-vLLM的LLM类设计与官方vLLM的LLM类API一致，
    使得代码可以相对容易地在两者间切换。
    
    对标的vLLM实现：
    from vllm import LLM, SamplingParams
    llm = LLM(model="meta-llama/Llama-2-7b-hf")
    output = llm.generate(["Hello"], SamplingParams())
    
    
    性能注意事项：
    ────────────
    1. 这个类本身不消耗额外资源（纯继承）
    2. 所有实际工作由LLMEngine完成
    3. 适合生产环境使用
    ""
    """
    pass  # 所有功能由LLMEngine提供，无需额外实现
