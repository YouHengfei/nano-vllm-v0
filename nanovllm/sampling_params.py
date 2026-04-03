from dataclasses import dataclass


@dataclass
class SamplingParams:
    """
    Token采样的参数配置类。
    
    在LLM生成过程中，不是简单地选择概率最高的token(贪心)，
    而是根据这些参数从概率分布中采样，产生多样化的结果。
    """
    
    temperature: float = 1.0
    """
    采样温度，控制输出的随机性和多样性。
    
    工作原理：对logits应用softmax(logits / temperature)
    - temperature < 1.0: 分布更尖锐，低概率token被压低，生成更确定性的文本
    - temperature = 1.0: 保持原始概率分布(默认)
    - temperature > 1.0: 分布更平坦，所有token概率接近，生成更随机/多样的文本
    
    典型值：
    - 创意写作: 0.8-1.2
    - 事实性任务: 0.6-0.8
    """
    
    max_tokens: int = 64
    """
    最多生成的token数量(不包括输入提示词)。
    
    例如：
    - 输入: "Tell me about AI" (8 tokens)
    - max_tokens: 128
    - 最终输出: 8 + 128 = 最多136个tokens
    """
    
    ignore_eos: bool = False
    """
    是否忽略EOS(End-of-Sequence)token。
    
    - False(默认): 当模型生成EOS token时停止生成(正常行为)
    - True: 即使生成EOS也继续生成，直到达到max_tokens
    
    用途：在某些任务中，模型在完成前可能多次生成EOS，
    设置为True可以强制继续生成。
    """

    def __post_init__(self):
        """
        dataclass初始化后的验证逻辑。
        
        确保temperature足够大，避免贪心采样。
        注意：这个项目不支持完全确定性的生成(temperature=0)，
        原因是确定性推理在批量处理和缓存等场景下会带来问题。
        """
        assert self.temperature > 1e-10, "贪心采样(temperature=0)暂不支持，temperature必须 > 1e-10"
