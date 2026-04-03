import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    """
    LLM推理引擎的全局配置类。
    
    该类使用Python的dataclass装饰器，定义了所有推理时需要的超参数和配置项。
    通过dataclass可以自动生成__init__等特殊方法，提高代码可读性。
    """
    
    # ================== 必需参数 ==================
    model: str  # 模型权重路径，应该是一个包含config.json和权重文件的目录
    
    # ================== 推理性能相关 ==================
    max_num_batched_tokens: int = 16384
    """单个批次中最多处理的token数。这是内存和吞吐量的权衡点。
    例如：max_num_batched_tokens=16384意味着可以同时处理16384个token
    这包括前缀词(prompt tokens)和已生成的token"""
    
    max_num_seqs: int = 512
    """并发处理的最大序列数(最多同时推理的请求数)。
    增加这个值可以提高吞吐量但会消耗更多显存。"""
    
    max_model_len: int = 4096
    """模型支持的最大上下文长度(context length)。
    包括提示词长度 + 生成长度。会被自动clip到模型的max_position_embeddings。"""
    
    gpu_memory_utilization: float = 0.9
    """目标GPU显存利用率(0.0-1.0)。
    用于自动计算max_num_kvcache_blocks时的目标值。"""
    
    # ================== 分布式推理相关 ==================
    tensor_parallel_size: int = 1
    """张量并行的大小，即使用的GPU数量(1-8)。
    >1时会启用多卡张量并行，每张卡上的张量被分割存储和计算。
    例如：tensor_parallel_size=2表示在2张GPU上并行推理"""
    
    enforce_eager: bool = False
    """是否禁用PyTorch的即时编译(torch.compile)。
    设置为True可以加快启动速度但降低推理性能;通常用于调试。"""
    
    # ================== 内部状态和派生参数 ==================
    hf_config: AutoConfig | None = None
    """从Hugging Face模型自动加载的模型配置对象。
    包含隐藏层大小、层数、词表大小等信息。在__post_init__中初始化。"""
    
    eos: int = -1
    """序列结束符(End-of-Sequence token)的ID。
    初始值为-1，会在LLMEngine.__init__中被设置为分词器的eos_token_id"""
    
    # ================== KV缓存相关 ==================
    kvcache_block_size: int = 256
    """KV缓存块的大小，必须是256的倍数。
    KV缓存被分成固定大小的块进行管理，便于动态分配和内存碎片管理。
    block_size=256意味着256个token对应的KV值为一个块单元。"""
    
    num_kvcache_blocks: int = -1
    """总的KV缓存块数量。-1表示根据gpu_memory_utilization自动计算。
    所需的总显存大约为: hidden_size * num_layers * kvcache_block_size * num_kvcache_blocks * 2"""

    def __post_init__(self):
        """初始化后的验证和补充逻辑。dataclass会在__init__后自动调用此方法。"""
        
        # 验证模型路径是否有效
        assert os.path.isdir(self.model), f"模型路径不存在或不是目录: {self.model}"
        
        # 验证KV缓存块大小必须是256的倍数(GPU内存对齐要求)
        assert self.kvcache_block_size % 256 == 0, "kvcache_block_size必须是256的倍数"
        
        # 验证张量并行大小在合理范围内
        assert 1 <= self.tensor_parallel_size <= 8, "tensor_parallel_size必须在1-8之间"
        
        # 从Hugging Face自动加载模型配置
        # 这会下载或从本地缓存加载config.json
        self.hf_config = AutoConfig.from_pretrained(self.model)
        
        # 确保配置的max_model_len不超过模型本身支持的最大长度
        # max_position_embeddings是模型架构中定义的最大位置编码长度
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        
        # 最后验证：单批最大token数不能小于模型最大长度
        # 否则即使单个序列也无法处理
        assert self.max_num_batched_tokens >= self.max_model_len, \
            f"max_num_batched_tokens({self.max_num_batched_tokens}) 必须 >= max_model_len({self.max_model_len})"
