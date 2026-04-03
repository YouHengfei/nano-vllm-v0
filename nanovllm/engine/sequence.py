from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    """
    序列的生命周期状态枚举。
    
    状态转移图：
    WAITING → RUNNING → FINISHED
       ↑        ↓
       └─────────┘
    (抢占机制在内存不足时会将RUNNING状态的序列
     降级回WAITING以释放内存)
    """
    WAITING = auto()   # 等待中：在队列中等待被调度推理
    RUNNING = auto()   # 运行中：正在进行推理
    FINISHED = auto()  # 已完成：生成完毕，可以返回结果


class Sequence:
    """
    代表单个用户请求的序列数据结构。
    
    一个Sequence包含：
    - 完整的token序列(输入提示词 + 生成的tokens)
    - 状态信息(WAITING/RUNNING/FINISHED)
    - KV缓存位置信息
    - 采样参数
    
    这个类在推理过程中被scheduler和model_runner频繁使用。
    """
    
    block_size = 256
    """KV缓存块的大小，与Config.kvcache_block_size必须保持一致。
    将token序列分成固定大小的块便于内存管理。"""
    
    counter = count()
    """全局的序列ID计数器，使用Python的itertools.count()实现无限递增。
    保证每个Sequence实例都有一个唯一的ID，用于结果匹配和追踪。"""

    def __init__(self, token_ids: list[int], sampling_params: SamplingParams = SamplingParams()):
        """
        初始化一个新的推理序列。
        
        参数：
            token_ids: 输入提示词的token ID列表，例如 [2, 487, 13, ...]
            sampling_params: 采样参数(temperature, max_tokens等)
        """
        # 为这个序列分配全局唯一ID，用于在并发环境中追踪序列
        self.seq_id = next(Sequence.counter)
        
        # 初始状态为等待，等待被调度器选中进行推理
        self.status = SequenceStatus.WAITING
        
        # ================== Token 管理 ==================
        # 深复制输入的token，避免被外部修改
        self.token_ids = copy(token_ids)
        
        # 最后一个token ID，用于在生成时快速获取前一个token
        self.last_token = token_ids[-1]
        
        # 序列中的总token数(提示词 + 已生成的token)
        self.num_tokens = len(self.token_ids)
        
        # 提示词的长度，用于区分哪些token是原始输入哪些是生成的
        self.num_prompt_tokens = len(token_ids)
        
        # ================== KV缓存管理 ==================
        # 已缓存的token数量，用于增量缓存优化
        # 例如：序列长度为100，但num_cached_tokens可能只有50(因为前50个token已缓存)
        self.num_cached_tokens = 0
        
        # KV缓存块表：存储该序列在block manager中的块位置
        # 例如 [0, 1, 3, 5] 表示该序列使用了第0,1,3,5号块
        self.block_table = []
        
        # ================== 采样参数 ==================
        # 直接存储采样参数以供推理时使用
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        """支持len()操作，返回序列的总token数。"""
        return self.num_tokens

    def __getitem__(self, key):
        """支持索引操作，访问特定位置的token。例如：seq[0]返回第一个token。"""
        return self.token_ids[key]

    # ================== 属性访问器 ==================
    # 使用@property装饰器将方法转换为属性，
    # 这样访问时看起来像访问属性而不是调用方法，提高代码可读性。

    @property
    def is_finished(self) -> bool:
        """序列是否已完成。布尔值，便于在循环中判断。"""
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self) -> int:
        """已生成的token数量(不包括输入提示词)。
        例如：总长度100，提示词30，则num_completion_tokens=70"""
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self) -> list[int]:
        """获取原始输入提示词的token ID列表。
        这些token不会在推理过程中改变。"""
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self) -> list[int]:
        """获取已生成的token ID列表(不包括输入)。
        这是要返回给用户的结果。"""
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self) -> int:
        """已计算并缓存的KV块数量。
        与num_blocks相比，这个值可能更小(增量缓存优化)。"""
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self) -> int:
        """该序列需要的总KV缓存块数量(向上取整)。
        计算：(num_tokens + block_size - 1) // block_size
        例如：num_tokens=300, block_size=256 → num_blocks=2"""
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self) -> int:
        """最后一个块中实际包含的token数量(可能不满块)。
        例如：num_tokens=300, block_size=256 → last_block_num_tokens=44"""
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i: int) -> list[int]:
        """
        获取第i个块中的token ID列表。
        
        参数：
            i: 块索引(从0开始)
        
        返回：
            该块中的token ID列表(可能少于block_size个，对最后一块)
        
        例如：block(0)返回[token_0, token_1, ..., token_255]
             block(1)返回[token_256, token_257, ..., token_299]
        """
        assert 0 <= i < self.num_blocks, f"块索引{i}超出范围[0, {self.num_blocks})"
        return self.token_ids[i * self.block_size : (i + 1) * self.block_size]

    def append_token(self, token_id: int) -> None:
        """
        在序列末尾添加一个新生成的token。
        这个方法在每个decode步骤后被调用一次。
        
        参数：
            token_id: 新生成的token ID
        """
        # 追加token到列表
        self.token_ids.append(token_id)
        
        # 更新最后一个token的缓存(避免频繁索引)
        self.last_token = token_id
        
        # 增加token计数
        self.num_tokens += 1

    # ================== 序列化支持 (进程间通信) ==================
    # 当使用多进程或序列化/反序列化时调用这些方法。
    # 目的是减少进程间通信的数据量。

    def __getstate__(self) -> tuple:
        """
        序列化方法：在将Sequence发送到其他进程前调用。
        
        优化：不序列化完整的token_ids列表，而是优化存储：
        - 如果还未生成任何token(num_completion_tokens==0)：保存整个token_ids
        - 如果已生成token：只保存最后一个token(last_token)
          其他生成过的token可以从结果中获取，不需要往回传
        
        返回：
            包含需要序列化的信息的元组
        """
        return (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.block_table,
            self.token_ids if self.num_completion_tokens == 0 else self.last_token
        )

    def __setstate__(self, state: tuple) -> None:
        """
        反序列化方法：从序列化数据恢复Sequence对象。
        与__getstate__对应，恢复被序列化的状态。
        
        参数：
            state: 通过__getstate__得到的序列化数据
        """
        # 恢复主要属性
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        
        # 根据是否有生成过token来恢复最后的数据
        if self.num_completion_tokens == 0:
            # 没有生成token，恢复完整的token_ids
            self.token_ids = state[-1]
        else:
            # 已生成token，只恢复最后一个token
            self.last_token = state[-1]
