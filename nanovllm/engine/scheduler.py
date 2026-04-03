from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:
    """
    LLM推理的调度器，核心职责是：
    
    1. 管理序列队列（waiting和running）
    2. 决定哪些序列在本轮被处理（prefill还是decode）
    3. 协调KV缓存的分配和回收
    4. 在内存不足时进行序列抢占
    5. 处理完成序列的清理
    
    核心理念：
    - Prefill优先级高于Decode（新序列优先处理）
    - 每轮选择尽可能多的序列（最多max_num_seqs个）
    - 不超过max_num_batched_tokens的显存限制
    - 内存不足时进行低优先级序列抢占
    """

    def __init__(self, config: Config):
        """
        初始化调度器。
        
        参数：
            config: 包含调度相关参数的配置对象
        """
        # 并发处理的最大序列数
        self.max_num_seqs = config.max_num_seqs
        
        # 单批最大token数
        self.max_num_batched_tokens = config.max_num_batched_tokens
        
        # EOS(End-of-Sequence) token ID，用于判断序列完成
        self.eos = config.eos
        
        # KV缓存块管理器，负责显存的动态分配
        self.block_manager = BlockManager(
            config.num_kvcache_blocks,  # 总块数
            config.kvcache_block_size   # 每块的大小
        )
        
        # 等待队列：新添加的序列先放这里，等待被调度
        # 使用deque(双端队列)是因为支持高效的appendleft操作
        self.waiting: deque[Sequence] = deque()
        
        # 运行队列：正在进行推理的序列
        self.running: deque[Sequence] = deque()

    def is_finished(self) -> bool:
        """
        检查是否所有序列都已完成。
        
        返回True当且仅当：
        - 等待队列为空 AND
        - 运行队列为空
        """
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        """
        添加一个新序列到调度器。
        
        新序列会被放入等待队列，等待被schedule()选中。
        
        参数：
            seq: 新的Sequence对象
        """
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """
        调度函数，决定本轮要处理哪些序列。
        
        这是调度器最核心的函数，每个推理步骤调用一次。
        
        调度策略：
        1. 优先处理Prefill（新序列的初始计算）
        2. 如果Prefill为空，处理Decode（逐token生成）
        3. 每轮选择尽可能多的序列（受max_num_seqs和max_num_batched_tokens限制）
        4. 内存不足时进行抢占
        
        返回值：
            (scheduled_seqs, is_prefill):
            - scheduled_seqs: 本轮要处理的序列列表
            - is_prefill: 是Prefill(True)还是Decode(False)阶段
        
        关键设计：
        - Prefill和Decode互斥，不会在同一轮混合处理
        - 这样更利于GPU的批处理优化
        """
        
        # ================== 第1阶段：Prefill处理 ==================
        # Prefill是处理新序列的输入提示词，生成初始KV缓存
        
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        
        # 循环从等待队列中选择序列进行prefill
        while self.waiting and num_seqs < self.max_num_seqs:
            # 获取等待队列的第一个序列(FIFO)
            seq = self.waiting[0]
            
            # 检查两个约束：
            # 1. Token数不能超过限制
            # 2. 的显存（KV缓存块）能否分配
            if (num_batched_tokens + len(seq) > self.max_num_batched_tokens or
                not self.block_manager.can_allocate(seq)):
                # 无法添加这个序列，停止选择
                break
            
            # ===== 能够添加这个序列 =====
            num_seqs += 1
            
            # 为该序列分配KV缓存块
            self.block_manager.allocate(seq)
            
            # 计算该序列贡献的token数
            # 注意：len(seq)是包括可能已缓存部分的总长度
            # len(seq) - seq.num_cached_tokens 是需要新计算的部分
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            
            # 标记序列状态为运行中
            seq.status = SequenceStatus.RUNNING
            
            # 从等待队列移出
            self.waiting.popleft()
            
            # 加入运行队列
            self.running.append(seq)
            
            # 加入本轮的调度列表
            scheduled_seqs.append(seq)
        
        # 如果找到了prefill序列，返回它们
        # 这样可以保证prefill优先级高于decode
        if scheduled_seqs:
            return scheduled_seqs, True
        
        # ================== 第2阶段：Decode处理 ==================
        # Decode是为已有KV缓存的序列生成下一个token
        
        # 从运行队列中选择序列进行decode
        while self.running and num_seqs < self.max_num_seqs:
            # 从运行队列取出一个序列
            seq = self.running.popleft()
            
            # Decode仅需要为当前最后一个token生成下一个，
            # 所以内存需求相对小。但仍需检查KV缓存是否能扩展
            
            # 循环处理：直到能为该序列分配内存
            while not self.block_manager.can_append(seq):
                # 无法扩展这个序列的缓存，需要进行抢占
                
                if self.running:
                    # 运行队列还有其他序列，抢占优先级最低的(最后一个)
                    self.preempt(self.running.pop())
                else:
                    # 运行队列已空，抢占当前序列本身
                    # 这会将该序列降级回WAITING状态
                    self.preempt(seq)
                    break
            else:
                # while-else: 如果while正常结束(not进入break)
                # 表示成功为序列分配了内存
                
                num_seqs += 1
                
                # 通知block_manager该序列可能需要追加缓存
                self.block_manager.may_append(seq)
                
                # 加入本轮的调度列表
                scheduled_seqs.append(seq)
        
        # 最后将处理过的decode序列放回运行队列
        # extendleft反向放入，保持相反的顺序
        self.running.extendleft(reversed(scheduled_seqs))
        
        # 确保至少有一个序列被调度！
        # 这是一个辅助的冗余检查
        assert scheduled_seqs, "无法调度任何序列，可能内存配置不当"
        
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        """
        抢占一个运行中的序列，将其降级回等待状态。
        
        目的：在内存不足时释放该序列的KV缓存。
        
        参数：
            seq: 要抢占的序列
        
        副作用：
        - 序列状态改为WAITING
        - 释放该序列的所有KV缓存块
        - 序列被移到等待队列的前面(高优先级重试)
        
        注意：
        - 序列的token_ids数据不会被清除，只是KV缓存被释放
        - 下次被选中时会重新计算KV缓存
        """
        # 标记状态为等待
        seq.status = SequenceStatus.WAITING
        
        # 释放该序列占用的所有KV缓存块
        self.block_manager.deallocate(seq)
        
        # 加入等待队列的最前面(高优先级)
        # appendleft确保下次有机会时会被重新选中
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]):
        """
        推理后的处理：更新序列状态，处理完成的序列。
        
        调用时机：每个推理步骤后，model_runner执行后调用。
        
        参数：
            seqs: 本轮被处理的所有序列
            token_ids: 为每个序列生成的下一个token ID列表
        
        处理流程：
        1. 为每个序列添加新生成的token
        2. 检查序列是否完成（EOS或达到max_tokens）
        3. 完成的序列清理资源并移出运行队列
        """
        
        # 逐个处理每个序列和它生成的token
        for seq, token_id in zip(seqs, token_ids):
            # 将新生成的token添加到序列
            seq.append_token(token_id)
            
            # 检查该序列是否应该完成
            # 完成条件：
            # 1. 生成了EOS且未设置ignore_eos，或
            # 2. 达到了max_tokens限制
            should_finish = (
                (not seq.ignore_eos and token_id == self.eos) or
                seq.num_completion_tokens == seq.max_tokens
            )
            
            if should_finish:
                # 标记序列完成
                seq.status = SequenceStatus.FINISHED
                
                # 释放该序列的KV缓存
                self.block_manager.deallocate(seq)
                
                # 从运行队列移出
                self.running.remove(seq)
                
                # 完成的序列会在LLMEngine.step()中被收集并返回给用户
