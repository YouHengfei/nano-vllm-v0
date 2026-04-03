import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:
    """
    LLM推理引擎的核心类。
    
    职责：
    1. 管理模型和分词器的加载
    2. 初始化多进程执行环境(用于张量并行)
    3. 协调scheduler和model_runner完成推理循环
    4. 提供高级别的生成接口(generate方法)
    
    架构流程：
    用户请求 → add_request() → Scheduler → step循环 → 结果返回
    
    与vLLM API兼容，但内部实现更简洁。
    """

    def __init__(self, model: str, **kwargs):
        """
        初始化LLM推理引擎。
        
        参数：
            model: 模型权重目录路径
            **kwargs: 转发给Config的其他参数，例如：
                - tensor_parallel_size: 张量并行数(默认1)
                - max_num_batched_tokens: 批处理最大token数
                - gpu_memory_utilization: GPU显存利用率
                - enforce_eager: 是否禁用torch编译
        """
        
        # ================== 1. 配置管理 ==================
        # 从dataclass的fields中提取所有有效的配置字段名
        # 这是一个优雅的方式来过滤kwargs中属于Config的参数
        config_fields = {field.name for field in fields(Config)}
        
        # 只保留属于Config的kwargs，其他参数会被忽略
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        
        # 创建Config对象，包含所有推理参数
        config = Config(model, **config_kwargs)
        
        # ================== 2. 多进程初始化(张量并行) ==================
        # 列表存储所有 worker 进程(进行张量并行的GPU)
        self.ps = []
        
        # 列表存储进程的同步事件
        self.events = []
        
        # 使用"spawn"方式创建新进程(而不是fork)，确保跨平台兼容性
        # spawn方式会在新进程中重新导入所有模块，避免复杂的状态问题
        ctx = mp.get_context("spawn")
        
        # 如果启用了张量并行(> 1)，为额外的GPU创建worker进程
        # 注意：循环从1开始，rank 0(主进程)不需要单独的进程
        for i in range(1, config.tensor_parallel_size):
            # 创建事件用于进程间同步
            event = ctx.Event()
            
            # 创建worker进程，运行ModelRunner的主循环
            # 每个进程对应一个GPU rank(0-indexed)
            process = ctx.Process(
                target=ModelRunner,  # 进程要执行的类(充当主循环)
                args=(config, i, event)  # 传入配置和该rank的ID
            )
            
            # 启动进程
            process.start()
            
            # 保存进程引用，用于后续清理
            self.ps.append(process)
            
            # 保存事件引用
            self.events.append(event)
        
        # ================== 3. 主进程的ModelRunner ==================
        # 创建主进程(rank 0)的ModelRunner实例
        # 这是一个特殊的对象，它有一个call()方法可以调用model_runner中的函数
        # 为什么主进程不创建独立进程？因为主进程需要控制推理循环
        self.model_runner = ModelRunner(config, 0, self.events)
        
        # ================== 4. 分词器初始化 ==================
        # 从Hugging Face加载分词器(负责文本→token的转换)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model,
            use_fast=True  # 使用快速的Rust实现
        )
        
        # 将EOS token ID保存到config中
        # EOS(End-of-Sequence)是模型生成完成的信号
        config.eos = self.tokenizer.eos_token_id
        
        # ================== 5. 调度器初始化 ==================
        # 创建调度器，负责序列的调度和内存管理
        self.scheduler = Scheduler(config)
        
        # ================== 6. 清理处理 ==================
        # 注册程序退出时的清理函数
        # 这确保当主程序结束时，worker进程被正确终止
        atexit.register(self.exit)

    def exit(self):
        """
        清理函数，在程序退出时调用。
        
        目的：
        1. 向所有worker进程发送退出信号
        2. 等待所有进程完成
        3. 释放资源
        """
        # 向model_runner(rank 0)发送"exit"命令
        # 主进程会处理这个命令并向其他进程发送信号
        self.model_runner.call("exit")
        
        # 删除model_runner对象，释放资源
        del self.model_runner
        
        # 等待所有worker进程完成
        for p in self.ps:
            p.join(timeout=5)

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        """
        添加一个新的推理请求到引擎。
        
        参数：
            prompt: 可以是字符串或token ID列表
                - 字符串: "Hello, world!"
                - token列表: [2, 487, 13, 1234]
            
            sampling_params: 该请求的采样参数
                - temperature: 控制多样性
                - max_tokens: 最多生成多少个token
                - ignore_eos: 是否忽略结束符
        
        流程：
        1. 如果是字符串，使用分词器转换为token ID
        2. 创建Sequence对象
        3. 添加到调度器的等待队列
        
        注意：这个方法只是添加请求，不执行推理。
              实际推理由step()方法循环执行。
        """
        # 如果输入是字符串，使用分词器编码成token ID列表
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        
        # 创建代表该请求的Sequence对象
        seq = Sequence(prompt, sampling_params)
        
        # 添加到调度器的等待队列
        self.scheduler.add(seq)

    def step(self) -> tuple[list[tuple[int, list[int]]], int]:
        """
        执行一个推理步骤。
        
        一个step包括：
        1. 调度器选择要推理的序列
        2. 执行模型前向传播
        3. 对每个序列生成下一个token
        4. 更新序列状态
        
        返回值：
            (outputs, num_tokens):
            - outputs: 已完成序列的列表，每个元素为(seq_id, token_ids)
            - num_tokens: 本次步骤处理的token数(用于性能统计)
              * 正数：prefill阶段处理的token数(较多)
              * 负数：decode阶段，取负值为处理的序列数
        """
        
        # ================== 第1步：调度 ==================
        # 从调度器获取本轮要推理的序列
        # is_prefill表示是prefill还是decode阶段
        seqs, is_prefill = self.scheduler.schedule()
        
        # ================== 第2步：执行推理 ==================
        # 调用model_runner执行模型前向传播
        # 返回为每个序列生成的下一个token ID列表
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        
        # ================== 第3步：后处理 ==================
        # 调度器处理推理结果
        # - 将新token添加到序列
        # - 检查是否完成(EOS或max_tokens)
        self.scheduler.postprocess(seqs, token_ids)
        
        # ================== 第4步：收集完成的序列 ==================
        # 找出本轮所有已完成的序列
        outputs = [
            (seq.seq_id, seq.completion_token_ids)  # (序列ID, 生成的token列表)
            for seq in seqs
            if seq.is_finished
        ]
        
        # ================== 第5步：计算统计信息 ==================
        # 计算本轮处理的token数，用于性能统计
        if is_prefill:
            # Prefill阶段：计算所有序列的总长度(减去缓存的部分)
            num_tokens = sum(len(seq) for seq in seqs)
        else:
            # Decode阶段：只是逐token生成，所以是负的序列数
            # 负数的含义是decode阶段处理的是小批量
            num_tokens = -len(seqs)
        
        return outputs, num_tokens

    def is_finished(self) -> bool:
        """检查是否所有请求都已完成处理。"""
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[dict[str, any]]:
        """
        高级生成接口，与vLLM API兼容。
        
        这是用户主要使用的接口。
        
        参数：
            prompts: 提示词列表，可以是字符串或token ID列表
                例如：["Hello", "World"] 或 [[1, 2, 3], [4, 5, 6]]
            
            sampling_params: 采样参数，可以是单个参数(应用到所有序列)或参数列表
                例如：SamplingParams(temperature=0.6)
                     [SamplingParams(...), SamplingParams(...)]
            
            use_tqdm: 是否显示进度条
        
        返回：
            结果列表，每个元素是一个字典：
            {
                "text": 生成的文本字符串,
                "token_ids": 对应的token ID列表
            }
            
            结果顺序与输入prompts的顺序相同。
        
        算法：
        1. 如果提供了单个sampling_params，复制到所有序列
        2. 为每个prompt创建Sequence并添加到调度器
        3. 推理循环，每次调用step()执行一轮推理
        4. 收集结果，按序列ID排序以保持顺序
        5. 将token ID解码回文本字符串
        """
        
        # ================== 初始化 ==================
        # 如果提供了进度条，创建tqdm进度条
        if use_tqdm:
            pbar = tqdm(
                total=len(prompts),
                desc="Generating",
                dynamic_ncols=True  # 根据终端宽度自适应
            )
        
        # 标准化sampling_params为列表(如果传入单个参数)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        
        # ================== 第1步：添加所有请求 ==================
        # 为每个prompt添加到引擎
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        
        # ================== 第2步：推理循环 ==================
        outputs = {}  # 用seq_id作为key存储结果
        
        # 性能统计变量
        prefill_throughput = decode_throughput = 0.0
        
        # 循环直到所有序列完成
        while not self.is_finished():
            # 记录步长开始时间(用于计算吞吐量)
            t = perf_counter()
            
            # 执行一个推理步骤
            output, num_tokens = self.step()
            
            # 计算吞吐量(tokens/秒)
            elapsed = perf_counter() - t
            if use_tqdm:
                if num_tokens > 0:
                    # Prefill阶段的吞吐量
                    prefill_throughput = num_tokens / elapsed
                else:
                    # Decode阶段的吞吐量
                    decode_throughput = -num_tokens / elapsed
                
                # 更新进度条显示
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            
            # 收集本轮完成的序列结果
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)  # 更新进度条
        
        # ================== 第3步：结果整理 ==================
        # 按序列ID排序，确保结果顺序与输入顺序一致
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        
        # 将token ID解码为文本字符串
        outputs = [
            {
                "text": self.tokenizer.decode(token_ids),
                "token_ids": token_ids
            }
            for token_ids in outputs
        ]
        
        # 关闭进度条
        if use_tqdm:
            pbar.close()
        
        return outputs
