# embedding_server.py - 显存优化版本
import asyncio
import logging
import time
import gc
from typing import List, Dict, Any, Optional
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModel
from concurrent.futures import ThreadPoolExecutor
import psutil
import threading

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingRequest(BaseModel):
    texts: List[str] = Field(..., description="要嵌入的文本列表")
    instruction: str = Field(
        default="Given a web search query, retrieve relevant passages that answer the query",
        description="任务指令"
    )
    normalize: bool = Field(default=True, description="是否标准化向量")
    batch_size: int = Field(default=8, description="批处理大小")  # 减小默认批次

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    usage: Dict[str, Any]
    model: str
    processing_time: float

class OptimizedEmbeddingServer:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",  # 默认使用小模型
        device: str = "auto",
        max_length: int = 512,  # 减少最大长度
        precision: str = "fp16",  # 使用半精度
        enable_gradient_checkpointing: bool = True,
        max_batch_size: int = 16,  # 最大批次限制
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.precision = precision
        self.max_batch_size = max_batch_size

        # 设备选择
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"使用设备: {self.device}")
        logger.info(f"模型: {model_name}")

        # 显存监控
        self._setup_memory_monitoring()

        # 初始化模型和tokenizer
        self._load_model()

        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=2)

        # 显存清理锁
        self._cleanup_lock = threading.Lock()

    def _setup_memory_monitoring(self):
        """设置显存监控"""
        if torch.cuda.is_available():
            # 设置显存分配策略
            torch.cuda.set_per_process_memory_fraction(0.85)
            torch.cuda.empty_cache()

            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU总显存: {total_memory:.2f}GB")

    def _load_model(self):
        """加载模型和tokenizer"""
        logger.info("正在加载模型...")

        try:
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side='right'
            )

            # 设置pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # 模型加载配置
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.precision == "fp16" else torch.float32,
                "device_map": "auto" if self.device == "cuda" else None,
                "low_cpu_mem_usage": True,
            }

            # 对于小显存，强制使用CPU offload
            if torch.cuda.is_available():
                available_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                if available_memory < 6:  # 6GB以下显存
                    logger.warning("检测到小显存，启用CPU offload")
                    model_kwargs["device_map"] = {"": "cpu"}

            self.model = AutoModel.from_pretrained(self.model_name, **model_kwargs)

            # 如果在GPU上，启用梯度检查点
            if self.device == "cuda" and hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()

            # 设置为评估模式
            self.model.eval()

            logger.info("模型加载完成")
            self._log_memory_usage()

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise

    def _log_memory_usage(self):
        """记录内存使用情况"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"GPU显存使用: {allocated:.2f}GB (已分配) / {reserved:.2f}GB (已保留)")

        # 系统内存
        memory = psutil.virtual_memory()
        logger.info(f"系统内存使用: {memory.percent}% ({memory.used / 1024**3:.2f}GB / {memory.total / 1024**3:.2f}GB)")

    def _aggressive_cleanup(self):
        """激进的内存清理"""
        with self._cleanup_lock:
            # 清理Python垃圾
            gc.collect()

            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # 强制内存整理（如果可用）
            try:
                import ctypes
                libc = ctypes.CDLL("libc.so.6")
                libc.malloc_trim(0)
            except:
                pass

    def format_texts_with_instruction(self, texts: List[str], instruction: str) -> List[str]:
        """格式化文本"""
        return [f'Instruct: {instruction}\nQuery: {text}' for text in texts]

    def _process_batch_safe(self, texts: List[str]) -> List[List[float]]:
        """安全的批次处理，包含显存监控"""
        batch_size = len(texts)

        # 动态调整批次大小
        if torch.cuda.is_available():
            available_memory = torch.cuda.memory_available() / 1024**3
            if available_memory < 1.0:  # 可用显存少于1GB
                # 强制减小批次
                if batch_size > 4:
                    logger.warning(f"显存不足，将批次从 {batch_size} 减少到 4")
                    # 分割批次递归处理
                    mid = len(texts) // 2
                    batch1 = self._process_batch_safe(texts[:mid])
                    batch2 = self._process_batch_safe(texts[mid:])
                    return batch1 + batch2

        try:
            # Tokenize
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )

            # 移动到设备
            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # 推理
            with torch.no_grad():
                if self.precision == "fp16" and self.device == "cuda":
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**inputs)
                else:
                    outputs = self.model(**inputs)

                # 提取embeddings
                embeddings = outputs.last_hidden_state

                # 池化 - 使用注意力掩码加权平均
                attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(embeddings.size()).float()
                masked_embeddings = embeddings * attention_mask
                summed = torch.sum(masked_embeddings, 1)
                summed_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
                mean_pooled = summed / summed_mask

                # 转换为CPU并返回
                result = mean_pooled.cpu().numpy().tolist()

            # 立即清理
            del inputs, outputs, embeddings, attention_mask, masked_embeddings
            if self.device == "cuda":
                torch.cuda.empty_cache()

            return result

        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"显存不足: {e}")
            # 强制清理并重试更小的批次
            self._aggressive_cleanup()

            if batch_size > 1:
                logger.info(f"重试更小批次: {batch_size} -> {batch_size//2}")
                mid = len(texts) // 2
                batch1 = self._process_batch_safe(texts[:mid])
                batch2 = self._process_batch_safe(texts[mid:])
                return batch1 + batch2
            else:
                raise HTTPException(status_code=500, detail="显存不足，无法处理请求")

    async def generate_embeddings_batch(
        self,
        texts: List[str],
        instruction: str,
        normalize: bool = True,
        batch_size: int = 8
    ) -> tuple[List[List[float]], Dict[str, Any]]:
        """批量生成embeddings"""
        start_time = time.time()
        total_texts = len(texts)

        # 限制批次大小
        batch_size = min(batch_size, self.max_batch_size)

        # 格式化文本
        formatted_texts = self.format_texts_with_instruction(texts, instruction)

        # 分批处理
        all_embeddings = []
        total_tokens = 0

        for i in range(0, len(formatted_texts), batch_size):
            batch_texts = formatted_texts[i:i+batch_size]

            # 在线程池中执行以避免阻塞
            loop = asyncio.get_event_loop()
            batch_embeddings = await loop.run_in_executor(
                self.executor,
                self._process_batch_safe,
                batch_texts
            )

            all_embeddings.extend(batch_embeddings)
            total_tokens += sum(len(self.tokenizer.tokenize(text)) for text in batch_texts)

            # 进度日志和内存清理
            batch_num = i // batch_size + 1
            total_batches = (len(formatted_texts) - 1) // batch_size + 1
            logger.info(f"完成批次 {batch_num}/{total_batches}")

            # 每隔几个批次清理一次
            if batch_num % 3 == 0:
                self._aggressive_cleanup()

        # 标准化
        if normalize:
            loop = asyncio.get_event_loop()
            all_embeddings = await loop.run_in_executor(
                self.executor,
                self._normalize_embeddings,
                all_embeddings
            )

        processing_time = time.time() - start_time

        # 最终清理
        self._aggressive_cleanup()

        usage = {
            "total_texts": total_texts,
            "total_tokens": total_tokens,
            "processing_time": processing_time,
            "texts_per_second": total_texts / processing_time if processing_time > 0 else 0,
            "batch_size_used": batch_size,
            "total_batches": total_batches
        }

        return all_embeddings, usage

    def _normalize_embeddings(self, embeddings: List[List[float]]) -> List[List[float]]:
        """标准化embeddings"""
        embeddings_array = np.array(embeddings)
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # 避免除零
        normalized = embeddings_array / norms
        return normalized.tolist()

# 创建FastAPI应用
app = FastAPI(title="Qwen3 Embedding API - Memory Optimized", version="1.0.0")

# 全局模型实例
embedding_server = None

@app.on_event("startup")
async def startup_event():
    global embedding_server
    logger.info("启动内存优化版Embedding服务器...")

    # 根据可用显存选择模型
    model_name = "Qwen/Qwen3-Embedding-0.6B"  # 默认小模型

    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if total_memory >= 8:
            model_name = "Qwen/Qwen3-Embedding-4B"
        elif total_memory >= 16:
            model_name = "Qwen/Qwen3-Embedding-8B"

        logger.info(f"检测到 {total_memory:.1f}GB 显存，选择模型: {model_name}")

    embedding_server = OptimizedEmbeddingServer(
        model_name=model_name,
        max_batch_size=8,  # 保守的批次大小
        precision="fp16"
    )

@app.on_event("shutdown")
async def shutdown_event():
    global embedding_server
    if embedding_server:
        # 清理资源
        del embedding_server.model
        del embedding_server.tokenizer
        embedding_server._aggressive_cleanup()

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """创建embeddings - 内存优化版"""
    if not embedding_server:
        raise HTTPException(status_code=503, detail="模型未加载")

    if not request.texts:
        raise HTTPException(status_code=400, detail="文本列表不能为空")

    # 限制单次请求大小
    max_texts = 100
    if len(request.texts) > max_texts:
        raise HTTPException(
            status_code=400,
            detail=f"单次请求文本数量不能超过{max_texts}，当前: {len(request.texts)}"
        )

    try:
        # 动态调整批次大小
        batch_size = min(request.batch_size, 8)
        if torch.cuda.is_available():
            available_memory = torch.cuda.memory_available() / 1024**3
            if available_memory < 2:
                batch_size = min(batch_size, 4)
            elif available_memory < 1:
                batch_size = min(batch_size, 2)

        embeddings, usage = await embedding_server.generate_embeddings_batch(
            texts=request.texts,
            instruction=request.instruction,
            normalize=request.normalize,
            batch_size=batch_size
        )

        return EmbeddingResponse(
            embeddings=embeddings,
            usage=usage,
            model=embedding_server.model_name,
            processing_time=usage["processing_time"]
        )

    except Exception as e:
        logger.error(f"生成embeddings时出错: {str(e)}")
        # 尝试强制清理
        if embedding_server:
            embedding_server._aggressive_cleanup()
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

@app.get("/health")
async def health_check():
    """健康检查"""
    health_info = {
        "status": "healthy",
        "model_loaded": embedding_server is not None,
        "gpu_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        health_info.update({
            "gpu_count": torch.cuda.device_count(),
            "gpu_memory_allocated": f"{torch.cuda.memory_allocated() / 1024**3:.2f}GB",
            "gpu_memory_available": f"{torch.cuda.memory_available() / 1024**3:.2f}GB"
        })

    return health_info

@app.get("/memory/cleanup")
async def manual_cleanup():
    """手动内存清理"""
    if embedding_server:
        embedding_server._aggressive_cleanup()
        return {"message": "内存清理完成"}
    return {"message": "服务未启动"}

@app.get("/memory/stats")
async def memory_stats():
    """内存统计"""
    stats = {}

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3

            stats[f"gpu_{i}"] = {
                "allocated_gb": round(allocated, 2),
                "reserved_gb": round(reserved, 2),
                "total_gb": round(total, 2),
                "utilization_percent": round(allocated / total * 100, 2)
            }

    # 系统内存
    memory = psutil.virtual_memory()
    stats["system_memory"] = {
        "used_gb": round(memory.used / 1024**3, 2),
        "total_gb": round(memory.total / 1024**3, 2),
        "percent": memory.percent
    }

    return stats
