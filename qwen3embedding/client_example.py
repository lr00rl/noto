import requests
import time
import asyncio
import aiohttp
from typing import List

class EmbeddingClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    def create_embeddings_sync(
        self,
        texts: List[str],
        instruction: str = "Given a web search query, retrieve relevant passages",
        batch_size: int = 32
    ):
        '''同步请求embeddings'''
        response = requests.post(
            f"{self.base_url}/v1/embeddings",
            json={
                "texts": texts,
                "instruction": instruction,
                "batch_size": batch_size,
                "normalize": True
            },
            timeout=300  # 5分钟超时
        )

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"请求失败: {response.status_code}, {response.text}")

    async def create_embeddings_async(
        self,
        texts: List[str],
        instruction: str = "Given a web search query, retrieve relevant passages",
        batch_size: int = 32
    ):
        '''异步请求embeddings'''
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/v1/embeddings",
                json={
                    "texts": texts,
                    "instruction": instruction,
                    "batch_size": batch_size,
                    "normalize": True
                },
                timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    text = await response.text()
                    raise Exception(f"请求失败: {response.status}, {text}")

# 使用示例
if __name__ == "__main__":
    client = EmbeddingClient()

    # 测试数据
    test_texts = [
        "What is machine learning?",
        "How does neural network work?",
        "Explain deep learning concepts",
        # ... 更多文本
    ] * 10  # 100个文本用于测试

    print(f"测试 {len(test_texts)} 个文本的embedding生成...")

    start_time = time.time()
    result = client.create_embeddings_sync(test_texts)
    end_time = time.time()

    print(f"处理完成!")
    print(f"总耗时: {end_time - start_time:.2f}秒")
    print(f"处理速度: {result['usage']['texts_per_second']:.2f} texts/sec")
    print(f"Embedding维度: {len(result['embeddings'][0])}")
