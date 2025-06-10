"""
测试embedding服务的轻量级客户端
用于验证服务是否正常工作
"""

import requests
import time
import json
from typing import List

class EmbeddingClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')

    def health_check(self) -> dict:
        """健康检查"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            return {"error": str(e)}

    def get_memory_stats(self) -> dict:
        """获取内存统计"""
        try:
            response = requests.get(f"{self.base_url}/memory/stats", timeout=10)
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            return {"error": str(e)}

    def embed_texts(self, texts: List[str], **kwargs) -> dict:
        """生成embeddings"""
        payload = {
            "texts": texts,
            "instruction": kwargs.get("instruction", "Given a web search query, retrieve relevant passages that answer the query"),
            "normalize": kwargs.get("normalize", True),
            "batch_size": kwargs.get("batch_size", 4)  # 默认小批次
        }

        try:
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/v1/embeddings",
                json=payload,
                timeout=60
            )
            end_time = time.time()

            if response.status_code == 200:
                result = response.json()
                result["client_processing_time"] = end_time - start_time
                return result
            else:
                return {"error": f"HTTP {response.status_code}: {response.text}"}
        except Exception as e:
            return {"error": str(e)}

    def test_progressive_load(self):
        """渐进式负载测试"""
        print("🧪 开始渐进式负载测试...")

        test_cases = [
            (["Hello world"], 1),
            (["Hello world", "Test embedding"], 2),
            (["Test text"] * 4, 4),
            (["Sample text"] * 8, 8),
            (["Large batch test"] * 16, 16),
        ]

        for texts, expected_count in test_cases:
            print(f"\n测试 {expected_count} 个文本...")

            # 检查内存状态
            mem_before = self.get_memory_stats()

            # 执行embedding
            result = self.embed_texts(texts, batch_size=min(4, expected_count))

            if "error" in result:
                print(f"  ❌ 失败: {result['error']}")
                break
            else:
                # 检查内存状态
                mem_after = self.get_memory_stats()

                usage = result.get("usage", {})
                print(f"  ✅ 成功: {len(result['embeddings'])} embeddings")
                print(f"     处理时间: {usage.get('processing_time', 0):.2f}s")
                print(f"     速度: {usage.get('texts_per_second', 0):.1f} texts/sec")

                if mem_before and mem_after and "gpu_0" in mem_before:
                    gpu_before = mem_before["gpu_0"]["utilization_percent"]
                    gpu_after = mem_after["gpu_0"]["utilization_percent"]
                    print(f"     GPU显存: {gpu_before:.1f}% -> {gpu_after:.1f}%")

        print("\n🎯 测试完成")

def main():
    """测试主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Embedding服务测试客户端")
    parser.add_argument("--url", default="http://localhost:8000", help="服务器地址")
    parser.add_argument("--test", choices=["health", "memory", "embed", "load"],
                        default="health", help="测试类型")
    parser.add_argument("--texts", nargs="+", default=["Hello world"], help="测试文本")

    args = parser.parse_args()

    client = EmbeddingClient(args.url)

    if args.test == "health":
        print("🔍 健康检查...")
        result = client.health_check()
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif args.test == "memory":
        print("📊 内存统计...")
        result = client.get_memory_stats()
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif args.test == "embed":
        print(f"🔤 测试embedding ({len(args.texts)} 个文本)...")
        result = client.embed_texts(args.texts)
        if "error" not in result:
            print(f"✅ 成功生成 {len(result['embeddings'])} 个向量")
            print(f"向量维度: {len(result['embeddings'][0])}")
            print(f"处理时间: {result['usage']['processing_time']:.2f}s")
        else:
            print(f"❌ 失败: {result['error']}")

    elif args.test == "load":
        client.test_progressive_load()

if __name__ == "__main__":
    main()

