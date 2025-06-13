"""
如果仍然显存不足，可以使用GGUF量化版本：
1. 安装llama.cpp:
pip install llama-cpp-python
- llama.cpp默认使用CPU，针对CPU做了大量优化
- 支持AVX2、AVX-512等SIMD指令集加速
- 量化格式（Q4_0、Q8_0等）可显著减少内存占用和提升CPU推理速度
- 适合内存较大但GPU显存不足的场景

wget https://github.com/ggml-org/llama.cpp/releases/download/b5631/llama-b5631-bin-ubuntu-x64.zip
export PATH=/opt/llama.cpp/bin:$PATH
export LD_LIBRARY_PATH="/opt/llama.cpp/bin:$LD_LIBRARY_PATH"
llama-cli - 主要的推理工具（以前叫main）
llama-server - HTTP API服务器
llama-quantize - 模型量化工具
llama-run - 简化的运行工具

2. 下载量化模型:
wget https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/main/Qwen3-Embedding-0.6B-Q8_0.gguf?download=true
3. 使用量化模型:

GGUF（GPT-Generated Unified Format）是一种用于存储和分发大型语言模型的文件格式，通常用于像llama.cpp这样的推理引擎
"""
from llama_cpp import Llama
from typing import List
import time

class GGUFEmbeddingRTX1650:
    def __init__(self, model_path: str):
        """GGUF量化版本，显存使用约500MB"""
        self.llm = Llama(
            model_path=model_path,
            embedding=True,      # 必须设置为True才能使用embedding功能
            n_gpu_layers=32,     # 将层加载到GPU
            n_ctx=512,          # 减少上下文长度
            n_batch=64,         # 调整批次大小（系统会自动调整到64）
            verbose=False
        )

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            try:
                result = self.llm.create_embedding(text)
                # 检查返回结果的结构
                if 'data' in result and len(result['data']) > 0:
                    embedding = result['data'][0]['embedding']
                    embeddings.append(embedding)
                else:
                    print(f"Warning: Failed to get embedding for text: {text[:50]}...")
                    # 创建一个零向量作为占位符
                    embeddings.append([0.0] * 1024)  # 假设嵌入维度为1024
            except Exception as e:
                print(f"Error processing text '{text[:50]}...': {e}")
                # 创建一个零向量作为占位符
                embeddings.append([0.0] * 1024)
        return embeddings

    def embed_texts_batch(self, texts: List[str]) -> List[List[float]]:
        """批量处理文本（如果模型支持）"""
        embeddings = []
        # llama.cpp 通常不支持真正的批量处理，所以逐个处理
        for i, text in enumerate(texts):
            if i % 5 == 0:  # 每5个文本显示进度
                print(f"Processing {i+1}/{len(texts)}")

            result = self.llm.create_embedding(text)
            embedding = result['data'][0]['embedding']
            embeddings.append(embedding)
        return embeddings

if __name__ == "__main__":
    # 测试优化版本
    print("=== RTX 1650 GGUF 优化测试 ===")

    # 确保模型文件路径正确
    # model_path = "Qwen3-Embedding-0.6B-Q8_0.gguf"
    model_path = "Qwen3-Embedding-0.6B-f16.gguf"
    # 或者使用完整路径
    # model_path = "/path/to/your/Qwen3-Embedding-0.6B-Q8_0.gguf"

    try:
        model = GGUFEmbeddingRTX1650(model_path)
        print("模型加载成功！")
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("请确保模型文件路径正确，或下载正确的模型文件")
        exit(1)

    # 测试数据
    test_texts = [
        "What is machine learning?",
        "How do neural networks work?",
        "Explain deep learning",
        "What is artificial intelligence?",
        "How to optimize GPU performance?",
        "What is embedding in NLP?",
        "How to reduce VRAM usage?",
        "What is batch processing?"
    ] * 2  # 16个文本

    print(f"\n测试 {len(test_texts)} 个文本...")
    start_time = time.time()

    try:
        embeddings = model.embed_texts_batch(test_texts)
        print(f"Embedding shape: {len(embeddings)} texts, {len(embeddings[0])} dimensions")
        ## Embedding shape: 16 texts, 5 dimensions, 1024 values per dimension
        end_time = time.time()

        print(f"\n=== 测试结果 ===")
        print(f"处理文本数: {len(embeddings)}")
        print(f"总耗时: {end_time - start_time:.2f}秒")
        print(f"平均速度: {len(embeddings)/(end_time - start_time):.1f} texts/sec")
        if embeddings:
            print(f"Embedding维度: {len(embeddings[0])}")

        # 显示前几个embedding的部分内容
        print(f"\n前3个文本的embedding示例:")
        for i in range(min(3, len(embeddings))):
            print(f"文本 {i+1}: {test_texts[i]}")
            print(f"Embedding (前10维): {embeddings[i][:10]}")
            print()

    except Exception as e:
        print(f"处理过程中出错: {e}")
        import traceback
        traceback.print_exc()
