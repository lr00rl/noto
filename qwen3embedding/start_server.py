# start_server.py - 内存优化版本
import uvicorn
import logging
import sys
import os
import psutil
import torch
from pathlib import Path

def setup_memory_optimized_environment():
    """设置内存优化环境"""
    # PyTorch内存优化
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,garbage_collection_threshold:0.6")
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "0")
    os.environ.setdefault("TORCH_CUDNN_BENCHMARK", "1")

    # Python内存优化
    os.environ.setdefault("PYTHONMALLOC", "malloc")
    os.environ.setdefault("MALLOC_TRIM_THRESHOLD_", "0")

    # 禁用tokenizer并行化以节省内存
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # 如果是低内存系统，启用内存映射
    memory_gb = psutil.virtual_memory().total / 1024**3
    if memory_gb < 16:
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

def check_system_resources():
    """检查系统资源并给出建议"""
    print("🔍 系统资源检查:")

    # 内存检查
    memory = psutil.virtual_memory()
    memory_gb = memory.total / 1024**3
    print(f"   系统内存: {memory_gb:.1f}GB (可用: {memory.available / 1024**3:.1f}GB)")

    if memory_gb < 8:
        print("   ⚠️  警告: 系统内存不足8GB，建议关闭其他程序")

    # GPU检查
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"   GPU数量: {gpu_count}")

        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            gpu_memory_gb = props.total_memory / 1024**3
            print(f"   GPU {i}: {props.name} ({gpu_memory_gb:.1f}GB)")

            if gpu_memory_gb < 4:
                print(f"   ⚠️  GPU {i} 显存不足4GB，将使用CPU offload")
            elif gpu_memory_gb < 6:
                print(f"   ⚠️  GPU {i} 显存较小，建议使用0.6B模型")
    else:
        print("   ❌ 未检测到CUDA GPU，将使用CPU模式")

    print()

def get_optimized_config():
    """根据系统资源获取优化配置"""
    config = {
        "host": "0.0.0.0",
        "port": 8000,
        "workers": 1,  # embedding服务必须单进程
        "loop": "asyncio",  # 在低内存系统使用asyncio而不是uvloop
        "http": "h11",     # 使用更轻量的HTTP解析器
        "log_level": "info",
        "access_log": True,
        "reload": False,
    }

    # 根据内存调整配置
    memory_gb = psutil.virtual_memory().total / 1024**3

    if memory_gb >= 32:
        # 高内存系统
        config.update({
            "loop": "uvloop",
            "http": "httptools",
            "limit_concurrency": 500,
            "limit_max_requests": 5000,
        })
        print("📈 检测到高内存系统，使用性能优化配置")

    elif memory_gb >= 16:
        # 中等内存系统
        config.update({
            "limit_concurrency": 200,
            "limit_max_requests": 2000,
        })
        print("📊 检测到中等内存系统，使用平衡配置")

    else:
        # 低内存系统
        config.update({
            "limit_concurrency": 50,
            "limit_max_requests": 500,
            "timeout_keep_alive": 5,  # 减少保持连接时间
        })
        print("🔋 检测到低内存系统，使用内存优化配置")

    # GPU特定配置
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory_gb < 6:
            config["limit_concurrency"] = min(config.get("limit_concurrency", 50), 20)
            print(f"🎯 GPU显存较小({gpu_memory_gb:.1f}GB)，限制并发数为20")

    return config

def setup_logging():
    """设置日志配置"""
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
            "access": {
                "format": "%(asctime)s - %(client_addr)s - \"%(request_line)s\" %(status_code)s - %(process_time).3fs",
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "formatter": "default",
                "class": "logging.handlers.RotatingFileHandler",
                "filename": "embedding_server.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 3,
            },
        },
        "loggers": {
            "uvicorn": {"handlers": ["default", "file"], "level": "INFO", "propagate": False},
            "uvicorn.error": {"level": "INFO"},
            "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
            "embedding_server": {"handlers": ["default", "file"], "level": "INFO"},
        },
    }
    return log_config

def main():
    """主启动函数"""
    print("🚀 启动内存优化版 Qwen3 Embedding 服务器")
    print("=" * 50)

    # 设置环境
    setup_memory_optimized_environment()

    # 检查系统资源
    check_system_resources()

    # 获取优化配置
    config = get_optimized_config()

    # 设置日志
    log_config = setup_logging()

    print("🛠️  服务器配置:")
    for key, value in config.items():
        if key not in ["log_config"]:
            print(f"   {key}: {value}")
    print()

    print("🌐 服务地址:")
    print(f"   API: http://{config['host']}:{config['port']}")
    print(f"   文档: http://{config['host']}:{config['port']}/docs")
    print(f"   健康检查: http://{config['host']}:{config['port']}/health")
    print(f"   内存统计: http://{config['host']}:{config['port']}/memory/stats")
    print()

    try:
        # 启动服务器
        uvicorn.run(
            "embedding_server:app",
            **config,
            log_config=log_config
        )
    except KeyboardInterrupt:
        print("\n👋 服务器已停止")
    except Exception as e:
        print(f"\n❌ 服务器启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
