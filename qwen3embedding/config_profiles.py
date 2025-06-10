# config_profiles.py
"""
不同显存大小的配置文件
自动根据硬件选择最佳配置
"""

import torch
import psutil
from dataclasses import dataclass
from typing import Dict, Any, Optional
import json

@dataclass
class MemoryProfile:
    """内存配置文件"""
    name: str
    model_name: str
    max_batch_size: int
    max_length: int
    precision: str
    gradient_checkpointing: bool
    cpu_offload: bool
    max_concurrent_requests: int
    cleanup_frequency: int  # 每N个批次清理一次内存
    description: str

class ProfileManager:
    """配置文件管理器"""

    def __init__(self):
        self.profiles = {
            # 低显存配置 (2-4GB)
            "low_memory": MemoryProfile(
                name="low_memory",
                model_name="Qwen/Qwen3-Embedding-0.6B",
                max_batch_size=2,
                max_length=256,
                precision="fp16",
                gradient_checkpointing=True,
                cpu_offload=True,
                max_concurrent_requests=10,
                cleanup_frequency=1,  # 每个批次都清理
                description="适用于2-4GB显存的低配置"
            ),

            # 中低显存配置 (4-6GB)
            "medium_low": MemoryProfile(
                name="medium_low",
                model_name="Qwen/Qwen3-Embedding-0.6B",
                max_batch_size=4,
                max_length=512,
                precision="fp16",
                gradient_checkpointing=True,
                cpu_offload=False,
                max_concurrent_requests=20,
                cleanup_frequency=2,
                description="适用于4-6GB显存"
            ),

            # 中等显存配置 (6-8GB)
            "medium": MemoryProfile(
                name="medium",
                model_name="Qwen/Qwen3-Embedding-4B",
                max_batch_size=8,
                max_length=512,
                precision="fp16",
                gradient_checkpointing=True,
                cpu_offload=False,
                max_concurrent_requests=50,
                cleanup_frequency=3,
                description="适用于6-8GB显存"
            ),

            # 中高显存配置 (8-12GB)
            "medium_high": MemoryProfile(
                name="medium_high",
                model_name="Qwen/Qwen3-Embedding-4B",
                max_batch_size=16,
                max_length=1024,
                precision="fp16",
                gradient_checkpointing=False,
                cpu_offload=False,
                max_concurrent_requests=100,
                cleanup_frequency=5,
                description="适用于8-12GB显存"
            ),

            # 高显存配置 (12GB+)
            "high_memory": MemoryProfile(
                name="high_memory",
                model_name="Qwen/Qwen3-Embedding-8B",
                max_batch_size=32,
                max_length=2048,
                precision="fp16",
                gradient_checkpointing=False,
                cpu_offload=False,
                max_concurrent_requests=200,
                cleanup_frequency=10,
                description="适用于12GB+显存"
            ),

            # CPU模式
            "cpu_only": MemoryProfile(
                name="cpu_only",
                model_name="Qwen/Qwen3-Embedding-0.6B",
                max_batch_size=1,
                max_length=512,
                precision="fp32",
                gradient_checkpointing=False,
                cpu_offload=True,
                max_concurrent_requests=5,
                cleanup_frequency=1,
                description="纯CPU模式"
            )
        }

    def detect_optimal_profile(self) -> MemoryProfile:
        """自动检测最佳配置"""
        if not torch.cuda.is_available():
            print("🔍 未检测到CUDA，使用CPU模式")
            return self.profiles["cpu_only"]

        # 获取GPU信息
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        system_memory_gb = psutil.virtual_memory().total / 1024**3

        print(f"🔍 硬件检测:")
        print(f"   GPU显存: {gpu_memory_gb:.1f}GB")
        print(f"   系统内存: {system_memory_gb:.1f}GB")

        # 根据显存大小选择配置
        if gpu_memory_gb >= 12:
            profile = self.profiles["high_memory"]
        elif gpu_memory_gb >= 8:
            profile = self.profiles["medium_high"]
        elif gpu_memory_gb >= 6:
            profile = self.profiles["medium"]
        elif gpu_memory_gb >= 4:
            profile = self.profiles["medium_low"]
        else:
            profile = self.profiles["low_memory"]

        # 如果系统内存不足，降级配置
        if system_memory_gb < 8:
            print("⚠️  系统内存不足8GB，降级到低内存配置")
            profile = self.profiles["low_memory"]

        print(f"📋 选择配置: {profile.name} - {profile.description}")
        return profile

    def get_profile(self, name: str) -> Optional[MemoryProfile]:
        """获取指定配置"""
        return self.profiles.get(name)

    def list_profiles(self):
        """列出所有可用配置"""
        print("📋 可用配置:")
        for name, profile in self.profiles.items():
            print(f"   {name}: {profile.description}")
            print(f"     模型: {profile.model_name}")
            print(f"     批次大小: {profile.max_batch_size}")
            print(f"     最大长度: {profile.max_length}")
            print()

    def save_config(self, profile: MemoryProfile, filepath: str = "current_config.json"):
        """保存当前配置到文件"""
        config_dict = {
            "profile_name": profile.name,
            "model_name": profile.model_name,
            "max_batch_size": profile.max_batch_size,
            "max_length": profile.max_length,
            "precision": profile.precision,
            "gradient_checkpointing": profile.gradient_checkpointing,
            "cpu_offload": profile.cpu_offload,
            "max_concurrent_requests": profile.max_concurrent_requests,
            "cleanup_frequency": profile.cleanup_frequency,
            "description": profile.description
        }

        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"💾 配置已保存到 {filepath}")

# 动态配置调整器
class DynamicConfigAdjuster:
    """动态配置调整器 - 运行时根据内存使用情况调整参数"""

    def __init__(self, initial_profile: MemoryProfile):
        self.profile = initial_profile
        self.adjustment_history = []
        self.oom_count = 0

    def handle_oom_error(self):
        """处理显存不足错误"""
        self.oom_count += 1
        old_batch_size = self.profile.max_batch_size

        # 减半批次大小
        new_batch_size = max(1, self.profile.max_batch_size // 2)
        self.profile.max_batch_size = new_batch_size

        # 记录调整
        adjustment = {
            "type": "oom_reduction",
            "old_batch_size": old_batch_size,
            "new_batch_size": new_batch_size,
            "oom_count": self.oom_count
        }
        self.adjustment_history.append(adjustment)

        print(f"⚠️  显存不足，批次大小调整: {old_batch_size} -> {new_batch_size}")

        return new_batch_size

    def optimize_batch_size(self, success_rate: float, avg_processing_time: float):
        """根据成功率和处理时间优化批次大小"""
        if success_rate > 0.95 and avg_processing_time < 2.0:
            # 性能良好，尝试增加批次
            if self.profile.max_batch_size < 32:
                old_size = self.profile.max_batch_size
                self.profile.max_batch_size = min(32, int(old_size * 1.5))
                print(f"📈 性能良好，增加批次大小: {old_size} -> {self.profile.max_batch_size}")

        elif success_rate < 0.8:
            # 成功率低，减少批次
            old_size = self.profile.max_batch_size
            self.profile.max_batch_size = max(1, old_size - 1)
            print(f"📉 成功率较低，减少批次大小: {old_size} -> {self.profile.max_batch_size}")

    def get_current_config(self) -> Dict[str, Any]:
        """获取当前配置"""
        return {
            "max_batch_size": self.profile.max_batch_size,
            "max_length": self.profile.max_length,
            "precision": self.profile.precision,
            "cleanup_frequency": self.profile.cleanup_frequency,
            "adjustments_made": len(self.adjustment_history),
            "oom_count": self.oom_count
        }

# 智能资源监控器
class ResourceMonitor:
    """智能资源监控器"""

    def __init__(self):
        self.history = []
        self.alert_thresholds = {
            "gpu_memory": 90,  # GPU显存使用率阈值
            "system_memory": 85,  # 系统内存使用率阈值
            "temperature": 85  # GPU温度阈值（如果可获取）
        }

    def get_current_usage(self) -> Dict[str, Any]:
        """获取当前资源使用情况"""
        usage = {
            "timestamp": torch.cuda.Event().record() if torch.cuda.is_available() else None,
            "system_memory": psutil.virtual_memory().percent
        }

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            usage.update({
                "gpu_memory_allocated": allocated,
                "gpu_memory_total": total,
                "gpu_memory_percent": (allocated / total) * 100,
                "gpu_memory_available": total - allocated
            })

        return usage

    def check_resource_pressure(self) -> Dict[str, bool]:
        """检查资源压力"""
        usage = self.get_current_usage()
        pressure = {}

        # 系统内存压力
        pressure["high_system_memory"] = usage["system_memory"] > self.alert_thresholds["system_memory"]

        # GPU显存压力
        if torch.cuda.is_available():
            pressure["high_gpu_memory"] = usage["gpu_memory_percent"] > self.alert_thresholds["gpu_memory"]
            pressure["low_gpu_memory"] = usage["gpu_memory_available"] < 1.0  # 少于1GB可用

        return pressure

    def suggest_adjustments(self) -> List[str]:
        """根据资源压力提供调整建议"""
        pressure = self.check_resource_pressure()
        suggestions = []

        if pressure.get("high_gpu_memory", False):
            suggestions.append("降低批次大小")
            suggestions.append("启用梯度检查点")
            suggestions.append("考虑使用CPU offload")

        if pressure.get("high_system_memory", False):
            suggestions.append("减少并发请求数")
            suggestions.append("增加内存清理频率")

        if pressure.get("low_gpu_memory", False):
            suggestions.append("立即执行内存清理")
            suggestions.append("暂停接收新请求")

        return suggestions

# 使用示例
def main():
    """配置管理器使用示例"""
    # 创建配置管理器
    manager = ProfileManager()

    # 列出所有配置
    manager.list_profiles()

    # 自动检测最佳配置
    optimal_profile = manager.detect_optimal_profile()

    # 保存配置
    manager.save_config(optimal_profile)

    # 创建动态调整器
    adjuster = DynamicConfigAdjuster(optimal_profile)

    # 创建资源监控器
    monitor = ResourceMonitor()

    # 检查当前资源状态
    current_usage = monitor.get_current_usage()
    print("📊 当前资源使用:")
    for key, value in current_usage.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")

    # 检查资源压力
    pressure = monitor.check_resource_pressure()
    if any(pressure.values()):
        print("⚠️  检测到资源压力:")
        for issue, present in pressure.items():
            if present:
                print(f"   {issue}")

        suggestions = monitor.suggest_adjustments()
        print("💡 建议:")
        for suggestion in suggestions:
            print(f"   - {suggestion}")

if __name__ == "__main__":
    main()
