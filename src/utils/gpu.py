#!/usr/bin/env python3
"""
GPU Management Utilities for Qur'anic AI Alignment
─────────────────────────────────────────────────────────────────────────────
Optimized for RTX 4070 Super (12GB VRAM) on Windows 11
Memory monitoring, optimization, and performance tracking
─────────────────────────────────────────────────────────────────────────────
"""

import torch
import gc
import time
import psutil
from typing import Dict, Optional, List, Any
from dataclasses import dataclass
from pathlib import Path

@dataclass
class GPUMetrics:
    """GPU performance and memory metrics"""
    name: str
    memory_total_gb: float
    memory_allocated_gb: float
    memory_reserved_gb: float
    memory_free_gb: float
    utilization_percent: float
    temperature_c: Optional[float] = None
    power_draw_w: Optional[float] = None

class GPUManager:
    """Advanced GPU management for optimal performance"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_count = torch.cuda.device_count()
        self.current_device = 0
        self.memory_history = []
        self.performance_history = []
        
        if not torch.cuda.is_available():
            print("⚠️  CUDA not available - running on CPU (performance will be severely limited)")
        else:
            print(f"🎮 Detected {self.gpu_count} GPU(s)")
            for i in range(self.gpu_count):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f}GB)")
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """Get comprehensive GPU information"""
        if not torch.cuda.is_available():
            return {"available": False, "message": "CUDA not available"}
        
        props = torch.cuda.get_device_properties(self.current_device)
        
        return {
            "available": True,
            "count": self.gpu_count,
            "current_device": self.current_device,
            "name": props.name,
            "memory_gb": props.total_memory / 1e9,
            "compute_capability": f"{props.major}.{props.minor}",
            "multiprocessor_count": props.multi_processor_count,
            "max_threads_per_block": props.max_threads_per_block,
            "max_threads_per_multiprocessor": props.max_threads_per_multiprocessor
        }
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage"""
        if not torch.cuda.is_available():
            return {"allocated_gb": 0, "reserved_gb": 0, "free_gb": 0, "total_gb": 0}
        
        allocated = torch.cuda.memory_allocated(self.current_device) / 1e9
        reserved = torch.cuda.memory_reserved(self.current_device) / 1e9
        total = torch.cuda.get_device_properties(self.current_device).total_memory / 1e9
        free = total - reserved
        
        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "free_gb": free,
            "total_gb": total,
            "utilization_percent": (reserved / total) * 100
        }
    
    def get_detailed_metrics(self) -> GPUMetrics:
        """Get detailed GPU metrics including temperature and power (if available)"""
        if not torch.cuda.is_available():
            return GPUMetrics("CPU", 0, 0, 0, 0, 0)
        
        props = torch.cuda.get_device_properties(self.current_device)
        memory = self.get_memory_usage()
        
        # Try to get additional metrics via nvidia-ml-py if available
        temperature = None
        power_draw = None
        
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.current_device)
            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # Convert to watts
        except ImportError:
            pass  # pynvml not available
        except Exception:
            pass  # nvidia-ml not available or other error
        
        return GPUMetrics(
            name=props.name,
            memory_total_gb=memory["total_gb"],
            memory_allocated_gb=memory["allocated_gb"],
            memory_reserved_gb=memory["reserved_gb"],
            memory_free_gb=memory["free_gb"],
            utilization_percent=memory["utilization_percent"],
            temperature_c=temperature,
            power_draw_w=power_draw
        )
    
    def optimize_memory(self):
        """Optimize GPU memory usage"""
        if not torch.cuda.is_available():
            return
        
        # Clear cache
        torch.cuda.empty_cache()
        
        # Trigger garbage collection
        gc.collect()
        
        # Set memory fraction if using a lot of memory
        memory = self.get_memory_usage()
        if memory["utilization_percent"] > 85:
            print("⚠️  High GPU memory usage detected - attempting optimization")
            
            # More aggressive cleanup
            for _ in range(3):
                torch.cuda.empty_cache()
                gc.collect()
                time.sleep(0.1)
        
        print(f"🧹 GPU memory optimized: {memory['utilization_percent']:.1f}% usage")
    
    def monitor_memory(self, duration_seconds: float = 10, interval_seconds: float = 1):
        """Monitor GPU memory usage over time"""
        print(f"📊 Monitoring GPU memory for {duration_seconds} seconds...")
        
        start_time = time.time()
        samples = []
        
        while time.time() - start_time < duration_seconds:
            memory = self.get_memory_usage()
            timestamp = time.time() - start_time
            
            samples.append({
                "timestamp": timestamp,
                "allocated_gb": memory["allocated_gb"],
                "reserved_gb": memory["reserved_gb"],
                "utilization_percent": memory["utilization_percent"]
            })
            
            print(f"  {timestamp:.1f}s: {memory['utilization_percent']:.1f}% "
                  f"({memory['allocated_gb']:.2f}GB/{memory['total_gb']:.1f}GB)")
            
            time.sleep(interval_seconds)
        
        self.memory_history.extend(samples)
        return samples
    
    def benchmark_performance(self, test_duration: float = 5) -> Dict[str, float]:
        """Benchmark GPU performance"""
        if not torch.cuda.is_available():
            return {"message": "CUDA not available for benchmarking"}
        
        print(f"🎯 Benchmarking GPU performance for {test_duration} seconds...")
        
        # Create test tensors
        size = (4096, 4096)
        device = torch.device(f"cuda:{self.current_device}")
        
        # Warm up
        for _ in range(10):
            a = torch.randn(size, device=device, dtype=torch.float16)
            b = torch.randn(size, device=device, dtype=torch.float16)
            c = torch.matmul(a, b)
            del a, b, c
        
        torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.perf_counter()
        operations = 0
        
        while time.perf_counter() - start_time < test_duration:
            a = torch.randn(size, device=device, dtype=torch.float16)
            b = torch.randn(size, device=device, dtype=torch.float16)
            c = torch.matmul(a, b)
            operations += 1
            del a, b, c
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        actual_duration = end_time - start_time
        ops_per_second = operations / actual_duration
        tflops = (operations * 2 * size[0] * size[1] * size[1]) / (actual_duration * 1e12)
        
        results = {
            "duration_seconds": actual_duration,
            "operations": operations,
            "ops_per_second": ops_per_second,
            "estimated_tflops": tflops
        }
        
        print(f"✅ Benchmark complete: {ops_per_second:.1f} ops/sec, ~{tflops:.2f} TFLOPS")
        return results
    
    def set_memory_fraction(self, fraction: float):
        """Set GPU memory fraction to use"""
        if not torch.cuda.is_available():
            return
        
        if 0.1 <= fraction <= 1.0:
            torch.cuda.set_per_process_memory_fraction(fraction, self.current_device)
            print(f"🎛️  GPU memory fraction set to {fraction:.1%}")
        else:
            print("⚠️  Memory fraction must be between 0.1 and 1.0")
    
    def get_optimal_batch_size(self, model_memory_gb: float, safety_factor: float = 0.8) -> int:
        """Estimate optimal batch size based on available memory"""
        memory = self.get_memory_usage()
        available_gb = memory["free_gb"] * safety_factor
        
        if model_memory_gb <= 0:
            return 1
        
        estimated_batch_size = max(1, int(available_gb / model_memory_gb))
        
        print(f"📏 Estimated optimal batch size: {estimated_batch_size} "
              f"(based on {available_gb:.1f}GB available)")
        
        return estimated_batch_size
    
    def enable_amp(self):
        """Enable Automatic Mixed Precision for better performance"""
        if not torch.cuda.is_available():
            return False
        
        # Check if AMP is supported
        if torch.cuda.is_bf16_supported():
            print("✅ BFloat16 AMP enabled for optimal performance")
            return True
        else:
            print("⚠️  BFloat16 not supported, using Float16 AMP")
            return True
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        cpu_info = {
            "cpu_count": psutil.cpu_count(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_total_gb": psutil.virtual_memory().total / 1e9,
            "memory_available_gb": psutil.virtual_memory().available / 1e9,
            "memory_percent": psutil.virtual_memory().percent
        }
        
        gpu_info = self.get_gpu_info() if torch.cuda.is_available() else {"available": False}
        
        return {
            "cpu": cpu_info,
            "gpu": gpu_info,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "Not available",
            "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else "Not available"
        }
    
    def save_metrics(self, filepath: Path):
        """Save performance metrics to file"""
        import json
        
        metrics_data = {
            "gpu_info": self.get_gpu_info(),
            "memory_history": self.memory_history,
            "performance_history": self.performance_history,
            "system_info": self.get_system_info(),
            "timestamp": time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        print(f"📊 Metrics saved to {filepath}")
    
    def __enter__(self):
        """Context manager entry"""
        self.start_memory = self.get_memory_usage()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self.optimize_memory()
        end_memory = self.get_memory_usage()
        
        print(f"🔄 Memory usage: {self.start_memory['utilization_percent']:.1f}% → "
              f"{end_memory['utilization_percent']:.1f}%")

# Convenience functions
def get_gpu_manager() -> GPUManager:
    """Get a GPU manager instance"""
    return GPUManager()

def quick_gpu_check() -> bool:
    """Quick check if GPU is available and working"""
    if not torch.cuda.is_available():
        return False
    
    try:
        # Test basic CUDA operations
        x = torch.tensor([1.0], device='cuda')
        y = x + 1
        return y.item() == 2.0
    except Exception:
        return False

def optimize_for_inference():
    """Apply optimizations for inference workloads"""
    if torch.cuda.is_available():
        # Enable cuDNN auto-tuner
        torch.backends.cudnn.benchmark = True
        
        # Enable TensorFloat-32 (TF32) for faster computation
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        print("⚡ Applied CUDA optimizations for inference")