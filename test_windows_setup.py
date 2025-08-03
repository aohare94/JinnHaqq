#!/usr/bin/env python3
"""
Test script for Quranic AI Alignment Project Windows setup verification.
Tests all critical components for Windows 11 + RTX 4070 Super configuration.
"""

import sys
import platform
import subprocess
from pathlib import Path

def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def print_result(test_name, success, details=""):
    """Print test result with formatting."""
    status = "✓ PASS" if success else "✗ FAIL"
    print(f"{status:<8} {test_name}")
    if details:
        print(f"         {details}")

def check_python_version():
    """Check if Python version is compatible."""
    print_header("PYTHON VERSION CHECK")
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    print(f"Python version: {version_str}")
    print(f"Platform: {platform.platform()}")
    
    # Check if version is in recommended range
    compatible = (3, 9) <= version[:2] < (3, 12)
    print_result("Python Version Compatibility", compatible, 
                f"Recommended: 3.9-3.11, Found: {version.major}.{version.minor}")
    
    return compatible

def test_pytorch():
    """Test PyTorch installation and CUDA availability."""
    print_header("PYTORCH & CUDA TEST")
    
    try:
        import torch
        pytorch_version = torch.__version__
        print(f"PyTorch version: {pytorch_version}")
        print_result("PyTorch Import", True, f"Version {pytorch_version}")
        
        # Test CUDA
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print_result("CUDA Availability", True, 
                        f"{device_count} device(s), {device_name}, {memory_gb:.1f}GB")
            
            # Test GPU tensor operation
            try:
                test_tensor = torch.randn(1000, 1000).cuda()
                result = torch.matmul(test_tensor, test_tensor.T)
                gpu_test = result.shape == (1000, 1000)
                print_result("GPU Tensor Operations", gpu_test, "Matrix multiplication test")
            except Exception as e:
                print_result("GPU Tensor Operations", False, str(e))
        else:
            print_result("CUDA Availability", False, "CPU-only mode")
            
        return True
        
    except ImportError as e:
        print_result("PyTorch Import", False, str(e))
        return False
    except Exception as e:
        print_result("PyTorch Test", False, str(e))
        return False

def test_transformers():
    """Test Transformers library."""
    print_header("TRANSFORMERS TEST")
    
    try:
        import torch
        import transformers
        from transformers import AutoTokenizer, AutoModel
        
        version = transformers.__version__
        print_result("Transformers Import", True, f"Version {version}")
        
        # Test loading a small model
        try:
            print("Testing model loading (this may take a moment)...")
            tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            model = AutoModel.from_pretrained("distilbert-base-uncased")
            
            # Test tokenization
            text = "Hello, this is a test."
            tokens = tokenizer(text, return_tensors="pt")
            print_result("Model Loading & Tokenization", True, "distilbert-base-uncased")
            
            # Test model inference
            with torch.no_grad():
                outputs = model(**tokens)
            print_result("Model Inference", True, f"Output shape: {outputs.last_hidden_state.shape}")
            
        except Exception as e:
            print_result("Model Loading Test", False, str(e))
            
        return True
        
    except ImportError as e:
        print_result("Transformers Import", False, str(e))
        return False

def test_attention_optimization():
    """Test attention optimization packages."""
    print_header("ATTENTION OPTIMIZATION TEST")
    
    # Test xFormers
    try:
        import xformers
        print_result("xFormers", True, "Available (optimized attention)")
    except ImportError:
        print_result("xFormers", False, "Not available - using standard attention")
    
    # Test einops
    try:
        import einops
        print_result("einops", True, "Available")
    except ImportError:
        print_result("einops", False, "Not available")
    
    # Test Flash Attention (likely to fail on Windows)
    try:
        import flash_attn
        print_result("Flash Attention", True, "Available")
    except ImportError:
        print_result("Flash Attention", False, "Not available (expected on Windows)")

def test_arabic_processing():
    """Test Arabic text processing libraries."""
    print_header("ARABIC TEXT PROCESSING TEST")
    
    packages = [
        ("arabic-reshaper", "arabic_reshaper"),
        ("python-bidi", "bidi"),
        ("pyarabic", "pyarabic"),
        ("camel-tools", "camel_tools"),
    ]
    
    for package_name, import_name in packages:
        try:
            __import__(import_name)
            print_result(package_name, True)
        except ImportError:
            print_result(package_name, False, "Not available")
    
    # Test Arabic text processing
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        
        arabic_text = "السلام عليكم ورحمة الله وبركاته"
        reshaped_text = arabic_reshaper.reshape(arabic_text)
        bidi_text = get_display(reshaped_text)
        
        print_result("Arabic Text Processing", True, "Text reshaping and BiDi working")
        
    except Exception as e:
        print_result("Arabic Text Processing", False, str(e))

def test_core_packages():
    """Test core scientific and ML packages."""
    print_header("CORE PACKAGES TEST")
    
    packages = [
        "numpy", "pandas", "scipy", "scikit-learn", 
        "matplotlib", "seaborn", "plotly",
        "faiss", "sentence_transformers",
        "datasets", "tokenizers",
        "rich", "typer", "tqdm", "loguru"
    ]
    
    failed_packages = []
    
    for package in packages:
        try:
            if package == "scikit-learn":
                import sklearn
            elif package == "sentence_transformers":
                import sentence_transformers
            else:
                __import__(package)
            print_result(package, True)
        except ImportError:
            print_result(package, False, "Not available")
            failed_packages.append(package)
    
    return len(failed_packages) == 0

def test_gpu_memory():
    """Test GPU memory availability for the RTX 4070 Super."""
    print_header("GPU MEMORY TEST")
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            print_result("GPU Memory Test", False, "CUDA not available")
            return False
            
        # Get GPU properties
        device = torch.cuda.get_device_properties(0)
        total_memory = device.total_memory / 1e9  # Convert to GB
        
        # RTX 4070 Super should have 12GB
        expected_memory = 12.0
        memory_ok = total_memory >= (expected_memory * 0.9)  # Allow 10% tolerance
        
        print_result("GPU Memory Check", memory_ok, 
                    f"{total_memory:.1f}GB available (expected ~{expected_memory}GB)")
        
        # Test memory allocation
        try:
            # Try to allocate a reasonably large tensor
            test_size = int(1e8)  # ~400MB tensor
            test_tensor = torch.randn(test_size).cuda()
            torch.cuda.empty_cache()
            print_result("GPU Memory Allocation", True, f"Successfully allocated {test_size/1e6:.0f}M elements")
        except Exception as e:
            print_result("GPU Memory Allocation", False, str(e))
            
        return memory_ok
        
    except Exception as e:
        print_result("GPU Memory Test", False, str(e))
        return False

def test_project_structure():
    """Test if project structure is correct."""
    print_header("PROJECT STRUCTURE TEST")
    
    required_paths = [
        "src/",
        "src/core/",
        "src/quran/",
        "src/utils/",
        "requirements.txt",
        "requirements_windows.txt"
    ]
    
    all_exist = True
    for path in required_paths:
        exists = Path(path).exists()
        print_result(f"Path: {path}", exists)
        if not exists:
            all_exist = False
    
    return all_exist

def main():
    """Run all tests."""
    print_header("QURANIC AI ALIGNMENT PROJECT - WINDOWS SETUP VERIFICATION")
    print("Testing setup for Windows 11 + RTX 4070 Super")
    
    tests = [
        ("Python Version", check_python_version),
        ("PyTorch & CUDA", test_pytorch),
        ("Transformers", test_transformers),
        ("Attention Optimization", test_attention_optimization),
        ("Arabic Processing", test_arabic_processing),
        ("Core Packages", test_core_packages),
        ("GPU Memory", test_gpu_memory),
        ("Project Structure", test_project_structure),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print_result(f"{test_name} (ERROR)", False, str(e))
            results.append((test_name, False))
    
    # Print summary
    print_header("SUMMARY")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {status:<6} {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Your Windows setup is ready for the Quranic AI Alignment Project.")
        print("\nNext steps:")
        print("1. Download Arabic models: python -c \"from transformers import AutoModel; AutoModel.from_pretrained('aubmindlab/bert-base-arabertv2')\"")
        print("2. Run the main application: python -m src.core.engine")
    else:
        print(f"\n⚠️  {total-passed} tests failed. Please check the setup and install missing packages.")
        print("\nFor missing packages, try:")
        print("pip install <package_name>")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)