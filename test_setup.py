#!/usr/bin/env python3
"""
Test script for Qur'anic AI Alignment setup verification
────────────────────────────────────────────────────────────────────────────
Run this script to verify your Windows 11 setup is working correctly
────────────────────────────────────────────────────────────────────────────
"""

import sys
import traceback
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported"""
    print("🔧 Testing module imports...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch: {e}")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
    except ImportError as e:
        print(f"❌ Transformers: {e}")
        return False
    
    try:
        import xformers
        print(f"✅ xFormers: Available")
    except ImportError:
        print(f"⚠️  xFormers: Not available (using standard attention)")
    
    try:
        from src.utils.gpu import GPUManager
        print(f"✅ GPU Manager: Available")
    except ImportError as e:
        print(f"❌ GPU Manager: {e}")
        return False
    
    try:
        from src.utils.arabic import ArabicHandler
        print(f"✅ Arabic Handler: Available")
    except ImportError as e:
        print(f"❌ Arabic Handler: {e}")
        return False
    
    try:
        from src.quran.processor import QuranProcessor
        print(f"✅ Qur'an Processor: Available")
    except ImportError as e:
        print(f"❌ Qur'an Processor: {e}")
        return False
    
    return True

def test_gpu():
    """Test GPU availability and performance"""
    print("\n🎮 Testing GPU setup...")
    
    try:
        import torch
        from src.utils.gpu import GPUManager, quick_gpu_check
        
        # Basic CUDA check
        if torch.cuda.is_available():
            print(f"✅ CUDA Available: {torch.cuda.is_available()}")
            print(f"✅ CUDA Version: {torch.version.cuda}")
            print(f"✅ Device Count: {torch.cuda.device_count()}")
            
            device_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ GPU: {device_name} ({memory_gb:.1f}GB)")
            
            # Quick functionality test
            if quick_gpu_check():
                print("✅ GPU Functionality: Working")
            else:
                print("❌ GPU Functionality: Failed")
                return False
            
            # GPU Manager test
            gpu_manager = GPUManager()
            memory_info = gpu_manager.get_memory_usage()
            print(f"✅ Memory Usage: {memory_info['utilization_percent']:.1f}%")
            
        else:
            print("❌ CUDA not available")
            return False
    
    except Exception as e:
        print(f"❌ GPU Test Failed: {e}")
        traceback.print_exc()
        return False
    
    return True

def test_quran_processor():
    """Test Qur'an processor functionality"""
    print("\n📖 Testing Qur'an Processor...")
    
    try:
        from src.quran.processor import QuranProcessor
        
        # Initialize processor
        processor = QuranProcessor()
        
        # Test loading sample data
        success = processor.load_complete_text()
        if success:
            print("✅ Sample Qur'an data loaded")
            
            # Get statistics
            stats = processor.get_statistics()
            print(f"✅ Loaded: {stats['total_surahs']} Surahs, {stats['total_verses']} verses")
            print(f"✅ Alignment verses: {stats['alignment_verses']}")
            print(f"✅ Muqatta'at: {stats['muqattaat_count']}")
            
            # Test verse retrieval
            verse = processor.get_verse(1, 1)
            if verse:
                print(f"✅ Verse retrieval: {verse.ref}")
                print(f"   Arabic: {verse.arabic}")
                print(f"   Translation: {verse.translation}")
            
        else:
            print("❌ Failed to load Qur'an data")
            return False
    
    except Exception as e:
        print(f"❌ Qur'an Processor Test Failed: {e}")
        traceback.print_exc()
        return False
    
    return True

def test_arabic_handler():
    """Test Arabic text processing"""
    print("\n🔤 Testing Arabic Handler...")
    
    try:
        from src.utils.arabic import ArabicHandler
        
        handler = ArabicHandler()
        
        # Test text processing
        test_text = "بِسْمِ ٱللَّهِ ٱلرَّحْمَـٰنِ ٱلرَّحِيمِ"
        
        # Test normalization
        normalized = handler.normalize_text(test_text)
        print(f"✅ Text normalization: Working")
        
        # Test analysis
        metrics = handler.analyze_text(test_text)
        print(f"✅ Text analysis: {metrics.word_count} words, {metrics.letter_count} letters")
        
        # Test diacritic handling
        no_diacritics = handler.remove_diacritics(test_text)
        print(f"✅ Diacritic removal: Working")
        
        # Test validation
        validation = handler.validate_quranic_text(test_text)
        print(f"✅ Qur'anic validation: {validation['confidence_score']:.2f} confidence")
        
    except Exception as e:
        print(f"❌ Arabic Handler Test Failed: {e}")
        traceback.print_exc()
        return False
    
    return True

def test_config():
    """Test configuration system"""
    print("\n⚙️  Testing Configuration...")
    
    try:
        from src.core.config import get_default_config, get_high_performance_config
        
        # Test default config
        config = get_default_config()
        print(f"✅ Default config: {config.model_id}")
        print(f"✅ Memory limit: {config.max_memory_gb}GB")
        print(f"✅ Target TPS: {config.target_tokens_per_second}")
        
        # Test high performance config
        perf_config = get_high_performance_config()
        print(f"✅ Performance config: Compile={perf_config.compile_model}")
        
    except Exception as e:
        print(f"❌ Config Test Failed: {e}")
        traceback.print_exc()
        return False
    
    return True

def main():
    """Run all tests"""
    print("🕌 Qur'anic AI Alignment - Setup Verification")
    print("=" * 60)
    print("This script will test your Windows 11 setup for the project")
    print()
    
    all_passed = True
    
    # Run all tests
    tests = [
        ("Module Imports", test_imports),
        ("GPU Setup", test_gpu), 
        ("Configuration", test_config),
        ("Arabic Handler", test_arabic_handler),
        ("Qur'an Processor", test_quran_processor),
    ]
    
    for test_name, test_func in tests:
        try:
            if not test_func():
                all_passed = False
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("Your setup is ready for Qur'anic AI alignment research.")
        print("\nNext steps:")
        print("1. Run: python scripts/setup_windows.bat")
        print("2. Start interactive mode: python -c 'from src.core.engine import create_engine; engine = create_engine(); print(\"Engine ready!\")'")
    else:
        print("❌ SOME TESTS FAILED")
        print("Please check the error messages above and fix any issues.")
        print("Refer to the README.md for installation instructions.")
    print("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)