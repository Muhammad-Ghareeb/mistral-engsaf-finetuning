"""
Setup verification script for Mistral fine-tuning notebook.
Run this before starting training to ensure everything is configured correctly.
"""

import sys
import os

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python version {version.major}.{version.minor}.{version.micro} is too old. Need 3.8+")
        return False

def check_pytorch():
    """Check PyTorch and CUDA"""
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.version.cuda}")
            print(f"✓ GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"    Memory: {props.total_memory / 1e9:.2f} GB")
            return True
        else:
            print("✗ CUDA not available - GPU training will not work")
            return False
    except ImportError:
        print("✗ PyTorch not installed")
        return False

def check_required_packages():
    """Check required packages"""
    required = [
        'transformers',
        'peft',
        'bitsandbytes',
        'datasets',
        'accelerate',
        'sklearn',
        'pandas',
        'numpy',
        'nltk',
    ]
    
    missing = []
    for package in required:
        try:
            if package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} not installed")
            missing.append(package)
    
    return len(missing) == 0

def check_dataset():
    """Check dataset files"""
    dataset_dir = "EngSAF dataset"
    
    if not os.path.exists(dataset_dir):
        print(f"✗ Dataset directory not found: {dataset_dir}")
        return False
    
    required_files = ['train.csv', 'val.csv', 'unseen_question.csv', 'unseen_answers.csv']
    missing_files = []
    
    for file in required_files:
        filepath = os.path.join(dataset_dir, file)
        if os.path.exists(filepath):
            # Check file size
            size = os.path.getsize(filepath) / 1024  # KB
            print(f"✓ {file} ({size:.1f} KB)")
        else:
            print(f"✗ {file} not found")
            missing_files.append(file)
    
    return len(missing_files) == 0

def check_disk_space():
    """Check available disk space"""
    import shutil
    total, used, free = shutil.disk_usage(".")
    free_gb = free / (1024**3)
    
    print(f"✓ Available disk space: {free_gb:.2f} GB")
    
    if free_gb < 20:
        print("⚠ Warning: Less than 20GB free. May need more space for checkpoints.")
        return False
    return True

def check_output_directory():
    """Check output directory permissions"""
    output_dir = "./output"
    try:
        os.makedirs(output_dir, exist_ok=True)
        test_file = os.path.join(output_dir, "test_write.txt")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        print(f"✓ Output directory writable: {output_dir}")
        return True
    except Exception as e:
        print(f"✗ Cannot write to output directory: {e}")
        return False

def main():
    """Run all checks"""
    print("=" * 60)
    print("Mistral Fine-tuning Setup Verification")
    print("=" * 60)
    print()
    
    checks = [
        ("Python Version", check_python_version),
        ("PyTorch & CUDA", check_pytorch),
        ("Required Packages", check_required_packages),
        ("Dataset Files", check_dataset),
        ("Disk Space", check_disk_space),
        ("Output Directory", check_output_directory),
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n[{name}]")
        print("-" * 40)
        result = check_func()
        results.append((name, result))
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    all_passed = True
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
        if not result:
            all_passed = False
    
    print()
    if all_passed:
        print("✓ All checks passed! You're ready to start training.")
    else:
        print("✗ Some checks failed. Please fix the issues above before training.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

