# gpu_diagnostic.py
import torch
import sys
import subprocess
import platform

print("="*60)
print("GPU DIAGNOSTIC TOOL")
print("="*60)

print(f"System: {platform.system()} {platform.release()}")
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")

# Check CUDA availability
print(f"\n1. PyTorch CUDA Status:")
print(f"   CUDA available: {torch.cuda.is_available()}")
print(f"   CUDA version (PyTorch): {torch.version.cuda if hasattr(torch.version, 'cuda') else 'N/A'}")

# Check GPU directly
print(f"\n2. GPU Detection:")
try:
    import nvidia_smi
    nvidia_smi.nvmlInit()
    device_count = nvidia_smi.nvmlDeviceGetCount()
    print(f"   NVIDIA GPU count: {device_count}")
    for i in range(device_count):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        print(f"   GPU {i}: {nvidia_smi.nvmlDeviceGetName(handle)}")
        print(f"     Memory: {info.total / 1024**3:.1f} GB")
    nvidia_smi.nvmlShutdown()
except:
    print("   NVIDIA SMI not available")

# Check system GPU
print(f"\n3. System GPU Check:")
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, shell=True)
    if result.returncode == 0:
        print("   nvidia-smi output:")
        lines = result.stdout.split('\n')[:5]
        for line in lines:
            print(f"   {line}")
    else:
        print("   nvidia-smi failed or not found")
except:
    print("   Could not run nvidia-smi")

# Check PyTorch build
print(f"\n4. PyTorch Build Info:")
print(f"   Build with CUDA: {'cuda' in torch.__version__.lower()}")
print(f"   CUDA_HOME: {torch.utils.cpp_extension.CUDA_HOME if hasattr(torch.utils.cpp_extension, 'CUDA_HOME') else 'Not set'}")

# List CUDA devices
print(f"\n5. CUDA Devices:")
try:
    for i in range(torch.cuda.device_count()):
        print(f"   Device {i}: {torch.cuda.get_device_name(i)}")
        print(f"     Capability: {torch.cuda.get_device_capability(i)}")
        print(f"     Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
except:
    print("   No CUDA devices found")

print("\n" + "="*60)
print("RECOMMENDED ACTION:")
if torch.cuda.is_available():
    print("✅ GPU is available! Run: python train_bert.py --use_gpu")
else:
    print("❌ GPU not detected. Follow steps below.")
print("="*60)