# test_gpu.py
import torch

print("Testing GPU setup...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Test tensor operations
    x = torch.randn(1000, 1000).to(device)
    y = torch.randn(1000, 1000).to(device)
    z = torch.matmul(x, y)
    print(f"GPU computation successful! Result shape: {z.shape}")
    
    # Memory info
    print(f"GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(f"GPU Memory cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    
    print("\n✅ GPU is READY for BERT training!")
    print("Run: python train_bert.py --use_gpu --sample 30000 --batch_size 32")
else:
    print("\n❌ GPU not available. Possible issues:")
    print("1. No NVIDIA GPU installed")
    print("2. NVIDIA drivers not installed")
    print("3. Wrong PyTorch version (CPU-only)")
    print("4. CUDA Toolkit not installed")