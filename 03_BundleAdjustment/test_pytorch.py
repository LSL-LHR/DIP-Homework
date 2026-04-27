import torch
import sys

print("=" * 50)
print("PyTorch 环境检测")
print("=" * 50)

# 1. 版本信息
print(f"✓ PyTorch 版本: {torch.__version__}")
print(f"✓ Python 版本: {sys.version}")

# 2. CUDA 支持
print(f"✓ CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  - CUDA 版本: {torch.version.cuda}")
    print(f"  - GPU 数量: {torch.cuda.device_count()}")
    print(f"  - GPU 名称: {torch.cuda.get_device_name(0)}")

# 3. 基本张量操作
try:
    x = torch.rand(3, 3)
    y = torch.rand(3, 3)
    z = x + y
    print(f"✓ 张量运算正常: {z.shape}")
except Exception as e:
    print(f"✗ 张量运算失败: {e}")

# 4. CPU 计算测试
try:
    a = torch.randn(100, 100)
    b = torch.mm(a, a)
    print(f"✓ CPU 矩阵乘法正常")
except Exception as e:
    print(f"✗ CPU 计算失败: {e}")

# 5. GPU 计算测试（如果可用）
if torch.cuda.is_available():
    try:
        a_gpu = torch.randn(100, 100).cuda()
        b_gpu = torch.mm(a_gpu, a_gpu)
        print(f"✓ GPU 计算正常")
    except Exception as e:
        print(f"✗ GPU 计算失败: {e}")

# 6. 神经网络模块测试
try:
    import torch.nn as nn
    model = nn.Linear(10, 5)
    input_tensor = torch.randn(1, 10)
    output = model(input_tensor)
    print(f"✓ 神经网络模块正常")
except Exception as e:
    print(f"✗ 神经网络模块失败: {e}")

# 7. 优化器测试
try:
    import torch.optim as optim
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print(f"✓ 优化器模块正常")
except Exception as e:
    print(f"✗ 优化器模块失败: {e}")

# 8. 检查关键DLL（Windows）
import os
if os.name == 'nt':
    torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
    critical_dlls = ['torch_cpu.dll', 'c10.dll', 'shm.dll']
    print(f"\n检查关键DLL文件:")
    for dll in critical_dlls:
        dll_path = os.path.join(torch_lib_path, dll)
        if os.path.exists(dll_path):
            print(f"  ✓ {dll} 存在")
        else:
            print(f"  ✗ {dll} 缺失")

print("=" * 50)
print("检测完成")
print("=" * 50)