#!/usr/bin/env python3
"""
运行 1x4 分光器优化

Usage:
    python scripts/run_optimization.py [--mock]
"""

import sys
import os

# 添加 src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core import SplitterOptimizer, OptimizerConfig


def main():
    # 解析参数
    mock_mode = '--mock' in sys.argv
    
    print("="*70)
    print("SRTP 1x4 Optical Splitter Optimization")
    print("="*70)
    
    if mock_mode:
        print("Mode: MOCK (no Tidy3D simulation)")
    else:
        print("Mode: FULL (with Tidy3D)")
        # 检查 API key
        if not os.environ.get('TINY3D_API_KEY'):
            print("\nWarning: TINY3D_API_KEY not set!")
            print("Set it with: export TINY3D_API_KEY='your-key'")
            print("Or run in mock mode: python scripts/run_optimization.py --mock")
            return
    
    print()
    
    # 创建配置
    config = OptimizerConfig(
        design_size=(3.0, 3.0),
        grid_resolution=0.1,  # 粗网格用于快速测试
        wavelength_range=(1.45, 1.65),
        n_wavelengths=5,
        use_symmetry=True,
        max_iterations=100,
        learning_rate=0.1
    )
    
    # 创建优化器并运行
    optimizer = SplitterOptimizer(config)
    result = optimizer.optimize(init_method="random")
    
    # 保存结果
    os.makedirs('output', exist_ok=True)
    optimizer.save_result(result, 'output/optimization_result')
    
    print("\n" + "="*70)
    print("Optimization complete!")
    print(f"Results saved to: output/")
    print("="*70)


if __name__ == "__main__":
    main()
