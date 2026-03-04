#!/usr/bin/env python3
"""
基础示例 - 快速开始

运行1x4分光器的基本优化。
"""

from srtp_splitter import SymmetricSplitterOptimizer


def main():
    print("=" * 60)
    print("SRTP 1x4光功率分光器优化 - 基础示例")
    print("=" * 60)
    
    # 创建优化器
    optimizer = SymmetricSplitterOptimizer(
        design_region_size=(3.0, 3.0),  # 3μm × 3μm 设计区域
        grid_resolution=0.1,  # 100nm 分辨率
        wavelength_range=(1.45, 1.65),  # C波段 1450-1650nm
        n_wavelengths=11,
        symmetry=True  # 开启4重对称性加速
    )
    
    # 运行优化
    print("\n开始优化...")
    params, history = optimizer.optimize(
        max_iterations=100,
        learning_rate=0.1,
        verbose=True
    )
    
    # 获取最终结构
    final_structure = optimizer.get_final_structure()
    print(f"\n最终结构尺寸: {final_structure.shape}")
    
    # 绘制收敛曲线
    optimizer.plot_convergence(save_path="results/convergence.png")
    print("\n收敛曲线已保存到 results/convergence.png")


if __name__ == "__main__":
    main()
