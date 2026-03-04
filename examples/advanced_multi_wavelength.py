#!/usr/bin/env python3
"""
高级示例 - 多波长优化

展示如何进行宽带多波长优化。
"""

import numpy as np
from srtp_splitter import SymmetricSplitterOptimizer, ManufacturingConstraints


def main():
    print("=" * 60)
    print("SRTP 1x4光功率分光器优化 - 多波长示例")
    print("=" * 60)
    
    # 创建优化器，增加波长点数
    optimizer = SymmetricSplitterOptimizer(
        design_region_size=(3.0, 3.0),
        grid_resolution=0.1,
        wavelength_range=(1.40, 1.70),  # 更宽的波段 300nm
        n_wavelengths=21,  # 更多波长点
        symmetry=True
    )
    
    # 制造约束
    mc = ManufacturingConstraints(
        min_feature_size=80e-9,  # 80nm 最小特征
        grid_size=20e-9
    )
    
    # 运行优化
    print("\n开始宽带优化...")
    params, history = optimizer.optimize(
        max_iterations=150,
        learning_rate=0.05,
        filter_radius=3,
        verbose=True
    )
    
    # 应用制造约束
    final_structure = optimizer.get_final_structure()
    print("\n验证可制造性...")
    is_valid = mc.validate_manufacturability(final_structure)
    
    if is_valid:
        print("✅ 结构可制造")
    else:
        print("⚠️ 需要调整制造参数")
    
    # 保存结果
    np.save("results/final_structure.npy", final_structure)
    optimizer.plot_convergence(save_path="results/multi_wavelength_convergence.png")


if __name__ == "__main__":
    main()
