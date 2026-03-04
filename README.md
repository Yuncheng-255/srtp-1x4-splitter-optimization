# SRTP: 1x4光功率分光器逆向设计

基于Tidy3D的拓扑优化实现，利用4重对称性实现4倍加速。

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Tidy3D](https://img.shields.io/badge/Tidy3D-2.0+-orange.svg)](https://docs.flexcompute.com/projects/tidy3d/en/latest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 项目目标

设计超紧凑、宽带、低损耗的1x4光功率分光器：

| 指标 | Lu 2019 | 我们的目标 | 提升 |
|------|---------|-----------|------|
| 带宽 | 200nm | **250nm** | +25% |
| 损耗 | 0.5dB | **0.3dB** | -40% |
| 优化速度 | ~小时 | **<15分钟** | 4x |
| 制造容差 | 无 | **±10nm** | 新增 |

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行示例

```bash
python examples/basic.py
```

### 基础用法

```python
from srtp_splitter import SymmetricSplitterOptimizer

optimizer = SymmetricSplitterOptimizer(
    design_region_size=(3.0, 3.0),
    grid_resolution=0.1,
    wavelength_range=(1.45, 1.65),
    n_wavelengths=11,
    symmetry=True  # 4x加速
)

params, history = optimizer.optimize(max_iterations=100)
```

## 📁 项目结构

```
srtp-1x4-splitter-optimization/
├── src/srtp_splitter/     # 核心代码
│   ├── optimizer.py        # 对称优化器
│   ├── manufacturing.py    # 制造约束
│   └── utils.py            # 工具函数
├── examples/               # 使用示例
│   ├── basic.py
│   └── advanced_multi_wavelength.py
├── tests/                  # 单元测试
├── docs/                   # 文档
│   ├── quickstart.md
│   └── reports/            # 阶段报告
├── requirements.txt        # 依赖清单
└── README.md              # 本文件
```

## 🔬 核心方法

1. **对称性加速** - 4倍速度提升，75%参数减少
2. **滤波投影法** - 密度滤波 + Heaviside投影
3. **宽带优化** - 21个波长点，最坏情况设计
4. **制造鲁棒性** - ±10nm制造容差

## 📚 文档

- [快速开始](docs/quickstart.md)
- [阶段报告](docs/reports/)

## 📝 许可证

MIT License

## 👤 作者

SRTP项目 - 浙江大学
