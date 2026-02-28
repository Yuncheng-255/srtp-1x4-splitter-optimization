# SRTP 1x4 Optical Power Splitter

基于 Tidy3D 的逆向设计优化项目

## 项目结构

```
srtp-1x4-splitter-optimization/
├── docs/                      # 文档
│   ├── reports/              # 阶段报告
│   └── research/             # 研究笔记
├── src/                       # 源代码
│   ├── core/                 # 核心优化器
│   ├── utils/                # 工具函数
│   └── tests/                # 测试代码
├── scripts/                   # 运行脚本
├── notebooks/                 # Jupyter notebooks
├── requirements.txt           # Python依赖
└── README.md                  # 本文档
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 设置 Tidy3D API Key
export TINY3D_API_KEY='your-api-key'

# 运行优化
python scripts/run_optimization.py
```

## 核心功能

- **对称性优化**: 利用4重对称性减少75%参数
- **宽带设计**: 支持 C+L 波段 (1450-1700nm)
- **制造容差**: 考虑 ±10nm 工艺偏差
- **自动收敛**: 智能学习率和早停机制

## 性能目标

| 指标 | 目标 | 参考 (Lu 2019) |
|------|------|----------------|
| 带宽 | 250nm+ | 200nm |
| 损耗 | <0.3dB | 0.5dB |
| 优化时间 | <10分钟 | - |
| 制造容差 | ±10nm | ±5nm |

## 文档

- [阶段1报告](docs/reports/PHASE1_REPORT.md)
- [多波长研究](docs/research/RESEARCH_MULTI_WAVELENGTH.md)
- [实现策略](docs/research/IMPLEMENTATION_STRATEGY.md)

## 许可证

MIT
