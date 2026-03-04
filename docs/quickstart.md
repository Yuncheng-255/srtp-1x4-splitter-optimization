# 快速开始

## 安装

```bash
pip install -r requirements.txt
```

## 基础用法

```python
from srtp_splitter import SymmetricSplitterOptimizer

# 创建优化器
optimizer = SymmetricSplitterOptimizer(
    design_region_size=(3.0, 3.0),
    grid_resolution=0.1,
    wavelength_range=(1.45, 1.65),
    n_wavelengths=11,
    symmetry=True
)

# 运行优化
params, history = optimizer.optimize(max_iterations=100)

# 获取最终结构
final_structure = optimizer.get_final_structure()
```

## 运行示例

```bash
python examples/basic.py
python examples/advanced_multi_wavelength.py
```

## 运行测试

```bash
pytest tests/
```
