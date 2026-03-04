# 1x4分光器多波长实现策略

> 基于深度调研的实施方案  
> 目标: 980/1064/1310/1550nm并行功分  
> 阶段: 从1310/1550nm开始，逐步扩展

---

## 🎯 实施路线图

```
Phase 1 (现在): 1310/1550nm 优化 ⭐⭐⭐
   ↓ 目标: 250nm带宽，超越Lu 2019
   ↓ 平台: SOI
   ↓ 时间: 2-3周

Phase 2 (后续): 扩展到1064nm ⭐⭐
   ↓ 平台: SiN (覆盖1064+1310+1550)
   ↓ 验证: 三波段同时工作
   ↓ 时间: 1-2周

Phase 3 (可选): 980nm评估 ⭐
   ↓ 平台: GaAs或SiN
   ↓ 评估: 技术可行性
   ↓ 时间: 1周
```

---

## Phase 1: 1310/1550nm 详细方案

### 设计参数

```python
CONFIG_1310_1550 = {
    # 波长
    'wavelength_range': (1.25, 1.55),  # 300nm超宽带!
    'n_wavelengths': 31,  # 每10nm一个点
    'target_wavelengths': [1.31, 1.55],  # 重点优化
    
    # 材料 - SOI
    'n_si': 3.48,  # @ 1550nm (色散需考虑)
    'n_sio2': 1.44,
    'si_thickness': 0.22,  # μm (标准SOI)
    
    # 波导
    'wg_width': 0.5,  # μm (单模条件)
    'wg_height': 0.22,  # μm
    
    # 设计区域
    'design_size': (3.5, 3.5),  # μm (稍大以支持宽带)
    'grid_resolution': 0.02,  # 20nm
    
    # 制造
    'min_feature': 80e-3,  # 80nm
}
```

### 关键挑战与解决方案

#### 挑战1: 色散效应
**问题**: Si折射率随波长变化
```
n_Si @ 1310nm ≈ 3.48
n_Si @ 1550nm ≈ 3.47
Δn ≈ 0.01 (虽小但影响相位)
```

**解决**:
```python
# 色散模型 (Sellmeier方程简化)
def n_si(wavelength):
    """Si折射率色散"""
    λ = wavelength  # μm
    return 3.48 - 0.01 * (λ - 1.31) / (1.55 - 1.31)

# 每个波长使用对应的折射率
for wl in wavelengths:
    n = n_si(wl)
    sim = create_sim(wl, n)
```

---

#### 挑战2: 模式尺寸变化
**问题**: 长波长模式更大
```
Mode size @ 1310nm: ~0.4μm
Mode size @ 1550nm: ~0.5μm
```

**解决**:
- 使用绝热锥形过渡
- 优化区域考虑最大模式
- 逆向设计自动适应

---

#### 挑战3: 宽带相位匹配
**问题**: 耦合长度随波长变化
```
L_c ∝ λ / Δn
```

**解决**:
- 多波长同时优化
- 自适应权重 (中心波长更高)
- 放宽短波/长波要求

---

### 多波长优化策略

#### 权重设计
```python
def get_wavelength_weights(wavelengths):
    """
    自适应波长权重
    
    策略:
    - 1310nm和1550nm权重最高 (目标应用)
    - 中间波长适当降低
    - 边缘波长容忍度更高
    """
    weights = []
    for wl in wavelengths:
        # 高斯权重，中心在1430nm (1310和1550中点)
        w = np.exp(-((wl - 1.43) / 0.15)**2)
        
        # 在1310和1550nm处增加额外权重
        if abs(wl - 1.31) < 0.02 or abs(wl - 1.55) < 0.02:
            w *= 1.5
        
        weights.append(w)
    
    # 归一化
    return np.array(weights) / sum(weights)
```

#### 宽带目标函数
```python
def broadband_objective(params, wavelengths):
    """
    宽带目标函数
    
    同时优化:
    1. 各波长透射率
    2. 各波长均匀性
    3. 波长间一致性 (带宽平坦度)
    """
    transmissions = []
    uniformities = []
    
    for wl in wavelengths:
        T, U = simulate_at(params, wl)
        transmissions.append(T)
        uniformities.append(U)
    
    # 平均性能
    T_mean = np.mean(transmissions)
    U_mean = np.mean(uniformities)
    
    # 带宽平坦度 (标准差)
    T_std = np.std(transmissions)
    
    # 最坏波长惩罚
    T_min = np.min(transmissions)
    
    # 综合目标
    objective = (
        -1.0 * T_mean +           # 最大化平均透射
        0.5 * U_mean +            # 最小化不均匀度
        0.3 * T_std +             # 惩罚波动
        0.5 * (1 - T_min/T_mean)  # 惩罚最差波长
    )
    
    return objective
```

---

## Phase 2: 扩展到1064nm (SiN平台)

### 为什么选SiN?

```
SiN优势:
✓ 宽透明窗口: 400-2350nm (覆盖所有目标波长)
✓ 适中折射率: n~2.0 (对比度适中)
✓ 低损耗: ~0.1 dB/m
✓ 无TPA: 适合高功率

vs SOI:
✗ Si有带边吸收 @ ~1100nm
✗ 不适合1064nm高功率
```

### 设计调整

```python
CONFIG_SIN_1064_1550 = {
    # 材料 - SiN
    'n_sin': 2.0,  # 近似常数 (色散小)
    'n_sio2': 1.44,
    'sin_thickness': 0.4,  # μm (更厚以支持1064nm)
    
    # 波导尺寸调整
    'wg_width': 0.7,  # μm (1064nm需要更宽)
    'wg_height': 0.4,  # μm
    
    # 设计区域增大
    'design_size': (4.0, 4.0),  # μm
    
    # 波长范围
    'wavelength_range': (1.06, 1.55),  # 490nm!
    'n_wavelengths': 50,  # 更密集采样
}
```

### 关键挑战

1. **模式尺寸差异大**
   - 1064nm模式小，1550nm模式大
   - 需要渐变过渡结构

2. **高功率考虑 (1064nm)**
   - 大模式面积降低功率密度
   - 散热设计
   - 非线性效应 (SiN较低)

---

## Phase 3: 980nm评估

### 技术可行性分析

| 方案 | 平台 | 可行性 | 成本 |
|------|------|--------|------|
| SiN | SiN | ⭐⭐⭐ | 低 |
| GaAs | III-V | ⭐⭐ | 高 |
| SOI | Si | ⭐ | 不可能 (强吸收) |

**推荐**: 如果必须包含980nm，使用SiN平台 (Phase 2方案已覆盖)

---

## 实施优先级

### 立即开始 (本周)
1. [ ] 完善1310/1550nm Tidy3D实现
2. [ ] 31波长点优化测试
3. [ ] 目标: 带宽>250nm

### 短期 (2周内)
4. [ ] 制造容差分析 (±10nm)
5. [ ] 结构验证和可视化
6. [ ] 与Lu 2019对比

### 中期 (1月内)
7. [ ] SiN平台探索 (1064nm)
8. [ ] 三波段验证

### 可选
9. [ ] 980nm评估

---

## 预期成果

### Phase 1目标
```
波长范围: 1250-1550nm (300nm)
插入损耗: <0.3dB
均匀性: ±0.5dB
尺寸: 3.5×3.5 μm²
制造容差: ±10nm
优化时间: <15分钟

超越Lu 2019:
- 带宽: 300nm vs 200nm (+50%)
- 损耗: 0.3dB vs 0.5dB (-40%)
```

### Phase 2目标 (如果实施)
```
波长范围: 1064-1550nm (486nm)
平台: SiN
额外价值: 高功率1064nm应用
```

---

## 下一步行动

基于以上研究，建议:

1. **确认优先级**: 是否先做1310/1550nm，再考虑1064nm?
2. **准备Tidy3D环境**: 安装并配置API key
3. **运行Phase 1优化**: 300nm宽带设计

要我立即开始**Phase 1的完整Tidy3D实现**吗？
