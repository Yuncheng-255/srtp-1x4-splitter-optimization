#!/usr/bin/env python3
"""
Tidy3D 1x4分光器自动优化器 - 迭代改进版

改进点:
1. 绝热锥形输入/输出 - 降低反射损耗
2. 优化设计区域 - 改善均匀性
3. 多波长目标函数 - 平衡宽带性能
4. 自动参数扫描 - 找到最佳配置

Author: SRTP
Date: 2026-02-26
"""

import os
import sys
import numpy as np
import json
from pathlib import Path

# 设置API Key
api_key = '6BEU36edpFWSDFrQWo2IE6h9PRyJWvTzEZSVs7NF8mFgafju'
os.environ['TINY3D_API_KEY'] = api_key

# 配置文件
config_dir = Path.home() / '.config' / 'tidy3d'
config_dir.mkdir(parents=True, exist_ok=True)
(config_dir / 'config').write_text(f"apikey = '{api_key}'")

import tidy3d as td
import tidy3d.web as web

print("="*70)
print("Tidy3D 1x4分光器自动优化")
print("="*70)
print(f"Tidy3D版本: {td.__version__}")
print()

# ========== 参数配置 ==========
# 波长范围
WAVELENGTH_MIN = 1.45  # μm
WAVELENGTH_MAX = 1.65  # μm
N_WAVELENGTHS = 7

wavelengths = np.linspace(WAVELENGTH_MIN, WAVELENGTH_MAX, N_WAVELENGTHS)
freqs = td.C_0 / wavelengths

print(f"波长范围: {WAVELENGTH_MIN*1000:.0f}-{WAVELENGTH_MAX*1000:.0f}nm")
print(f"波长点数: {N_WAVELENGTHS}")
print()

# 材料
n_si = 3.476
n_sio2 = 1.444
si = td.Medium(permittivity=n_si**2)
sio2 = td.Medium(permittivity=n_sio2**2)

def create_tapered_splitter(taper_length=1.5, design_size=(4.0, 4.0)):
    """
    创建带绝热锥形的1x4分光器
    
    参数:
        taper_length: 锥形长度 (μm)
        design_size: 设计区域大小 (μm, μm)
    """
    
    # 输入波导 + 锥形
    structures = []
    
    # 1. 输入直波导
    wg_in = td.Structure(
        geometry=td.Box(
            center=(-taper_length - 1, 0, 0),
            size=(2, 0.5, 0.22)
        ),
        medium=si,
        name="input_wg"
    )
    structures.append(wg_in)
    
    # 2. 输入锥形 (绝热过渡)
    # 使用PolySlab创建锥形
    taper_vertices = [
        (-taper_length - 1, -0.25),  # 后端左
        (-taper_length - 1, 0.25),   # 后端右
        (-1, -design_size[1]/4),     # 前端左
        (-1, design_size[1]/4)       # 前端右
    ]
    
    # 简化为Box近似
    taper_in = td.Structure(
        geometry=td.Box(
            center=(-taper_length/2 - 0.5, 0, 0),
            size=(taper_length, 0.5 + taper_length*0.3, 0.22)
        ),
        medium=si,
        name="input_taper"
    )
    structures.append(taper_in)
    
    # 3. 设计区域 (耦合区)
    design = td.Structure(
        geometry=td.Box(
            center=(0, 0, 0),
            size=(design_size[0], design_size[1], 0.22)
        ),
        medium=si,
        name="design_region"
    )
    structures.append(design)
    
    # 4. 4个输出锥形
    angles = [45, 135, 225, 315]
    for i, angle in enumerate(angles):
        rad = np.radians(angle)
        r_center = 1 + taper_length/2
        x = r_center * np.cos(rad)
        y = r_center * np.sin(rad)
        
        taper_out = td.Structure(
            geometry=td.Box(
                center=(x, y, 0),
                size=(taper_length*0.8, 0.5 + taper_length*0.25, 0.22)
            ),
            medium=si,
            name=f"output_taper_{i}"
        )
        structures.append(taper_out)
    
    # 5. 输出直波导
    for i, angle in enumerate(angles):
        rad = np.radians(angle)
        r_out = 2 + taper_length
        x = r_out * np.cos(rad)
        y = r_out * np.sin(rad)
        
        wg_out = td.Structure(
            geometry=td.Box(
                center=(x, y, 0),
                size=(1.5, 0.5, 0.22)
            ),
            medium=si,
            name=f"output_wg_{i}"
        )
        structures.append(wg_out)
    
    return structures

def run_simulation_with_config(config, task_name="test"):
    """运行指定配置的仿真"""
    
    taper_length = config.get('taper_length', 1.5)
    design_size = config.get('design_size', (4.0, 4.0))
    
    print(f"\n配置: taper={taper_length}μm, design={design_size}")
    
    # 创建结构
    structures = create_tapered_splitter(taper_length, design_size)
    
    # 模式源
    mode_source = td.ModeSource(
        center=(-taper_length - 1.5, 0, 0),
        size=(0, 2.5, 2.5),
        source_time=td.GaussianPulse(freq0=freqs[N_WAVELENGTHS//2], fwidth=freqs[0]/15),
        direction="+",
        mode_spec=td.ModeSpec(num_modes=1),
        mode_index=0
    )
    
    # 输出监视器
    monitors = []
    for i in range(4):
        angle = [45, 135, 225, 315][i]
        rad = np.radians(angle)
        r = 2.5 + taper_length
        x, y = r * np.cos(rad), r * np.sin(rad)
        
        monitors.append(td.ModeMonitor(
            center=(x, y, 0),
            size=(0, 2.5, 2.5),
            freqs=freqs.tolist(),
            name=f"port_{i}",
            mode_spec=td.ModeSpec(num_modes=1)
        ))
    
    # 仿真
    sim_size = max(8, 4 + 2*taper_length)
    sim = td.Simulation(
        size=(sim_size, sim_size, 3),
        grid_spec=td.GridSpec.uniform(dl=0.04),
        structures=structures,
        sources=[mode_source],
        monitors=monitors,
        run_time=6e-12,
        boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML())
    )
    
    print(f"  网格: {sim.grid.num_cells}")
    
    # 运行
    try:
        data = web.run(sim, task_name=task_name, verbose=False)
        
        # 分析结果
        results = analyze_results(data, freqs)
        return results
        
    except Exception as e:
        print(f"  ❌ 错误: {e}")
        return None

def analyze_results(data, freqs):
    """分析仿真结果"""
    
    T_per_wavelength = []
    uniformity_per_wavelength = []
    
    for wl_idx in range(len(freqs)):
        T_list = []
        for i in range(4):
            mode_data = data[f"port_{i}"]
            amp_data = mode_data.amps.sel(direction="+", f=freqs[wl_idx])
            amp_val = amp_data.values
            
            if isinstance(amp_val, np.ndarray):
                amp_val = amp_val.item() if amp_val.size == 1 else amp_val[0]
            
            T = abs(amp_val)**2
            T_list.append(T)
        
        T_total = sum(T_list)
        T_mean = np.mean(T_list)
        T_std = np.std(T_list)
        
        T_per_wavelength.append(T_total)
        uniformity_per_wavelength.append(T_std / (T_mean + 1e-10))
    
    T_array = np.array(T_per_wavelength)
    T_max = np.max(T_array)
    T_min = np.min(T_array)
    
    # 3dB带宽
    above_3db = T_array >= T_max * 0.5
    if np.any(above_3db):
        indices = np.where(above_3db)[0]
        bandwidth_3db = (wavelengths[indices[-1]] - wavelengths[indices[0]]) * 1000
    else:
        bandwidth_3db = 0
    
    # 平均均匀性
    avg_uniformity = np.mean(uniformity_per_wavelength)
    
    return {
        'bandwidth_3db_nm': bandwidth_3db,
        'peak_transmission': T_max,
        'insertion_loss_db': -10 * np.log10(T_max),
        'avg_uniformity': avg_uniformity,
        'transmissions': T_per_wavelength
    }

# ========== 主程序 ==========
print("🔧 开始参数扫描优化...")
print()

# 测试不同配置
configs = [
    {'taper_length': 1.0, 'design_size': (3.5, 3.5)},
    {'taper_length': 1.5, 'design_size': (3.5, 3.5)},
    {'taper_length': 2.0, 'design_size': (3.5, 3.5)},
    {'taper_length': 1.5, 'design_size': (4.0, 4.0)},
    {'taper_length': 2.0, 'design_size': (4.0, 4.0)},
]

all_results = []

for i, config in enumerate(configs):
    print(f"\n{'='*60}")
    print(f"测试配置 {i+1}/{len(configs)}")
    print(f"{'='*60}")
    
    result = run_simulation_with_config(config, task_name=f"1x4_config_{i+1}")
    
    if result:
        print(f"\n  结果:")
        print(f"    带宽: {result['bandwidth_3db_nm']:.0f}nm")
        print(f"    峰值透射: {result['peak_transmission']*100:.1f}%")
        print(f"    损耗: {result['insertion_loss_db']:.2f}dB")
        print(f"    均匀性: {result['avg_uniformity']:.2f}")
        
        all_results.append({
            'config': config,
            'result': result
        })

# 找出最佳配置
print(f"\n{'='*70}")
print("📊 所有配置对比")
print(f"{'='*70}")

for i, item in enumerate(all_results):
    cfg = item['config']
    res = item['result']
    print(f"\n配置 {i+1}: taper={cfg['taper_length']}μm, design={cfg['design_size']}")
    print(f"  带宽: {res['bandwidth_3db_nm']:.0f}nm")
    print(f"  损耗: {res['insertion_loss_db']:.2f}dB")
    print(f"  均匀性: {res['avg_uniformity']:.2f}")

# 选择最佳 (综合考虑带宽和均匀性)
best_idx = 0
best_score = 0

for i, item in enumerate(all_results):
    res = item['result']
    # 评分: 带宽权重50%，均匀性权重30%，损耗权重20%
    score = (res['bandwidth_3db_nm'] / 300) * 0.5 + \
            (1 / (res['avg_uniformity'] + 1)) * 0.3 + \
            (1 / (res['insertion_loss_db'] + 1)) * 0.2
    
    if score > best_score:
        best_score = score
        best_idx = i

best_config = all_results[best_idx]['config']
best_result = all_results[best_idx]['result']

print(f"\n{'='*70}")
print("🎉 最佳配置")
print(f"{'='*70}")
print(f"  锥形长度: {best_config['taper_length']}μm")
print(f"  设计区域: {best_config['design_size']}μm")
print(f"\n  性能:")
print(f"    带宽: {best_result['bandwidth_3db_nm']:.0f}nm")
print(f"    峰值透射: {best_result['peak_transmission']*100:.1f}%")
print(f"    插入损耗: {best_result['insertion_loss_db']:.2f}dB")
print(f"    均匀性指数: {best_result['avg_uniformity']:.2f}")

# 保存结果
with open('optimization_results.json', 'w') as f:
    json.dump({
        'best_config': best_config,
        'best_result': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                       for k, v in best_result.items()},
        'all_results': all_results
    }, f, indent=2)

print(f"\n✅ 结果已保存到 optimization_results.json")
print(f"\n查看任务: https://tidy3d.simulation.cloud/workbench")
print(f"{'='*70}")
