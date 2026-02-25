#!/usr/bin/env python3
"""
真实Tidy3D 1x4分光器优化 - 修正版

修正:
- 正确处理ModeMonitor数据格式
- 支持多波长结果提取
"""

import os
import sys
import numpy as np

# 检查Python版本
if sys.version_info >= (3, 14):
    print("❌ 错误: Python 3.14+与Tidy3D不兼容")
    sys.exit(1)

try:
    import tidy3d as td
    import tidy3d.web as web
    print(f"✅ Tidy3D版本: {td.__version__}")
except ImportError:
    print("❌ Tidy3D未安装")
    sys.exit(1)

# 配置API Key
api_key = '6BEU36edpFWSDFrQWo2IE6h9PRyJWvTzEZSVs7NF8mFgafju'
os.environ['TINY3D_API_KEY'] = api_key

# 创建配置文件
import pathlib
config_dir = pathlib.Path.home() / '.config' / 'tidy3d'
config_dir.mkdir(parents=True, exist_ok=True)
(config_dir / 'config').write_text(f"apikey = '{api_key}'")

print("✅ API Key已配置")
print("\n" + "="*70)
print("真实Tidy3D 1x4分光器优化")
print("="*70)

# 参数
WAVELENGTH_CENTER = 1.55
WAVELENGTH_RANGE = 0.3
N_WAVELENGTHS = 7

wavelengths = np.linspace(
    WAVELENGTH_CENTER - WAVELENGTH_RANGE/2,
    WAVELENGTH_CENTER + WAVELENGTH_RANGE/2,
    N_WAVELENGTHS
)
freqs = td.C_0 / wavelengths

print(f"\n波长范围: {wavelengths[0]*1000:.0f}-{wavelengths[-1]*1000:.0f}nm")

# 材料
n_si = 3.476
si = td.Medium(permittivity=n_si**2)

# 结构
wg_input = td.Structure(
    geometry=td.Box(center=(-2.5, 0, 0), size=(1.5, 0.5, 0.22)),
    medium=si
)

outputs = []
for angle in [45, 135, 225, 315]:
    rad = np.radians(angle)
    r = 2.5
    x, y = r * np.cos(rad), r * np.sin(rad)
    outputs.append(td.Structure(
        geometry=td.Box(center=(x, y, 0), size=(0.5, 1.0, 0.22)),
        medium=si
    ))

design_region = td.Structure(
    geometry=td.Box(center=(0, 0, 0), size=(3.5, 3.5, 0.22)),
    medium=si
)

# 源和监视器
mode_source = td.ModeSource(
    center=(-2.0, 0, 0),
    size=(0, 2, 2),
    source_time=td.GaussianPulse(freq0=freqs[N_WAVELENGTHS//2], fwidth=freqs[0]/20),
    direction="+",
    mode_spec=td.ModeSpec(num_modes=1),
    mode_index=0
)

monitors = []
for i, angle in enumerate([45, 135, 225, 315]):
    rad = np.radians(angle)
    x, y = 2.0 * np.cos(rad), 2.0 * np.sin(rad)
    monitors.append(td.ModeMonitor(
        center=(x, y, 0),
        size=(0, 2, 2),
        freqs=freqs.tolist(),
        name=f"port_{i}",
        mode_spec=td.ModeSpec(num_modes=1)
    ))

# 仿真
sim = td.Simulation(
    size=(8, 8, 3),
    grid_spec=td.GridSpec.uniform(dl=0.05),
    structures=[wg_input, design_region] + outputs,
    sources=[mode_source],
    monitors=monitors,
    run_time=5e-12,
    boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML())
)

print(f"✅ 仿真创建: {sim.grid.num_cells} 单元")
print("\n🚀 提交到Tidy3D Cloud...")
print("(约需1-2分钟)")

try:
    data = web.run(sim, task_name="1x4_splitter_v2")
    print("✅ 仿真完成!\n")
    
    print("📊 结果分析:")
    print()
    
    # 提取结果 - 正确的方法
    for wl_idx, wl in enumerate(wavelengths):
        T_list = []
        for i in range(4):
            # 获取mode amplitude data
            mode_data = data[f"port_{i}"]
            # 提取指定频率的amplitude
            amp_data = mode_data.amps.sel(direction="+", f=freqs[wl_idx])
            # 获取数值并计算透射率
            amp_val = amp_data.values
            if isinstance(amp_val, np.ndarray):
                amp_val = amp_val.item() if amp_val.size == 1 else amp_val[0]
            T = abs(amp_val)**2
            T_list.append(T)
        
        T_total = sum(T_list)
        T_mean = np.mean(T_list)
        T_std = np.std(T_list)
        
        print(f"  {wl*1000:.0f}nm: 总透射={T_total*100:.1f}%, "
              f"每端口={T_mean*100:.1f}%±{T_std*100:.1f}%")
    
    # 计算带宽
    T_center = []
    for wl_idx in range(N_WAVELENGTHS):
        T_sum = 0
        for i in range(4):
            mode_data = data[f"port_{i}"]
            amp_data = mode_data.amps.sel(direction="+", f=freqs[wl_idx])
            amp_val = amp_data.values
            if isinstance(amp_val, np.ndarray):
                amp_val = amp_val.item() if amp_val.size == 1 else amp_val[0]
            T_sum += abs(amp_val)**2
        T_center.append(T_sum)
    
    T_array = np.array(T_center)
    T_max = np.max(T_array)
    
    # 3dB带宽
    above_3db = T_array >= T_max * 0.5
    if np.any(above_3db):
        indices = np.where(above_3db)[0]
        bandwidth_3db = (wavelengths[indices[-1]] - wavelengths[indices[0]]) * 1000
    else:
        bandwidth_3db = 0
    
    print()
    print("="*70)
    print("🎉 最终成果")
    print("="*70)
    print(f"  3dB带宽: {bandwidth_3db:.0f}nm")
    print(f"  峰值透射: {T_max*100:.1f}%")
    print(f"  插入损耗: {-10*np.log10(T_max):.2f}dB")
    print("="*70)
    
    if bandwidth_3db >= 200:
        print("\n✅ 带宽达标!")
    if -10*np.log10(T_max) < 1.0:
        print("✅ 损耗达标!")
    
    if bandwidth_3db >= 200 and -10*np.log10(T_max) < 1.0:
        print("\n🎉🎉🎉 显著成果达成! 🎉🎉🎉")

except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ 完成!")
