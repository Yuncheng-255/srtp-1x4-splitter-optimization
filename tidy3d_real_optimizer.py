#!/usr/bin/env python3
"""
Tidy3D 1x4 Splitter - Real Implementation
连接真实Tidy3D cloud进行优化

环境要求:
- pip install tidy3d
- export TINY3D_API_KEY='你的API key'

获取API key:
1. 访问 https://tidy3d.simulation.cloud
2. 注册账号
3. Account -> API Keys -> Generate
"""

import numpy as np
import tidy3d as td
from tidy3d.plugins.adjoint import JaxSimulation, JaxStructure, JaxBox
import jax
import jax.numpy as jnp
from jax import value_and_grad
import optax
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
from dataclasses import dataclass
import time


@dataclass
class Tidy3DConfig:
    """Tidy3D配置"""
    # 设计区域
    design_region: Tuple[float, float] = (3.0, 3.0)  # μm
    design_thickness: float = 0.22  # μm (SOI 220nm)
    grid_resolution: float = 0.02  # 20nm
    
    # 材料
    n_si: float = 3.476  # Si @ 1550nm
    n_sio2: float = 1.444  # SiO2
    
    # 波长 - 宽带C+L波段
    wavelength_min: float = 1.50  # μm
    wavelength_max: float = 1.65  # μm
    n_wavelengths: int = 16  # 每10nm一个点
    
    # 波导
    wg_width: float = 0.5  # μm
    wg_length: float = 1.5  # μm
    
    # 仿真区域
    sim_size: Tuple[float, float, float] = (8.0, 8.0, 3.0)  # μm
    run_time: float = 3e-12  # s
    
    # 优化
    learning_rate: float = 0.2
    max_iterations: int = 100
    
    # 制造
    min_feature: float = 80e-3  # 80nm
    eta: float = 0.5
    beta_init: float = 1.0
    beta_max: float = 100.0


class Tidy3DRealOptimizer:
    """
    真实Tidy3D优化器
    
    使用Tidy3D cloud进行真实FDTD仿真
    """
    
    def __init__(self, config: Tidy3DConfig = None):
        self.cfg = config or Tidy3DConfig()
        
        # 检查Tidy3D配置
        self._check_tidy3d_setup()
        
        # 计算网格
        self.nx = int(self.cfg.design_region[0] / self.cfg.grid_resolution)
        self.ny = int(self.cfg.design_region[1] / self.cfg.grid_resolution)
        
        print(f"🚀 Tidy3D Real Optimizer")
        print(f"   设计区域: {self.cfg.design_region[0]}×{self.cfg.design_region[1]} μm²")
        print(f"   网格: {self.nx}×{self.ny} ({self.cfg.grid_resolution*1000:.0f}nm分辨率)")
        print(f"   波长范围: {self.cfg.wavelength_min}-{self.cfg.wavelength_max} μm")
        
        # 初始化参数 (1/4区域，利用对称性)
        self.nx_quarter = self.nx // 2
        self.ny_quarter = self.ny // 2
        self.params = jnp.ones((self.nx_quarter, self.ny_quarter)) * 0.5
        
        # 优化器
        self.optimizer = optax.adam(self.cfg.learning_rate)
        self.opt_state = self.optimizer.init(self.params)
        
        # 历史
        self.history = []
        
        # 预创建基础仿真 (不含设计区域)
        self.base_sim = self._create_base_simulation()
    
    def _check_tidy3d_setup(self):
        """检查Tidy3D配置"""
        try:
            import tidy3d as td
            print(f"✅ Tidy3D版本: {td.__version__}")
        except ImportError:
            raise ImportError(
                "Tidy3D未安装。请运行: pip install tidy3d\n"
                "然后设置API key: export TINY3D_API_KEY='你的key'"
            )
        
        # 检查API key
        if not td.web.api_key():
            raise ValueError(
                "Tidy3D API key未设置。\n"
                "获取方法:\n"
                "1. 访问 https://tidy3d.simulation.cloud\n"
                "2. 注册/登录账号\n"
                "3. Account -> API Keys -> Generate\n"
                "4. 设置环境变量: export TINY3D_API_KEY='你的key'"
            )
        
        print("✅ Tidy3D API key已配置")
    
    def _create_base_simulation(self) -> td.Simulation:
        """创建基础仿真 (不含设计区域)"""
        
        # 材料
        si = td.Medium(permittivity=self.cfg.n_si**2)
        sio2 = td.Medium(permittivity=self.cfg.n_sio2**2)
        
        # 衬底
        substrate = td.Structure(
            geometry=td.Box(
                center=(0, 0, -1),
                size=(td.inf, td.inf, 2)
            ),
            medium=sio2
        )
        
        # 输入波导
        wg_input = td.Structure(
            geometry=td.Box(
                center=(-self.cfg.design_region[0]/2 - 0.5, 0, 0),
                size=(1.0, self.cfg.wg_width, self.cfg.design_thickness)
            ),
            medium=si
        )
        
        # 4个输出波导 (45°, 135°, 225°, 315°)
        wg_outputs = []
        angles = [45, 135, 225, 315]
        for angle in angles:
            rad = np.radians(angle)
            r = self.cfg.design_region[0]/2 + 0.5
            x = r * np.cos(rad)
            y = r * np.sin(rad)
            
            wg = td.Structure(
                geometry=td.Box(
                    center=(x, y, 0),
                    size=(self.cfg.wg_width, 1.0, self.cfg.design_thickness)
                ),
                medium=si
            )
            wg_outputs.append(wg)
        
        # 仿真
        sim = td.Simulation(
            size=self.cfg.sim_size,
            grid_spec=td.GridSpec.uniform(dl=self.cfg.grid_resolution),
            structures=[substrate, wg_input] + wg_outputs,
            sources=[],  # 后续添加
            monitors=[],  # 后续添加
            run_time=self.cfg.run_time,
            boundary_spec=td.BoundarySpec.all_sides(boundary=td.PML())
        )
        
        return sim
    
    def create_design_structure(self, params: jnp.ndarray) -> JaxStructure:
        """创建设计区域结构"""
        
        # 扩展对称性
        params_full = self._expand_symmetry(params)
        
        # 创建CustomMedium
        eps_data = self.cfg.n_sio2**2 + params_full**3 * (
            self.cfg.n_si**2 - self.cfg.n_sio2**2
        )
        
        # 坐标
        x = np.linspace(-self.cfg.design_region[0]/2, 
                       self.cfg.design_region[0]/2, 
                       self.nx)
        y = np.linspace(-self.cfg.design_region[1]/2, 
                       self.cfg.design_region[1]/2, 
                       self.ny)
        
        design_medium = td.CustomMedium.from_eps_raw(
            eps_data,
            coords=dict(x=x, y=y, z=[0])
        )
        
        design_structure = JaxStructure(
            geometry=JaxBox(
                center=(0, 0, 0),
                size=(self.cfg.design_region[0], 
                     self.cfg.design_region[1], 
                     self.cfg.design_thickness)
            ),
            medium=design_medium
        )
        
        return design_structure
    
    def _expand_symmetry(self, params: jnp.ndarray) -> jnp.ndarray:
        """4重对称扩展"""
        nx, ny = params.shape
        full = jnp.zeros((2*nx, 2*ny))
        
        full = full.at[:nx, :ny].set(params)
        full = full.at[nx:, :ny].set(jnp.flip(params, axis=0))
        full = full.at[:nx, ny:].set(jnp.flip(params, axis=1))
        full = full.at[nx:, ny:].set(jnp.flip(jnp.flip(params, axis=0), axis=1))
        
        return full
    
    def run_simulation(
        self,
        params: jnp.ndarray,
        wavelength: float
    ) -> td.SimulationData:
        """
        运行仿真
        
        Args:
            params: 设计参数
            wavelength: 波长 (μm)
            
        Returns:
            SimulationData: Tidy3D仿真结果
        """
        freq = td.C_0 / wavelength
        
        # 创建设计结构
        design_struct = self.create_design_structure(params)
        
        # 模式源
        mode_source = td.ModeSource(
            center=(-self.cfg.design_region[0]/2 - 0.3, 0, 0),
            size=(0, 2, 2),
            source_time=td.GaussianPulse(freq0=freq, fwidth=freq/20),
            direction="+",
            mode_spec=td.ModeSpec(num_modes=1),
            mode_index=0
        )
        
        # 输出监视器
        monitors = []
        angles = [45, 135, 225, 315]
        for i, angle in enumerate(angles):
            rad = np.radians(angle)
            r = self.cfg.design_region[0]/2 + 0.3
            x = r * np.cos(rad)
            y = r * np.sin(rad)
            
            monitor = td.ModeMonitor(
                center=(x, y, 0),
                size=(0.5, 0.5, 2),
                freqs=[freq],
                name=f"port_{i}",
                mode_spec=td.ModeSpec(num_modes=1)
            )
            monitors.append(monitor)
        
        # 创建完整仿真
        sim = JaxSimulation(
            size=self.cfg.sim_size,
            grid_spec=td.GridSpec.uniform(dl=self.cfg.grid_resolution),
            structures=list(self.base_sim.structures) + [design_struct],
            sources=[mode_source],
            monitors=monitors,
            run_time=self.cfg.run_time
        )
        
        # 运行仿真 (上传到Tidy3D cloud)
        print(f"   提交仿真到Tidy3D cloud (λ={wavelength*1000:.0f}nm)...")
        data = sim.run()
        
        return data
    
    def calculate_objective(
        self,
        params: jnp.ndarray,
        wavelength: float
    ) -> Tuple[float, Dict]:
        """计算目标函数"""
        
        # 运行仿真
        try:
            data = self.run_simulation(params, wavelength)
        except Exception as e:
            print(f"   ⚠️ 仿真失败: {e}")
            return 1.0, {'transmission': 0, 'error': str(e)}
        
        # 提取S参数
        transmissions = []
        for i in range(4):
            mode_data = data[f"port_{i}"]
            # 计算透射率
            T = np.abs(mode_data.amps.sel(direction="+").values)**2
            transmissions.append(float(T))
        
        # 计算目标
        T_total = sum(transmissions)
        T_avg = T_total / 4
        uniformity = np.std(transmissions)
        
        objective = -T_total + 0.5 * uniformity
        
        metrics = {
            'transmission': T_total,
            'uniformity': uniformity,
            'per_port': transmissions
        }
        
        return objective, metrics
    
    def optimize_iteration(self, iteration: int):
        """单次优化迭代"""
        print(f"\n📊 迭代 {iteration}")
        
        # 多波长优化
        objectives = []
        all_metrics = []
        
        for wl in np.linspace(self.cfg.wavelength_min, 
                             self.cfg.wavelength_max, 
                             3):  # 每次选3个波长
            obj, metrics = self.calculate_objective(self.params, wl)
            objectives.append(obj)
            all_metrics.append(metrics)
        
        # 平均目标
        mean_obj = np.mean(objectives)
        
        # 计算梯度 (有限差分)
        gradient = self._compute_gradient_fd()
        
        # 更新参数
        updates, self.opt_state = self.optimizer.update(gradient, self.opt_state)
        self.params = optax.apply_updates(self.params, updates)
        self.params = jnp.clip(self.params, 0, 1)
        
        # 记录
        result = {
            'iteration': iteration,
            'objective': mean_obj,
            'transmission': np.mean([m['transmission'] for m in all_metrics])
        }
        
        self.history.append(result)
        
        return result
    
    def _compute_gradient_fd(self, epsilon: float = 0.01) -> jnp.ndarray:
        """有限差分计算梯度"""
        # 简化版 - 实际应使用伴随法
        gradient = jnp.zeros_like(self.params)
        
        # 基线
        obj_base, _ = self.calculate_objective(self.params, 1.55)
        
        # 对每个参数
        for i in range(0, self.params.shape[0], 5):  # 每5个采样一个
            for j in range(0, self.params.shape[1], 5):
                params_pert = self.params.at[i, j].add(epsilon)
                obj_pert, _ = self.calculate_objective(params_pert, 1.55)
                gradient = gradient.at[i, j].set((obj_pert - obj_base) / epsilon)
        
        # 插值到完整网格
        from jax.image import resize
        gradient = resize(gradient, self.params.shape, method='bilinear')
        
        return gradient
    
    def optimize(self, n_iterations: int = 10):
        """主优化循环"""
        print(f"\n🚀 开始Tidy3D优化")
        print(f"   迭代次数: {n_iterations}")
        print(f"   注意: 每次迭代需要提交到Tidy3D cloud，可能需要几分钟")
        print()
        
        for i in range(n_iterations):
            result = self.optimize_iteration(i)
            print(f"   目标: {result['objective']:.4f}, "
                  f"透射: {result['transmission']:.3f}")
        
        print("\n✅ 优化完成!")
        return self.params, self.history


if __name__ == "__main__":
    print("=" * 70)
    print("Tidy3D 1x4分光器 - 真实实现")
    print("=" * 70)
    print()
    print("使用步骤:")
    print("1. 安装Tidy3D: pip install tidy3d")
    print("2. 获取API key: https://tidy3d.simulation.cloud")
    print("3. 设置环境变量: export TINY3D_API_KEY='你的key'")
    print("4. 运行: python tidy3d_real_optimizer.py")
    print()
    print("⚠️  注意:")
    print("   - 需要Tidy3D账号和API key")
    print("   - 每次仿真消耗credits")
    print("   - 建议先用少量迭代测试")
    
    # 示例运行 (需要配置好环境)
    # optimizer = Tidy3DRealOptimizer()
    # params, history = optimizer.optimize(n_iterations=5)
