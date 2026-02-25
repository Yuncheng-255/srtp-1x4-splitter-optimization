#!/usr/bin/env python3
"""
OptimizationUtils - 优化工具集
基于研究和最佳实践的实用技巧
"""

import numpy as np
import jax.numpy as jnp
from scipy import signal
from typing import Tuple, List


class InitializationStrategies:
    """初始化策略 - 影响收敛速度和最终性能"""
    
    @staticmethod
    def random_uniform(nx: int, ny: int, seed: int = 42) -> np.ndarray:
        """随机均匀初始化 - 最基础"""
        np.random.seed(seed)
        return np.random.uniform(0.3, 0.7, (nx, ny))
    
    @staticmethod
    def constant(nx: int, ny: int, value: float = 0.5) -> np.ndarray:
        """常数初始化 - 保守但慢"""
        return np.ones((nx, ny)) * value
    
    @staticmethod
    def radial_gradient(nx: int, ny: int) -> np.ndarray:
        """径向梯度 - 中心高，边缘低"""
        x = np.linspace(-1, 1, nx)
        y = np.linspace(-1, 1, ny)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        return 0.8 * np.exp(-2 * R**2) + 0.1
    
    @staticmethod
    def y_branch_like(nx: int, ny: int) -> np.ndarray:
        """Y分支启发式 - 模拟分支结构"""
        params = np.zeros((nx, ny))
        
        # 输入波导
        params[:, ny//2-2:ny//2+2] = 0.9
        
        # 分支区域 (V形)
        for i in range(nx//2, nx):
            width = 2 + int((i - nx//2) / (nx//2) * (ny//4))
            params[i, ny//2-width:ny//2+width] = 0.9
        
        return params
    
    @staticmethod
    def pretrained_small(nx: int, ny: int) -> np.ndarray:
        """从小规模预训练初始化 - 最佳策略"""
        # 先在小网格(10x10)上预优化
        # 然后插值到目标网格
        small = np.random.rand(10, 10)
        from scipy.ndimage import zoom
        return zoom(small, (nx/10, ny/10), order=1)


class ConstraintEnforcement:
    """约束强制执行 - 确保物理可实现"""
    
    @staticmethod
    def filter_kernel(radius: int) -> np.ndarray:
        """创建滤波核"""
        size = 2 * radius + 1
        x = np.linspace(-radius, radius, size)
        y = np.linspace(-radius, radius, size)
        X, Y = np.meshgrid(x, y)
        kernel = np.exp(-(X**2 + Y**2) / (2 * radius**2))
        return kernel / kernel.sum()
    
    @staticmethod
    def density_filter(params: np.ndarray, radius: int) -> np.ndarray:
        """密度滤波 - 消除棋盘格"""
        from scipy.signal import convolve2d
        kernel = ConstraintEnforcement.filter_kernel(radius)
        return convolve2d(params, kernel, mode='same', boundary='symm')
    
    @staticmethod
    def heaviside_projection(
        params: np.ndarray,
        beta: float,
        eta: float = 0.5
    ) -> np.ndarray:
        """Heaviside投影"""
        return (np.tanh(beta * eta) + np.tanh(beta * (params - eta))) / \
               (np.tanh(beta * eta) + np.tanh(beta * (1 - eta)))
    
    @staticmethod
    def connectivity_check(structure: np.ndarray) -> bool:
        """检查连通性"""
        from scipy import ndimage
        labeled, num_features = ndimage.label(structure > 0.5)
        return num_features == 1  # 应该只有一个连通区域


class LearningRateSchedules:
    """学习率调度 - 影响收敛"""
    
    @staticmethod
    def constant(lr: float, iteration: int, max_iter: int) -> float:
        """恒定学习率"""
        return lr
    
    @staticmethod
    def exponential_decay(
        lr_init: float,
        iteration: int,
        max_iter: int,
        decay_rate: float = 0.99
    ) -> float:
        """指数衰减"""
        return lr_init * (decay_rate ** iteration)
    
    @staticmethod
    def cosine_annealing(
        lr_init: float,
        iteration: int,
        max_iter: int,
        lr_min: float = 1e-4
    ) -> float:
        """余弦退火 - 推荐"""
        import math
        progress = iteration / max_iter
        return lr_min + 0.5 * (lr_init - lr_min) * (1 + math.cos(progress * math.pi))
    
    @staticmethod
    def warm_restart(
        lr_init: float,
        iteration: int,
        max_iter: int,
        restart_period: int = 50
    ) -> float:
        """热重启 - 跳出局部最优"""
        import math
        progress = (iteration % restart_period) / restart_period
        return lr_init * 0.5 * (1 + math.cos(progress * math.pi))


class ObjectiveWeighting:
    """目标函数加权策略"""
    
    @staticmethod
    def adaptive_weights(
        transmission: float,
        uniformity: float,
        iteration: int
    ) -> Tuple[float, float]:
        """
        自适应权重
        
        早期: 重视透射率
        后期: 重视均匀性
        """
        progress = min(iteration / 100, 1.0)
        
        # 透射率权重随时间降低
        w_t = 1.0 - 0.3 * progress
        
        # 均匀性权重随时间增加
        w_u = 0.5 + 0.5 * progress
        
        return w_t, w_u
    
    @staticmethod
    def performance_based_weights(
        current_performance: Dict[str, float]
    ) -> Dict[str, float]:
        """基于当前性能的动态权重"""
        T = current_performance.get('transmission', 0.5)
        U = current_performance.get('uniformity', 1.0)
        
        # 如果透射率低，增加透射率权重
        w_t = 1.0 + (1.0 - T) * 2
        
        # 如果均匀性差，增加均匀性权重
        w_u = 0.5 + U * 2
        
        return {'transmission': w_t, 'uniformity': w_u}


class MultiScaleOptimization:
    """多尺度优化 - 从粗到精"""
    
    @staticmethod
    def create_pyramid(params: np.ndarray, levels: int = 3) -> List[np.ndarray]:
        """创建参数金字塔"""
        from scipy.ndimage import zoom
        
        pyramid = [params]
        for _ in range(levels - 1):
            coarse = zoom(pyramid[-1], 0.5, order=1)
            pyramid.append(coarse)
        
        return pyramid[::-1]  # 从粗到精
    
    @staticmethod
    def upsample_params(
        coarse_params: np.ndarray,
        target_shape: Tuple[int, int]
    ) -> np.ndarray:
        """上采样参数"""
        from scipy.ndimage import zoom
        ratio = (target_shape[0] / coarse_params.shape[0],
                target_shape[1] / coarse_params.shape[1])
        return zoom(coarse_params, ratio, order=1)


class ConvergenceMonitoring:
    """收敛监控 - 智能停止"""
    
    def __init__(self, patience: int = 20, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.best_value = float('inf')
        self.counter = 0
        self.history = []
    
    def check(self, value: float) -> Tuple[bool, str]:
        """
        检查是否收敛
        
        Returns:
            (should_stop, reason)
        """
        self.history.append(value)
        
        # 改进足够大
        if value < self.best_value - self.min_delta:
            self.best_value = value
            self.counter = 0
            return False, "improving"
        
        # 没有改进
        self.counter += 1
        
        if self.counter >= self.patience:
            return True, f"no improvement for {self.patience} iterations"
        
        # 检查震荡
        if len(self.history) >= 10:
            recent_std = np.std(self.history[-10:])
            if recent_std < self.min_delta:
                return True, "converged (stable)"
        
        return False, "waiting"


class PerformanceMetrics:
    """性能指标计算"""
    
    @staticmethod
    def calculate_bandwidth(
        wavelengths: np.ndarray,
        transmissions: np.ndarray,
        threshold: float = 0.9
    ) -> float:
        """计算带宽 (nm)"""
        T_max = np.max(transmissions)
        T_threshold = T_max * threshold
        
        above_threshold = transmissions >= T_threshold
        indices = np.where(above_threshold)[0]
        
        if len(indices) == 0:
            return 0.0
        
        wl_min = wavelengths[indices[0]]
        wl_max = wavelengths[indices[-1]]
        
        return (wl_max - wl_min) * 1000  # nm
    
    @staticmethod
    def calculate_imbalance(powers: List[float]) -> float:
        """计算分光不平衡度 (dB)"""
        P_max = max(powers)
        P_min = min(powers)
        return 10 * np.log10(P_max / P_min) if P_min > 0 else float('inf')
    
    @staticmethod
    def calculate_fom(transmission: float, uniformity: float) -> float:
        """计算品质因数 (Figure of Merit)"""
        # 综合考虑透射率和均匀性
        return transmission / (1 + uniformity)


if __name__ == "__main__":
    print("=" * 60)
    print("优化工具集 - 最佳实践")
    print("=" * 60)
    
    # 测试初始化策略
    print("\n初始化策略对比:")
    for name, func in [
        ("Random", InitializationStrategies.random_uniform),
        ("Constant", InitializationStrategies.constant),
        ("Radial", InitializationStrategies.radial_gradient),
        ("Y-branch", InitializationStrategies.y_branch_like)
    ]:
        init = func(20, 20)
        print(f"  {name}: mean={init.mean():.3f}, std={init.std():.3f}")
    
    # 测试学习率调度
    print("\n学习率调度对比 (前10步):")
    for name, func in [
        ("Constant", LearningRateSchedules.constant),
        ("Exponential", LearningRateSchedules.exponential_decay),
        ("Cosine", LearningRateSchedules.cosine_annealing)
    ]:
        lrs = [func(0.1, i, 100) for i in range(10)]
        print(f"  {name}: {[f'{lr:.4f}' for lr in lrs]}")
    
    print("\n工具集加载完成!")
