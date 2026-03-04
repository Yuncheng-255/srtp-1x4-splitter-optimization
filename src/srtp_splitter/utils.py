#!/usr/bin/env python3
"""
Optimization Utilities - 优化工具集

提供初始化策略、学习率调度、性能指标等工具函数。
"""

import numpy as np
from scipy import ndimage
from typing import Tuple, List, Dict, Optional


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


class LearningRateSchedules:
    """学习率调度 - 影响收敛"""
    
    @staticmethod
    def constant(lr: float, iteration: int, max_iter: int) -> float:
        """恒定学习率"""
        return lr
    
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


class ConvergenceMonitor:
    """收敛监控 - 智能停止"""
    
    def __init__(self, patience: int = 20, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.best_value = float('inf')
        self.counter = 0
        self.history = []
    
    def check(self, value: float) -> Tuple[bool, str]:
        """检查是否收敛"""
        self.history.append(value)
        
        if value < self.best_value - self.min_delta:
            self.best_value = value
            self.counter = 0
            return False, "improving"
        
        self.counter += 1
        if self.counter >= self.patience:
            return True, f"no improvement for {self.patience} iterations"
        
        return False, "waiting"


class ObjectiveFunctions:
    """目标函数工具"""
    
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
        
        return (wl_max - wl_min) * 1000
    
    @staticmethod
    def calculate_imbalance(powers: List[float]) -> float:
        """计算分光不平衡度 (dB)"""
        P_max = max(powers)
        P_min = min(powers)
        return 10 * np.log10(P_max / P_min) if P_min > 0 else float('inf')
