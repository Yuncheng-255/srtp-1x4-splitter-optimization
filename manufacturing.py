#!/usr/bin/env python3
"""
制造约束处理模块
确保优化结果可制造
"""

import numpy as np
from scipy import ndimage
from typing import Tuple, Optional


class ManufacturingConstraints:
    """
    制造约束处理
    
    功能:
    1. 最小线宽约束
    2. 最小间隙约束  
    3. 连通性检查
    4. 边缘平滑
    """
    
    def __init__(
        self,
        min_feature_size: float = 80e-9,  # 80nm
        grid_size: float = 20e-9,  # 20nm
        eta: float = 0.5
    ):
        """
        初始化制造约束
        
        Args:
            min_feature_size: 最小特征尺寸 (m)
            grid_size: 网格尺寸 (m)
            eta: 投影阈值
        """
        self.rmin = min_feature_size
        self.dx = grid_size
        self.eta = eta
        
        # 滤波半径 (像素)
        self.filter_radius = int(self.rmin / self.dx)
        print(f"制造约束初始化:")
        print(f"  最小特征尺寸: {min_feature_size*1e9:.0f}nm")
        print(f"  网格尺寸: {grid_size*1e9:.0f}nm")
        print(f"  滤波半径: {self.filter_radius} 像素")
    
    def density_filter(self, params: np.ndarray) -> np.ndarray:
        """
        密度滤波 - 消除棋盘格，保证最小线宽
        
        使用高斯滤波平滑密度场
        
        Args:
            params: 原始密度场 (nx, ny), 值域[0,1]
            
        Returns:
            filtered: 滤波后的密度场
        """
        if self.filter_radius <= 0:
            return params
        
        # 高斯滤波
        filtered = ndimage.gaussian_filter(
            params, 
            sigma=self.filter_radius
        )
        
        return filtered
    
    def heaviside_projection(
        self, 
        params: np.ndarray, 
        beta: float = 8
    ) -> np.ndarray:
        """
        Heaviside投影 - 促进二值化
        
        将灰度密度场投影到接近0或1
        
        Args:
            params: 输入密度场
            beta: 投影陡峭度
            eta: 投影阈值
            
        Returns:
            projected: 投影后的场
        """
        return (np.tanh(beta * self.eta) + 
                np.tanh(beta * (params - self.eta))) / \
               (np.tanh(beta * self.eta) + np.tanh(beta * (1 - self.eta)))
    
    def erosion_dilation_cycle(
        self, 
        structure: np.ndarray, 
        n_cycles: int = 1
    ) -> np.ndarray:
        """
        腐蚀-膨胀循环 - 消除小岛和细缝
        
        Args:
            structure: 二值结构 (0或1)
            n_cycles: 循环次数
            
        Returns:
            cleaned: 清理后的结构
        """
        binary = (structure > 0.5).astype(np.uint8)
        
        for _ in range(n_cycles):
            # 腐蚀 (消除小岛)
            eroded = ndimage.binary_erosion(binary)
            # 膨胀 (恢复原尺寸)
            dilated = ndimage.binary_dilation(eroded)
            binary = dilated.astype(np.uint8)
        
        return binary.astype(float)
    
    def check_connectivity(
        self, 
        structure: np.ndarray,
        input_port: Tuple[int, int],
        output_ports: List[Tuple[int, int]]
    ) -> Tuple[bool, np.ndarray]:
        """
        检查连通性
        
        确保输入和输出端口连通
        
        Args:
            structure: 二值结构
            input_port: 输入端口位置 (i, j)
            output_ports: 输出端口位置列表
            
        Returns:
            is_connected: 是否全部连通
            labeled: 标记的连通区域
        """
        binary = (structure > 0.5).astype(np.uint8)
        
        # 标记连通区域
        labeled, num_features = ndimage.label(binary)
        
        # 获取输入端口所在区域
        input_label = labeled[input_port]
        
        # 检查所有输出端口是否在同一区域
        connected = True
        for port in output_ports:
            if labeled[port] != input_label:
                connected = False
                break
        
        return connected, labeled
    
    def measure_feature_sizes(self, structure: np.ndarray) -> dict:
        """
        测量特征尺寸
        
        分析结构中的线宽和间隙
        
        Args:
            structure: 二值结构
            
        Returns:
            stats: 尺寸统计
        """
        binary = (structure > 0.5).astype(np.uint8)
        
        # 骨架提取
        from skimage import morphology
        skeleton = morphology.skeletonize(binary)
        
        # 距离变换 (到最近边界的距离)
        distance = ndimage.distance_transform_edt(binary)
        
        # 在骨架上的距离 = 半线宽
        line_widths = 2 * distance[skeleton]
        
        # 反转后的距离变换 = 间隙
        inverted = 1 - binary
        gap_distance = ndimage.distance_transform_edt(inverted)
        gap_sizes = 2 * gap_distance[skeleton]
        
        stats = {
            'min_line_width': np.min(line_widths) * self.dx * 1e9,  # nm
            'mean_line_width': np.mean(line_widths) * self.dx * 1e9,
            'min_gap': np.min(gap_sizes) * self.dx * 1e9,
            'mean_gap': np.mean(gap_sizes) * self.dx * 1e9,
        }
        
        return stats
    
    def apply_all_constraints(
        self, 
        params: np.ndarray,
        iteration: int,
        max_iter: int
    ) -> np.ndarray:
        """
        应用所有制造约束
        
        完整的约束处理流程:
        1. 密度滤波
        2. 渐进投影
        3. 二值化
        4. 清理 (可选)
        
        Args:
            params: 原始密度场
            iteration: 当前迭代
            max_iter: 最大迭代
            
        Returns:
            constrained: 约束处理后的场
        """
        # 1. 密度滤波
        filtered = self.density_filter(params)
        
        # 2. 渐进投影
        progress = iteration / max_iter
        beta = 1 + 31 * progress
        projected = self.heaviside_projection(filtered, beta=beta)
        
        # 3. 最终强投影 (最后阶段)
        if progress > 0.8:
            projected = self.heaviside_projection(projected, beta=32)
        
        return projected
    
    def validate_manufacturability(
        self, 
        structure: np.ndarray,
        verbose: bool = True
    ) -> bool:
        """
        验证可制造性
        
        Args:
            structure: 最终结构
            verbose: 是否打印详情
            
        Returns:
            is_valid: 是否可制造
        """
        stats = self.measure_feature_sizes(structure)
        
        min_feature_ok = stats['min_line_width'] >= self.rmin * 1e9
        min_gap_ok = stats['min_gap'] >= self.rmin * 1e9
        
        is_valid = min_feature_ok and min_gap_ok
        
        if verbose:
            print("\n可制造性验证:")
            print(f"  最小线宽: {stats['min_line_width']:.1f}nm "
                  f"({'✓' if min_feature_ok else '✗'} 要求≥{self.rmin*1e9:.0f}nm)")
            print(f"  最小间隙: {stats['min_gap']:.1f}nm "
                  f"({'✓' if min_gap_ok else '✗'} 要求≥{self.rmin*1e9:.0f}nm)")
            print(f"  平均线宽: {stats['mean_line_width']:.1f}nm")
            print(f"  平均间隙: {stats['mean_gap']:.1f}nm")
            print(f"\n  结论: {'可制造 ✓' if is_valid else '不可制造 ✗'}")
        
        return is_valid


if __name__ == "__main__":
    # 测试制造约束
    print("=" * 60)
    print("制造约束模块测试")
    print("=" * 60)
    
    # 创建约束处理器
    mc = ManufacturingConstraints(
        min_feature_size=80e-9,  # 80nm
        grid_size=20e-9  # 20nm
    )
    
    # 生成测试结构
    np.random.seed(42)
    test_params = np.random.rand(50, 50)
    
    # 应用约束
    print("\n应用制造约束...")
    constrained = mc.apply_all_constraints(test_params, 50, 100)
    
    # 验证
    mc.validate_manufacturability(constrained)
    
    print("\n制造约束模块测试完成!")
