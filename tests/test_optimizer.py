"""
单元测试
"""

import numpy as np
import pytest
from srtp_splitter import SymmetricSplitterOptimizer, ManufacturingConstraints
from srtp_splitter.utils import InitializationStrategies, ObjectiveFunctions


class TestInitializationStrategies:
    """测试初始化策略"""
    
    def test_random_uniform_shape(self):
        init = InitializationStrategies.random_uniform(10, 20)
        assert init.shape == (10, 20)
        assert np.all(init >= 0.3) and np.all(init <= 0.7)
    
    def test_constant_shape(self):
        init = InitializationStrategies.constant(10, 20, value=0.6)
        assert init.shape == (10, 20)
        assert np.allclose(init, 0.6)
    
    def test_radial_gradient_shape(self):
        init = InitializationStrategies.radial_gradient(10, 20)
        assert init.shape == (10, 20)
        assert np.all(init >= 0) and np.all(init <= 1)


class TestSymmetricSplitterOptimizer:
    """测试对称优化器"""
    
    def test_initialization(self):
        opt = SymmetricSplitterOptimizer(
            design_region_size=(3.0, 3.0),
            grid_resolution=0.1,
            wavelength_range=(1.45, 1.65),
            n_wavelengths=11,
            symmetry=True
        )
        assert opt.use_symmetry is True
        assert opt.params.shape[0] == 15  # 3.0/0.1/2
    
    def test_expand_symmetry(self):
        opt = SymmetricSplitterOptimizer(
            design_region_size=(2.0, 2.0),
            grid_resolution=0.5,
            wavelength_range=(1.45, 1.65),
            n_wavelengths=5,
            symmetry=True
        )
        quarter = np.random.rand(2, 2)
        full = opt.expand_symmetry(quarter)
        assert full.shape == (4, 4)


class TestManufacturingConstraints:
    """测试制造约束"""
    
    def test_initialization(self):
        mc = ManufacturingConstraints(
            min_feature_size=80e-9,
            grid_size=20e-9
        )
        assert mc.rmin == 80e-9
        assert mc.dx == 20e-9
    
    def test_density_filter(self):
        mc = ManufacturingConstraints(min_feature_size=100e-9, grid_size=20e-9)
        params = np.random.rand(10, 10)
        filtered = mc.density_filter(params)
        assert filtered.shape == params.shape


class TestObjectiveFunctions:
    """测试目标函数工具"""
    
    def test_calculate_bandwidth(self):
        wavelengths = np.linspace(1.45, 1.65, 21)
        transmissions = np.ones(21) * 0.95
        bw = ObjectiveFunctions.calculate_bandwidth(wavelengths, transmissions)
        assert bw > 0
    
    def test_calculate_imbalance(self):
        powers = [0.24, 0.25, 0.25, 0.26]
        imbalance = ObjectiveFunctions.calculate_imbalance(powers)
        assert imbalance > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
