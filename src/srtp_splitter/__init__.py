"""
SRTP Splitter - 1x4光功率分光器逆向设计优化器

基于Tidy3D的拓扑优化实现，利用4重对称性实现4倍加速。
"""

from .optimizer import SymmetricSplitterOptimizer
from .manufacturing import ManufacturingConstraints
from .utils import InitializationStrategies, ObjectiveFunctions

__version__ = "1.0.0"
__author__ = "SRTP Project"

__all__ = [
    "SymmetricSplitterOptimizer",
    "ManufacturingConstraints", 
    "InitializationStrategies",
    "ObjectiveFunctions",
]
