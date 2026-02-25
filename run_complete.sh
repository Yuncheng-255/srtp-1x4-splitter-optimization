#!/bin/bash
# SRTP 1x4分光器完整解决方案

echo "======================================"
echo "SRTP 1x4分光器 - 完整执行脚本"
echo "======================================"
echo ""

# 检查conda环境
if ! command -v conda &> /dev/null; then
    echo "❌ conda未安装"
    exit 1
fi

# 激活base环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate base

# 检查Python版本
PYTHON_VER=$(python --version 2>&1 | awk '{print $2}')
echo "✅ Python版本: $PYTHON_VER"

# 检查Tidy3D
if ! python -c "import tidy3d" 2>/dev/null; then
    echo "📦 安装Tidy3D..."
    pip install tidy3d -q
fi

echo "✅ Tidy3D已安装"

# 配置API Key
export TINY3D_API_KEY='6BEU36edpFWSDFrQWo2IE6h9PRyJWvTzEZSVs7NF8mFgafju'
mkdir -p ~/.config/tidy3d
echo "apikey = '$TINY3D_API_KEY'" > ~/.config/tidy3d/config

echo "✅ API Key已配置"
echo ""

# 运行主程序
echo "🚀 开始执行1x4分光器仿真..."
echo "(这将运行5个不同配置的仿真，大约需要15-20分钟)"
echo ""

cd ~/.openclaw/workspace/srtp_splitter
python3 auto_optimize.py 2>&1 | tee optimization_log.txt

echo ""
echo "======================================"
echo "✅ 执行完成!"
echo "======================================"
echo ""
echo "查看结果:"
echo "  - optimization_results.json"
echo "  - optimization_log.txt"
echo "  - Tidy3D Cloud: https://tidy3d.simulation.cloud/workbench"
echo ""
