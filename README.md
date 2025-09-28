# CDS implied solver

## 概述

这个求解器实现了从给定的 CDS spread 反解 asset volatility (sigma)的功能，类似于 Black-Scholes 期权定价模型中的隐含波动率计算。

## 文件结构

- `implied_vol_solver.py` - 主要的求解器类
- `cds2vol.py` - 后续大规模接入数据接口的 py（尚未实现）
- `demo.ipynb ` - 流程参考 notebook
- `README.md` - 本说明文档

## 主要功能

### CDSImpliedVolatilitySolver 类

该类完全基于`cgm_core.py`中的函数结构实现，确保计算的一致性。

#### 主要方法

1. **calculate_survival_probability(S, t, sigma)**

   - 计算生存概率 P(t)
   - 基于公式 2.11

2. **calculate_cds_spread_continuous(S, sigma, R)**

   - 计算 CDS 连续支付价差
   - 基于公式 2.15

3. **solve_implied_volatility(target_cds_spread, R, method)**
   - 主要求解接口
   - 支持两种方法：'brent' 和 'minimize'
   - 自动处理基点转换

#### 数值求解方法

1. **Brent 方法** (`method='brent'`)

   - 使用 scipy.optimize.brentq
   - 需要函数在区间端点异号
   - 通常更快、更稳定

2. **最小化方法** (`method='minimize'`)
   - 使用 scipy.optimize.minimize_scalar
   - 最小化|模型价差 - 市场价差|
   - 作为备选方案

## 使用方法

### 基本用法

```python
from implied_vol_solver import CDSImpliedVolatilitySolver

# 1. 设置参数
S = 100.0        # 资产价值
D = 150.0        # 债务面值
t = 5.0          # CDS期限
r = 0.05         # 无风险利率
R = 0.4          # 回收率
L = 0.5          # 债务回收率
lamb = 0.3       # 违约壁垒参数

# 2. 创建求解器
solver = CDSImpliedVolatilitySolver(S, D, t, r, L, lamb)

# 3. 求解隐含波动率
market_cds_spread = 250  # 250bp
implied_vol = solver.solve_implied_volatility(
    market_cds_spread,
    R=R,
    method='brent'
)

print(f"隐含波动率: {implied_vol:.6f}")
```

### 参数说明

- **S**: 当前资产价值 (股价代理)
- **D**: 债务面值
- **t**: CDS 期限 (年)
- **r**: 无风险利率
- **R**: 回收率 (违约时的回收比例)
- **L**: 全局债务回收率
- **lamb**: 违约壁垒标准差参数

### 输入格式

- CDS 价差可以是基点形式 (如 250) 或小数形式 (如 0.025)
- 程序会自动识别并转换 (>10 视为基点)

## 验证和测试

### 一致性测试

求解器包含一致性测试功能：

```python
# 测试：给定波动率计算价差，然后反解波动率
test_consistency_with_cgm_core()
```

该测试验证：

1. 使用已知波动率计算 CDS 价差
2. 从该价差反解波动率
3. 检查误差是否在容忍范围内

### 运行示例

```bash
cd /home/yicheng/credit
python implied_vol_solver.py      # 运行完整测试
```
