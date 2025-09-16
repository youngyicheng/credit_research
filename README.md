# Credit Grade Model - CDS定价系统

基于学术论文公式实现的信用违约互换(CDS)定价计算系统。

## 核心功能

### 1. 生存概率计算 (公式 2.11)
```
P(t) = Φ(-At/2 + log(d)/√At) - d * Φ(-At/2 - log(d)/√At)
```
其中：
- `d = V₀*e^(λ²)/LD`
- `At = σ²*t + λ²`

### 2. CDS价差计算 (公式 2.15)
```
c* = r(1-R) * [1-P(0)+e^(rξ)(G(t+ξ)-G(ξ))] / [P(0)-P(t)e^(-rt)-e^(rξ)(G(t+ξ)-G(ξ))]
```

### 3. 隐含波动率计算
从市场CDS价差反推资产波动率

## 文件结构

```
credit/
├── cgm_core.py          # 核心计算模块
├── data_input.py        # 数据输入接口
├── run_examples.py      # 示例和演示
├── simple_requirements.txt  # 依赖包
└── README.md           # 说明文档
```

## 快速开始

### 1. 安装依赖
```bash
pip install -r simple_requirements.txt
```

### 2. 基本使用
```python
from data_input import CDSCalculator

# 创建计算器
calculator = CDSCalculator()

# 单个案例计算
result = calculator.calculate_single_case(
    asset_value=100.0,      # 资产价值
    asset_volatility=0.25,  # 25% 波动率
    debt_amount=60.0,       # 债务金额
    time_years=5.0,         # 5年期
    risk_free_rate=0.05     # 5% 利率
)

print(f"违约概率: {result['results']['default_probability_pct']}")
print(f"CDS价差: {result['results']['cds_spread_bps']:.1f} bps")
```

### 3. 从市场价差计算隐含波动率
```python
market_result = calculator.calculate_from_market_cds(
    asset_value=100.0,
    debt_amount=60.0,
    time_years=5.0,
    risk_free_rate=0.05,
    market_cds_spread=0.02  # 市场CDS价差 2%
)

implied_vol = market_result['implied_volatility']['implied_asset_volatility']
print(f"隐含波动率: {implied_vol:.1%}")
```

### 4. 批量计算
```python
batch_data = [
    {
        'asset_value': 100, 'asset_volatility': 0.2, 'debt_amount': 50,
        'time_years': 3, 'risk_free_rate': 0.05
    },
    {
        'asset_value': 80, 'asset_volatility': 0.3, 'debt_amount': 60,
        'time_years': 5, 'risk_free_rate': 0.04
    }
]

results_df = calculator.batch_calculate(batch_data)
print(results_df)
```

## 运行示例

```bash
python run_examples.py
```

示例包括：
1. 基本计算功能
2. 敏感性分析
3. 隐含波动率计算
4. 公司比较分析
5. CDS期限结构
6. 批量处理

## 参数说明

### 输入参数
- `asset_value`: 资产价值 (V₀)，通常用股票价格表示
- `asset_volatility`: 资产波动率 (σ)，年化波动率
- `debt_amount`: 债务金额 (LD)
- `time_years`: 期限 (t)，以年为单位
- `risk_free_rate`: 无风险利率 (r)
- `recovery_rate`: 回收率 (R)，默认0.4 (40%)
- `barrier_param`: 违约壁垒参数 (λ)，默认0.2

### 输出结果
- `survival_probability`: 生存概率
- `default_probability`: 违约概率
- `cds_spread`: CDS价差
- `cds_spread_bps`: CDS价差(基点)
- `delta`: 资产价值敏感性
- `vega`: 波动率敏感性
- `theta`: 时间衰减

## 公式参考

本实现基于以下学术论文的公式：
- 公式 2.11: 生存概率计算
- 公式 2.12: d参数定义
- 公式 2.13: At参数定义
- 公式 2.15: CDS价差计算
- 公式 2.16: G函数定义

## 注意事项

1. **数值稳定性**: 模型在极端参数下可能存在数值不稳定问题
2. **参数范围**: 建议波动率在1%-300%范围内，期限在0.01-30年
3. **简化假设**: G函数使用了简化实现，完整版本可根据需要扩展
4. **计算精度**: 结果精度取决于数值方法的收敛性

## 扩展功能

可以基于核心模块扩展：
- Web API接口
- 图形用户界面
- 更复杂的G函数实现
- 蒙特卡洛模拟
- 历史数据回测
