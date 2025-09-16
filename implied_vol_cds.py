"""
CDS隐含波动率计算模块
基于Merton模型从CDS spread反推资产波动率σA
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq, minimize_scalar
import warnings


def calculate_d1_d2(L, sigma_A, T_minus_t, r):
    """
    计算Merton模型中的d1和d2参数
    
    参数:
    L: 杠杆率 (A/D*e^(-r*(T-t)))
    sigma_A: 资产波动率
    T_minus_t: 到期时间 (T-t)
    r: 无风险利率
    
    返回:
    d1, d2: Merton模型参数
    """
    if T_minus_t <= 0:
        raise ValueError("到期时间必须大于0")
    if sigma_A <= 0:
        raise ValueError("资产波动率必须大于0")
    
    sqrt_T_minus_t = np.sqrt(T_minus_t)
    
    d1 = (-np.log(L)) / (sigma_A * sqrt_T_minus_t) + 0.5 * sigma_A * sqrt_T_minus_t
    d2 = d1 - sigma_A * sqrt_T_minus_t
    
    return d1, d2


def calculate_cds_spread(sigma_A, L, T_minus_t, r):
    """
    根据Merton模型计算CDS spread
    
    参数:
    sigma_A: 资产波动率
    L: 杠杆率
    T_minus_t: 到期时间
    r: 无风险利率
    
    返回:
    CDS spread (以小数形式，如0.01表示100bp)
    """
    try:
        d1, d2 = calculate_d1_d2(L, sigma_A, T_minus_t, r)
        
        # 计算CDS spread公式中的组成部分
        N_d2 = norm.cdf(d2)
        N_minus_d1 = norm.cdf(-d1)
        
        # CDS spread公式: s = -1/(T-t) * ln(N(d2) + N(-d1)/L)
        inner_term = N_d2 + N_minus_d1 / L
        
        if inner_term <= 0:
            return np.inf  # 无效的参数组合
            
        spread = -(1 / T_minus_t) * np.log(inner_term)
        
        return spread
        
    except (ValueError, ZeroDivisionError, OverflowError):
        return np.inf


def implied_volatility_from_cds_spread(market_spread, L, T_minus_t, r, 
                                       vol_min=0.01, vol_max=3.0, 
                                       tolerance=1e-8, max_iterations=100):
    """
    从市场CDS spread反推隐含资产波动率σA
    
    参数:
    market_spread: 市场观察到的CDS spread (小数形式)
    L: 杠杆率 (A/D*e^(-r*(T-t)))
    T_minus_t: 到期时间 (年)
    r: 无风险利率 (小数形式)
    vol_min: 搜索的最小波动率 (默认1%)
    vol_max: 搜索的最大波动率 (默认300%)
    tolerance: 收敛容差
    max_iterations: 最大迭代次数
    
    返回:
    implied_vol: 隐含资产波动率
    """
    
    # 输入验证
    if market_spread <= 0:
        raise ValueError("市场CDS spread必须大于0")
    if L <= 0:
        raise ValueError("杠杆率必须大于0")
    if T_minus_t <= 0:
        raise ValueError("到期时间必须大于0")
    if vol_min >= vol_max:
        raise ValueError("vol_min必须小于vol_max")
    
    def objective_function(sigma_A):
        """目标函数: 理论spread与市场spread的差值"""
        theoretical_spread = calculate_cds_spread(sigma_A, L, T_minus_t, r)
        return theoretical_spread - market_spread
    
    # 检查边界条件
    f_min = objective_function(vol_min)
    f_max = objective_function(vol_max)
    
    # 如果在边界上没有根，尝试扩展搜索范围
    if f_min * f_max > 0:
        # 尝试更大的波动率范围
        vol_max_extended = min(vol_max * 2, 5.0)
        f_max_extended = objective_function(vol_max_extended)
        
        if f_min * f_max_extended > 0:
            warnings.warn(f"在波动率范围[{vol_min:.3f}, {vol_max_extended:.3f}]内未找到根。"
                         f"边界函数值: f({vol_min:.3f})={f_min:.6f}, "
                         f"f({vol_max_extended:.3f})={f_max_extended:.6f}")
            
            # 返回使目标函数最小的波动率
            result = minimize_scalar(lambda x: abs(objective_function(x)), 
                                   bounds=(vol_min, vol_max_extended), 
                                   method='bounded')
            return result.x
        else:
            vol_max = vol_max_extended
    
    try:
        # 使用Brent方法求根
        implied_vol = brentq(objective_function, vol_min, vol_max, 
                           xtol=tolerance, maxiter=max_iterations)
        return implied_vol
        
    except ValueError as e:
        raise ValueError(f"数值求解失败: {str(e)}")


def calculate_leverage_ratio(A, D, r, T_minus_t):
    """
    计算杠杆率 L = A / (D * e^(-r*(T-t)))
    
    参数:
    A: 资产价值
    D: 债务面值
    r: 无风险利率
    T_minus_t: 到期时间
    
    返回:
    杠杆率 L
    """
    if D <= 0:
        raise ValueError("债务面值必须大于0")
    if A <= 0:
        raise ValueError("资产价值必须大于0")
    
    discount_factor = np.exp(-r * T_minus_t)
    L = A / (D * discount_factor)
    return L


def cds_spread_to_basis_points(spread):
    """
    将CDS spread从小数转换为基点(bp)
    
    参数:
    spread: CDS spread (小数形式)
    
    返回:
    基点形式的spread
    """
    return spread * 10000


def basis_points_to_cds_spread(bp):
    """
    将基点转换为CDS spread小数形式
    
    参数:
    bp: 基点形式的spread
    
    返回:
    小数形式的CDS spread
    """
    return bp / 10000


# 示例使用函数
def example_usage():
    """
    使用示例
    """
    print("=== CDS隐含波动率计算示例 ===")
    
    # 示例1: 正向测试 - 先用已知波动率计算spread，然后反推
    print("\n--- 示例1: 正向验证测试 ---")
    known_vol = 0.25  # 已知波动率25%
    A = 100  # 资产价值
    D = 70   # 债务面值
    r = 0.03  # 3%无风险利率
    T_minus_t = 3.0  # 3年到期
    
    L = calculate_leverage_ratio(A, D, r, T_minus_t)
    print(f"杠杆率 L: {L:.4f}")
    
    # 用已知波动率计算理论spread
    theoretical_spread = calculate_cds_spread(known_vol, L, T_minus_t, r)
    theoretical_spread_bp = cds_spread_to_basis_points(theoretical_spread)
    print(f"已知波动率: {known_vol:.4f} ({known_vol*100:.2f}%)")
    print(f"理论CDS spread: {theoretical_spread_bp:.2f} bp")
    
    # 反推波动率
    try:
        implied_vol = implied_volatility_from_cds_spread(
            theoretical_spread, L, T_minus_t, r
        )
        
        print(f"反推隐含波动率: {implied_vol:.4f} ({implied_vol*100:.2f}%)")
        print(f"误差: {abs(known_vol - implied_vol):.6f}")
        
    except Exception as e:
        print(f"反推失败: {e}")
    
    # 示例2: 实际市场数据示例
    print("\n--- 示例2: 市场数据示例 ---")
    market_spread_bp = 250  # 150基点
    market_spread = basis_points_to_cds_spread(market_spread_bp)
    A2 = 100  # 资产价值
    D2 = 150   # 债务面值
    r2 = 0.045  # 2.5%无风险利率
    T_minus_t2 = 5.0  # 2年到期
    
    L2 = calculate_leverage_ratio(A2, D2, r2, T_minus_t2)
    print(f"杠杆率 L: {L2:.4f}")
    print(f"市场CDS spread: {market_spread_bp} bp")
    
    try:
        implied_vol2 = implied_volatility_from_cds_spread(
            market_spread, L2, T_minus_t2, r2
        )
        
        print(f"隐含资产波动率: {implied_vol2:.4f} ({implied_vol2*100:.2f}%)")
        
        # 验证: 用计算出的波动率反推spread
        verify_spread = calculate_cds_spread(implied_vol2, L2, T_minus_t2, r2)
        verify_spread_bp = cds_spread_to_basis_points(verify_spread)
        
        print(f"验证 - 理论CDS spread: {verify_spread_bp:.2f} bp")
        print(f"差异: {abs(market_spread_bp - verify_spread_bp):.4f} bp")
        
    except Exception as e:
        print(f"计算失败: {e}")




def solve_implied_vol_simple(market_spread_bp, asset_value, debt_value, 
                             risk_free_rate, time_to_maturity):
    """
    简化接口：从CDS spread (基点) 计算隐含波动率
    
    参数:
    market_spread_bp: 市场CDS spread (基点, 如150表示150bp)
    asset_value: 资产价值
    debt_value: 债务面值
    risk_free_rate: 无风险利率 (小数形式, 如0.03表示3%)
    time_to_maturity: 到期时间 (年)
    
    返回:
    字典包含: {'implied_vol': 隐含波动率, 'leverage': 杠杆率, 'status': 状态信息}
    """
    try:
        # 转换输入
        market_spread = basis_points_to_cds_spread(market_spread_bp)
        leverage = calculate_leverage_ratio(asset_value, debt_value, 
                                          risk_free_rate, time_to_maturity)
        
        # 计算隐含波动率
        implied_vol = implied_volatility_from_cds_spread(
            market_spread, leverage, time_to_maturity, risk_free_rate
        )
        
        # 验证结果
        verify_spread = calculate_cds_spread(implied_vol, leverage, 
                                           time_to_maturity, risk_free_rate)
        verify_spread_bp = cds_spread_to_basis_points(verify_spread)
        error_bp = abs(market_spread_bp - verify_spread_bp)
        
        status = "成功"
        if error_bp > 1.0:  # 误差超过1bp
            status = f"警告: 验证误差较大 ({error_bp:.2f}bp)"
            
        return {
            'implied_vol': implied_vol,
            'leverage': leverage,
            'verification_spread_bp': verify_spread_bp,
            'error_bp': error_bp,
            'status': status
        }
        
    except Exception as e:
        return {
            'implied_vol': None,
            'leverage': None,
            'verification_spread_bp': None,
            'error_bp': None,
            'status': f"计算失败: {str(e)}"
        }


if __name__ == "__main__":
    example_usage()
