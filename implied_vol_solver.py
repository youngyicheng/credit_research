"""
CDS隐含波动率求解器
从给定的CDS spread反解asset volatility (sigma)

基于cgm_core.py中的函数结构实现
类似于Black-Scholes期权定价模型中的隐含波动率计算
使用数值方法求解非线性方程: CDS_spread_market = CDS_spread_model(sigma)

主要功能:
1. 从市场CDS价差反解资产波动率
2. 直接使用cgm_core.py中的计算函数
3. 提供多种数值求解方法 (Brent, Newton-Raphson等)
4. 鲁棒性检验和边界条件处理

作者: Yicheng
日期: 2025
"""

import numpy as np
from scipy.optimize import brentq, minimize_scalar, newton
from scipy.stats import norm
import logging
from typing import Optional, Tuple
import warnings


# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CDSImpliedVolatilitySolver:
    """
    CDS隐含波动率求解器
    
    从市场观察到的CDS价差反解出隐含的资产波动率
    """
    
    def __init__(self, S: float, D: float, t: float, r: float, 
                 L: float = 0.5, lamb: float = 0.3):
        """
        初始化CDS隐含波动率求解器
        
        Parameters:
        -----------
        S : float
            当前资产价值 (股价代理)
        D : float  
            债务面值
        t : float
            CDS期限 (年)
        r : float
            无风险利率
        L : float
            全球债务回收率 (默认0.5)
        lamb : float
            违约壁垒标准差参数 (默认0.3)
        """
        self.S = S
        self.D = D
        self.t = t
        self.r = r
        self.L = L
        self.lamb = lamb
        
        logger.info(f"CDS隐含波动率求解器初始化完成")
        logger.info(f"参数: S={S}, D={D}, t={t}, r={r}, L={L}, lamb={lamb}")
    
    def calculate_survival_probability(self, S: float, t: float, sigma: float) -> float:
        """
        计算生存概率 P(t) - 基于公式 2.11
        完全基于cgm_core.py中的实现
        
        Parameters:
        -----------
        S : float
            当前资产价值(Stock Price)
        t : float
            时间期限 (年)
        sigma : float  
            资产波动率 sigma
            
        Returns:
        --------
        float : 生存概率 P(t)
        """
        # 计算 d 参数 (公式 2.12) - 与cgm_core.py完全一致
        d = (S + self.L * self.D) * np.exp(self.lamb**2) / (self.L * self.D)
        
        # 计算 At 参数 (公式 2.13) - 与cgm_core.py完全一致
        At_square = sigma**2 * t + self.lamb**2
        At = np.sqrt(At_square)
        
        # 计算生存概率 P(t) (公式 2.11) - 与cgm_core.py完全一致
        log_d = np.log(d)
        
        # 第一项: Φ(-At/2 + log(d)/√At)
        term1_arg = -At/2 + log_d/At
        term1 = norm.cdf(term1_arg)
        
        # 第二项: d * Φ(-At/2 - log(d)/√At)  
        term2_arg = -At/2 - log_d/At
        term2 = d * norm.cdf(term2_arg)
        
        survival_prob = term1 - term2

        # 确保概率在[0,1]范围内
        survival_prob = max(0.0, min(1.0, survival_prob))
        
        return survival_prob
    
    def _calculate_G_function_approximate(self, S: float, u: float, sigma: float) -> float:
        """
        G函数的近似计算 (公式2.16的简化版本)
        完全基于cgm_core.py中的实现
        
        Parameters:
        -----------
        S : float
            当前资产价值
        u : float
            自变量
        sigma : float
            资产波动率
            
        Returns:
        --------
        float : G函数近似值
        """
        # 使用简化的近似公式 - 与cgm_core.py完全一致
        d = (S + self.L*self.D) * np.exp(self.lamb**2) / (self.L*self.D)
        
        # 简化的G函数近似 - 与cgm_core.py完全一致
        z = np.sqrt(0.25 + 2 * self.r / (sigma**2))
        
        # 基于经验的近似公式 - 与cgm_core.py完全一致
        G_approx = d**(z+0.5)* norm.cdf(-np.log(d)/(sigma*np.sqrt(u)) - z*sigma*np.sqrt(u)) + d**(-1*z + 0.5)* norm.cdf(-np.log(d)/(sigma*np.sqrt(u)) + z*sigma*np.sqrt(u))
        
        return G_approx
    
    def calculate_cds_spread_continuous(self, S: float, sigma: float, R: float = 0.4) -> float:
        """
        计算CDS连续支付价差 - 基于公式 2.15
        完全基于cgm_core.py中的实现
        
        Parameters:
        -----------
        S : float
            当前资产价值
        sigma : float
            资产波动率 sigma
        R : float
            回收率 (默认0.4，即40%)
            
        Returns:
        --------
        float : CDS价差 c*
        """
        # 计算 ξ = λ²/σ² - 与cgm_core.py完全一致
        xi = (self.lamb**2) / (sigma**2)
        
        # 计算 P(0) 和 P(t) - 与cgm_core.py完全一致
        P_0 = self.calculate_survival_probability(S=S, t=0.001, sigma=sigma)  # t=0的近似值
        P_t = self.calculate_survival_probability(S=S, t=self.t, sigma=sigma)
        
        # 计算G函数相关项 (简化版本，完整实现需要公式2.16) - 与cgm_core.py完全一致
        G_term = self._calculate_G_function_approximate(S=S, u=self.t + xi, sigma=sigma) - self._calculate_G_function_approximate(S=S, u=xi, sigma=sigma)
        
        # 计算分子 - 与cgm_core.py完全一致
        numerator = self.r * (1 - R) * (1 - P_0 + np.exp(self.r * xi) * G_term)
        
        # 计算分母 - 与cgm_core.py完全一致
        denominator = P_0 - P_t * np.exp(-self.r * self.t) - np.exp(self.r * xi) * G_term
        
        # 避免除零 - 与cgm_core.py完全一致
        if abs(denominator) < 1e-10:
            logger.warning("分母接近零，使用替代计算方法")
            # 使用简化公式
            default_prob = 1 - P_t
            spread = (default_prob * (1 - R)) / self.t
            return spread
        
        cds_spread = numerator / denominator
        
        # 确保价差为正 - 与cgm_core.py完全一致
        cds_spread = max(0.0, cds_spread)
        
        return cds_spread
    
    def objective_function(self, sigma: float, target_cds_spread: float, R: float) -> float:
        """
        目标函数: 模型CDS价差 - 市场CDS价差
        
        Parameters:
        -----------
        sigma : float
            资产波动率 (待求解)
        target_cds_spread : float
            目标CDS价差 (市场观察值)
        R : float
            回收率
            
        Returns:
        --------
        float : 目标函数值
        """
        try:
            model_spread = self.calculate_cds_spread_continuous(self.S, sigma, R)
            return model_spread - target_cds_spread
        except Exception as e:
            logger.warning(f"计算CDS价差时出错 (sigma={sigma}): {e}")
            return np.inf
    
    def solve_implied_volatility_brent(self, target_cds_spread: float, R: float = 0.4,
                                     vol_min: float = 0.01, vol_max: float = 2.0,
                                     tolerance: float = 1e-6) -> Optional[float]:
        """
        使用Brent方法求解隐含波动率
        
        Parameters:
        -----------
        target_cds_spread : float
            目标CDS价差 (市场观察值)
        R : float
            回收率
        vol_min : float
            波动率搜索下界
        vol_max : float  
            波动率搜索上界
        tolerance : float
            收敛容差
            
        Returns:
        --------
        Optional[float] : 隐含波动率，如果求解失败返回None
        """
        
        def obj_func(sigma):
            return self.objective_function(sigma, target_cds_spread, R)
        
        try:
            # 检查边界条件
            f_min = obj_func(vol_min)
            f_max = obj_func(vol_max)
            
            logger.info(f"边界检查: f({vol_min})={f_min:.6f}, f({vol_max})={f_max:.6f}")
            
            # 如果边界同号，尝试扩展搜索范围
            if f_min * f_max > 0:
                logger.warning("边界同号，尝试扩展搜索范围")
                # 向下扩展
                if f_min > 0:
                    vol_min = max(0.001, vol_min / 2)
                # 向上扩展  
                if f_max < 0:
                    vol_max = min(5.0, vol_max * 2)
                
                f_min = obj_func(vol_min)
                f_max = obj_func(vol_max)
                
                if f_min * f_max > 0:
                    logger.error("无法找到合适的搜索区间")
                    return None
            
            # 使用Brent方法求解
            implied_vol = brentq(obj_func, vol_min, vol_max, xtol=tolerance)
            
            # 验证结果
            verification_spread = self.calculate_cds_spread_continuous(self.S, implied_vol, R)
            error = abs(verification_spread - target_cds_spread)
            
            logger.info(f"Brent方法求解成功:")
            logger.info(f"  隐含波动率: {implied_vol:.6f}")
            logger.info(f"  目标CDS价差: {target_cds_spread:.6f}")
            logger.info(f"  模型CDS价差: {verification_spread:.6f}")
            logger.info(f"  误差: {error:.8f}")
            
            return implied_vol ,error 
            
        except Exception as e:
            logger.error(f"Brent方法求解失败: {e}")
            return None
    
    def solve_implied_volatility_minimize(self, target_cds_spread: float, R: float = 0.4,
                                        vol_min: float = 0.01, vol_max: float = 2.0) -> Optional[float]:
        """
        使用最小化方法求解隐含波动率
        
        Parameters:
        -----------
        target_cds_spread : float
            目标CDS价差
        R : float
            回收率
        vol_min : float
            波动率搜索下界
        vol_max : float
            波动率搜索上界
            
        Returns:
        --------
        Optional[float] : 隐含波动率，如果求解失败返回None
        """
        
        def obj_func_abs(sigma):
            return abs(self.objective_function(sigma, target_cds_spread, R))
        
        try:
            result = minimize_scalar(
                obj_func_abs,
                bounds=(vol_min, vol_max),
                method='bounded'
            )
            
            if result.success:
                implied_vol = result.x
                verification_spread = self.calculate_cds_spread_continuous(self.S, implied_vol, R)
                error = abs(verification_spread - target_cds_spread)
                
                # logger.info(f"最小化方法求解成功:")
                # logger.info(f"  隐含波动率: {implied_vol:.6f}")
                # logger.info(f"  目标CDS价差: {target_cds_spread:.6f}")
                # logger.info(f"  模型CDS价差: {verification_spread:.6f}")
                # logger.info(f"  误差: {error:.8f}")
                
                return implied_vol,error
            else:
                logger.error("最小化方法求解失败")
                return None
                
        except Exception as e:
            logger.error(f"最小化方法求解出错: {e}")
            return None
    
    def solve_implied_volatility(self, target_cds_spread: float, R: float = 0.4,
                               method: str = 'brent', **kwargs) -> Optional[float]:
        """
        求解隐含波动率的主接口
        
        Parameters:
        -----------
        target_cds_spread : float
            目标CDS价差 (可以是基点形式，会自动转换)
        R : float
            回收率
        method : str
            求解方法 ('brent', 'minimize')
        **kwargs : 
            其他参数传递给具体的求解函数
            
        Returns:
        --------
        Optional[float] : 隐含波动率
        """
        
        # 如果输入的是基点形式，转换为小数形式
        if target_cds_spread > 10:  # 假设大于10的是基点
            target_cds_spread = target_cds_spread / 10000
            logger.info(f"输入转换: {target_cds_spread*10000:.0f}bp -> {target_cds_spread:.6f}")
        
        # logger.info(f"开始求解隐含波动率 (方法: {method})")
        # logger.info(f"目标CDS价差: {target_cds_spread:.6f} ({target_cds_spread*10000:.1f}bp)")
        
        if method.lower() == 'brent':
            implied_vol,error =  self.solve_implied_volatility_brent(target_cds_spread, R, **kwargs)
            return implied_vol,error
        elif method.lower() == 'minimize':
            implied_vol,error = self.solve_implied_volatility_minimize(target_cds_spread, R, **kwargs)
            return implied_vol,error
        else:
            logger.error(f"不支持的求解方法: {method}")
            return None


def demo_implied_volatility():
    """演示隐含波动率计算"""
    
    logger.info("=== CDS隐含波动率求解演示 ===")
    
    # 参数设置
    S = 100.0        # 资产价值
    D = 150.0        # 债务面值  
    t = 5.0          # 5年期
    r = 0.05         # 5% 无风险利率
    R = 0.4          # 40% 回收率
    L = 0.5          # 债务回收率
    lamb = 0.3       # 违约壁垒参数
    
    # 创建求解器
    solver = CDSImpliedVolatilitySolver(S, D, t, r, L, lamb)
    
    # 测试案例：给定不同的市场CDS价差，求解隐含波动率
    test_cases = [100, 200, 300, 500]  # 基点
    
    for market_spread_bp in test_cases:
        logger.info(f"\n--- 测试案例: 市场CDS价差 = {market_spread_bp}bp ---")
        
        # 使用Brent方法求解
        implied_vol_brent = solver.solve_implied_volatility(
            market_spread_bp, R, method='brent'
        )
        
        # 使用最小化方法求解
        implied_vol_minimize = solver.solve_implied_volatility(
            market_spread_bp, R, method='minimize'
        )
        
        if implied_vol_brent is not None:
            # 验证结果
            verification_spread = solver.calculate_cds_spread_continuous(solver.S, implied_vol_brent, R)
            logger.info(f"Brent方法结果验证: 模型价差 = {verification_spread*10000:.2f}bp")
        
        if implied_vol_minimize is not None:
            verification_spread = solver.calculate_cds_spread_continuous(solver.S, implied_vol_minimize, R)
            logger.info(f"最小化方法结果验证: 模型价差 = {verification_spread*10000:.2f}bp")


def test_consistency_with_cgm_core():
    """测试与cgm_core.py的一致性"""
    
    logger.info("=== 测试与cgm_core.py的一致性 ===")
    
    # 使用相同的参数
    S = 100.0
    D = 150.0
    t = 5.0
    r = 0.05
    R = 0.4
    L = 0.5
    lamb = 0.3
    test_sigma = 0.3  # 测试用的波动率
    
    # 创建求解器
    solver = CDSImpliedVolatilitySolver(S, D, t, r, L, lamb)
    
    # 计算CDS价差
    cds_spread = solver.calculate_cds_spread_continuous(S, test_sigma, R)
    logger.info(f"使用波动率 {test_sigma:.4f} 计算的CDS价差: {cds_spread:.6f} ({cds_spread*10000:.2f}bp)")
    
    # 现在反解波动率
    logger.info(f"\n从CDS价差 {cds_spread*10000:.2f}bp 反解波动率:")
    
    implied_vol = solver.solve_implied_volatility(cds_spread, R, method='brent')
    
    if implied_vol is not None:
        error = abs(implied_vol[0] - test_sigma)
        logger.info(f"原始波动率: {test_sigma:.6f}")
        logger.info(f"隐含波动率: {implied_vol[0]:.6f}")
        logger.info(f"绝对误差: {error:.8f}")
        logger.info(f"相对误差: {error/test_sigma:.6%}")
        
        if error < 1e-4:
            logger.info("✓ 一致性测试通过!")
        else:
            logger.warning("✗ 一致性测试失败!")
    else:
        logger.error("✗ 反解失败!")


if __name__ == "__main__":
    # 先测试一致性
    test_consistency_with_cgm_core()
    
    # 再运行演示
    print("\n" + "="*60 + "\n")
    # demo_implied_volatility()
