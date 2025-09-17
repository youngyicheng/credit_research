"""
Credit Grade Model 
基于论文公式实现的信用违约概率和CDS定价计算

主要功能:
1. 生存概率计算 (公式 2.11)
2. CDS价差计算 (公式 2.15-2.16)
3. 从CDS价格隐含资产波动率

作者: Yicheng
日期: 2025
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import logging
from implied_vol_cds import *


import os
from datetime import datetime

def log_init():
    """初始化日志配置"""
    # 创建logs目录
    log_dir = "/home/yicheng/credit/logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 日志文件路径
    log_file = os.path.join(log_dir, "cgm_core.log")
    
    # 清除之前的配置
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # 配置日志格式
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # 配置日志记录器
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            # 文件处理器 - 保存到文件
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            # 控制台处理器 - 输出到控制台
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("CGM Core Module Started")
    logger.info(f"Log File: {log_file}")
    logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    return logger

# 初始化日志
logger = log_init()


class CreditGradeModel:
    """
    实现基于Merton模型的信用风险计算:
    - 违约概率/生存概率计算
    - CDS公平价差计算
    - 资产波动率隐含计算
    """
    
    def __init__(self, S , D, t, r, R, market_cds_spread ,L = 0.5, lamb = 0.3):
        """Initialize the Credit Grade Model
        
        Parameters:
        -----------
        S : float
            current asset value (Stock Price proxy)
        sigma : float  
            asset volatility sigma(to be implied from CDS price) from implied_vol_cds.py
        D : float 
            debt per share
        L(hat): float
            global debt recovery rate (default 0.5)
        t : float
            CDS maturity (year)

        market_cds_spread: float
            market observed CDS spread

        r : float
            risk free rate
        R : float
            asset recovery rate (default 0.4) when default
        lamb : float
            percentage standard deviation of the default barrier (default 0.3)
        
        """

        self.S = S

        # 此处的sigma从cds price imply 过来
        L2 = calculate_leverage_ratio(S, D, r, t)

        # target implied 
        self.sigma = implied_volatility_from_cds_spread(
            basis_points_to_cds_spread(market_cds_spread), L2, t, r
        )

        # print(sigma)
        
        self.L = L
        self.D = D
        self.t = t
        self.r = r
        self.R = R
        self.lamb = lamb
    

    def calculate_survival_probability(self,S, t ,sigma):
        """
        计算生存概率 P(t) - 基于公式 2.11
        
        Parameters:
        -----------
        S : float
            当前资产价值(Stock Price)

        sigma : float  
            资产波动率 sigma

        LD : float
            债务面值 (Liability * D) 
            L: global debt recovery rate (0.5)

        t : float
            时间期限 (年)
        lamb : float
            违约壁垒参数 λ (默认0.2)
            
        Returns:
        --------
        float : 生存概率 P(t)
        
        公式说明:
        P(t) = Φ(-At/2 + log(d)/√At) - d * Φ(-At/2 - log(d)/√At)
        其中: d = S*e^(λ²)/LD, At = σ²*t + λ²
        """
        
        # 计算 d 参数 (公式 2.12)
        d = (S + self.L* self.D) * np.exp(self.lamb**2) / (self.L*self.D)
        # 计算 At 参数 (公式 2.13) 
        At_square = sigma**2 * t + self.lamb**2
        At = np.sqrt(At_square)
        
        # 计算生存概率 P(t) (公式 2.11)
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
        
        # logger.info(f"生存概率计算: P({t}) = {survival_prob:.6f}")
        # logger.info(f"参数: S={self.S}, asset volatility={sigma}, L={self.L}, D={self.D}, λ={self.lamb}")
        # logger.info(f"中间值: d={d:.6f}, At={At:.6f}")
        
        return survival_prob
    
    def calculate_default_probability(self, S, sigma):
        """
        计算违约概率
        
        Returns:
        --------
        float : 违约概率 = 1 - 生存概率
        """
        survival_prob = self.calculate_survival_probability(S, self.t,sigma)
        return 1.0 - survival_prob
    

    def calculate_cds_spread_continuous(self, S, sigma , R=0.4):
        """
        计算CDS连续支付价差 - 基于公式 2.15
        此处为CDS定价公式
        
        Parameters:
        -----------
        S : float
            当前资产价值
        sigma : float
            资产波动率 sigma  
        LD : float
            债务面值
        t : float
            CDS期限 (年)
        r : float
            无风险利率
        R : float
            回收率 (默认0.4，即40%)
        lamb : float
            deviatino of liability
            违约壁垒参数 λ
            
        Returns:
        --------
        float : CDS价差 c*
        
        公式说明:
        c* = r(1-R) * [1-P(0)+e^(rξ)(G(t+ξ)-G(ξ))] / [P(0)-P(t)e^(-rt)-e^(rξ)(G(t+ξ)-G(ξ))]
        其中 ξ = λ²/σ²
        """
        
        # 计算 ξ = λ²/σ² 
        xi = (self.lamb**2) / (sigma**2)
        
        # 计算 P(0) 和 P(t)
        P_0 = self.calculate_survival_probability(S=S,t=0.001,sigma=sigma)  # t=0的近似值
        P_t = self.calculate_survival_probability(S=S,t=self.t,sigma=sigma)
        
        # 计算G函数相关项 (简化版本，完整实现需要公式2.16)
        G_term = self._calculate_G_function_approximate(S=S, u=self.t + xi, sigma=sigma) - self._calculate_G_function_approximate(S=S,u=xi, sigma=sigma)
        # G_term_denominator = self._calculate_G_function_approximate(xi)
        
        # 计算分子
        numerator = self.r * (1 - R) * (1 - P_0 + np.exp(self.r * xi) * G_term)
        
        # 计算分母  
        denominator = P_0 - P_t * np.exp(-self.r * self.t) - np.exp(self.r * xi) * G_term
        
        # 避免除零
        if abs(denominator) < 1e-10:
            logger.warning("分母接近零，使用替代计算方法")
            # 使用简化公式
            default_prob = 1 - P_t
            spread = (default_prob * (1 - R)) / self.t
            return spread
        
        cds_spread = numerator / denominator
        
        # 确保价差为正
        cds_spread = max(0.0, cds_spread)
        
        # logger.info(f"CDS价差计算: c* = {cds_spread:.6f}")
        # logger.info(f"P(0)={P_0:.6f}, P({t})={P_t:.6f}")
        
        return cds_spread
    
    def _calculate_G_function_approximate(self, S, u, sigma):
        """
        xi: 传入的自变量
        G函数的近似计算 (公式2.16的简化版本)
        
        完整的G函数实现较为复杂，这里提供简化版本
        在实际应用中可以根据需要实现完整版本
        """
        
        # 使用简化的近似公式
        d = (S + self.L*self.D) * np.exp(self.lamb**2) / (self.L*self.D)
        
        # 简化的G函数近似
        z = np.sqrt(0.25 + 2 * self.r / (sigma**2))
        
        # 基于经验的近似公式
        G_approx = d**(z+0.5)* norm.cdf(-np.log(d)/(sigma*np.sqrt(u)) - z*sigma*np.sqrt(u)) + d**(-1*z + 0.5)* norm.cdf(-np.log(d)/(sigma*np.sqrt(u)) + z*sigma*np.sqrt(u))
        
        return G_approx
    
    def imply_asset_volatility_from_cds(self, market_cds_spread, 
                                       R=0.4, ):
        """
        从市场CDS价差隐含资产波动率
        
        Parameters:
        -----------
        S : float
            当前资产价值
        LD : float  
            债务面值
        t : float
            CDS期限
        r : float
            无风险利率
        market_cds_spread : float
            市场观察到的CDS价差
        R : float
            回收率
        lamb : float
            违约壁垒参数
        vol_guess : float
            波动率初始猜测值
            
        Returns:
        --------
        float : 隐含的资产波动率
        """
        pass



    def calculate_risk_metrics(self, R=0.4, sigma_matching=0.3):
        """
        计算风险指标
        
        Returns:
        --------
        dict : 包含各种风险指标的字典
        """
        
        # 基准计算
        base_spread = self.calculate_cds_spread_continuous(self.S, R, sigma_matching)
        base_default_prob = self.calculate_default_probability(self.S, sigma_matching)
        
        # Delta: 资产价值敏感性
        dV = self.S * 0.01  # 1% 冲击
        spread_up = self.calculate_cds_spread_continuous(self.S + dV, R, sigma_matching)
        delta = (spread_up - base_spread) / dV


        # Gamma: 资产价值二阶敏感性 (凸性)
        spread_down = self.calculate_cds_spread_continuous(self.S - dV, R, sigma_matching)
        gamma = (spread_up - 2*base_spread + spread_down) / (dV**2)
        # Vega: 波动率敏感性
        # 此处vega尚未实现 后续接口等待接入
        # dsigma = 0.01  # 1% 波动率冲击
        # spread_vol_up = self.calculate_cds_spread_continuous(t, R)
        # vega = (spread_vol_up - base_spread) / dsigma
        # Vega: 波动率敏感性


        dsigma = 0.01  # 1% 波动率冲击
        # 创建新的模型实例用于计算波动率敏感性
        # temp_model_up = CreditGradeModel(
        #     S=self.S, 
        #     D=self.D, 
        #     t=self.t, 
        #     r=self.r, 
        #     R=self.R, 
        #     market_cds_spread=basis_points_to_cds_spread((sigma_matching + dsigma) * 10000), 
        #     L=self.L, 
        #     lamb=self.lamb
        # )

        # spread_vol_up = self.calculate_cds_spread_continuous(sigma_matching + dsigma, R)
        # vega = (spread_vol_up - base_spread) / dsigma
        
        # Theta: 时间衰减
        # dt = -1/365  # 1天
        # spread_time = self.calculate_cds_spread_continuous(R, sigma_matching)
        # theta = (spread_time - base_spread) / (-dt)
        
        return {
            'cds_spread': base_spread,
            'cds_spread_bps': base_spread * 10000,
            'default_probability': base_default_prob,
            'survival_probability': 1 - base_default_prob,
            'delta': delta,
            'gamma': gamma,
            # 'vega': vega, 
            # 'theta': theta
        }


def demo_calculation():
    """演示计算功能"""
    
    logger.info("=== Credit Grade Model===")
    S = 100.0        # 资产价值 (股价)
    L = 0.5         # 债务面值
    D = 150         # 债务面值
    t = 5           # 5年期
    r = 0.05          # 5% 无风险利率  
    R = 0.5           # 40% 回收率
    lamb = 0.3        # 违约壁垒参数
    market_cds_spread = 250  # 市场CDS价差
    # 创建模型实例
    cgm = CreditGradeModel(S = S, L = L, D = D, t = t, r = r, R = R, lamb = lamb, market_cds_spread = market_cds_spread)

    logger.info("Parameters:")
    logger.info(f"Asset Value S: {S}")
    logger.info(f"Market CDS Spread: {market_cds_spread}bp")
    logger.info(f"Debt Value L: {L}")
    logger.info(f"Debt Value D: {D}")
    logger.info(f"Maturity t: {t} years")
    logger.info(f"Risk Free Rate r: {r:.1%}")
    logger.info(f"Recovery Rate R: {R:.1%}")
    logger.info(f"Default Barrier Parameter λ: {lamb}")
    logger.info(f"Implied Asset Volatility: {cgm.sigma:.4f}")
    
    # 1. 计算生存概率
    survival_prob = cgm.calculate_survival_probability(S,t, cgm.sigma)
    default_prob = 1 - survival_prob
    
    logger.info("=== Survival Probability Calculation (Formula 2.11) ===")
    logger.info(f"Survival Probability P({t}): {survival_prob:.4f}")
    logger.info(f"Default Probability: {default_prob:.4f}")
    
    # 2. 计算CDS价差
    cds_spread = cgm.calculate_cds_spread_continuous(S, R, cgm.sigma)
    
    logger.info("=== CDS Spread Calculation (Formula 2.15) ===")
    logger.info(f"CDS Spread: {cds_spread:.4f} ({cds_spread*10000:.1f} bps)")
    
    # 3. 风险指标
    # sigma should be implied from market cds spread and c star(cds_spread) which used to calculate the delta
    risk_metrics = cgm.calculate_risk_metrics(R, cgm.sigma)
    
    logger.info("=== Risk Metrics ===")
    logger.info(f"Delta (Asset Value Sensitivity): {risk_metrics['delta']:.6f}")
    logger.info(f"Gamma (Asset Value Second Order Sensitivity): {risk_metrics['gamma']:.6f}")
    # logger.info(f"Vega (Volatility Sensitivity): {risk_metrics['vega']:.6f}")
    # logger.info(f"Theta (Time Decay): {risk_metrics['theta']:.6f}")
    

    return cgm


if __name__ == "__main__":
    demo_calculation()
