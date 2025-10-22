"""
Options Analytics Module

This module contains classes and functions for equity options analytics.
"""

from enum import Enum
from math import log, sqrt
from scipy.stats import norm

class OptionType(Enum):
    CALL = "call"
    PUT = "put"

class EquityOption:
    """
    A class to represent and analyze equity options.
    """

    def __init__(self, strike: float, tenor_in_years: float, option_type: OptionType, implied_vol: float, option_price: float, number_contracts: int):
        """
        Initialize the EquityOption instance.

        Parameters:
        -----------
        strike : float
            The strike price of the option.
        tenor_in_years : float
            The time to expiration in years.
        option_type : OptionType
            The type of the option (CALL or PUT).
        implied_vol : float
            The implied volatility of the option.
        option_price : float
            The market price of the option.
        number_contracts : int
            The number of contracts.
        """
        self.strike = strike
        self.tenor_in_years = tenor_in_years
        self.option_type = option_type
        self.implied_vol = implied_vol
        self.option_price = option_price
        self.number_contracts = number_contracts
    
    def __repr__(self):
        """String representation of the EquityOption instance."""
        return f"EquityOption(strike={self.strike}, tenor_in_years={self.tenor_in_years}, option_type={self.option_type.value}, implied_vol={self.implied_vol}, option_price={self.option_price})"
    
    def calculate_bs_delta(self, stock_price: float, interest_rate: float = 0.0):
        """
        Calculate the delta of the option.
        
        Parameters:
        -----------
        stock_price : float
            The current stock price.
        interest_rate : float, optional
            The risk-free interest rate (default is 0.0).
        """

        S = stock_price
        K = self.strike
        T = self.tenor_in_years
        r = interest_rate
        sigma = self.implied_vol
        
        d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T)) if sigma > 0 and T > 0 else 0.0

        if self.option_type == OptionType.CALL:
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1
    
    def equivalent_stock_shares(
        self,
        stock_price_grids: list[float],
        interest_rate: float = 0.0
    ) -> list[float]:
        """
        Compute the equivalent stock share exposure for each price in the grid.

        Parameters
        ----------
        stock_price_grids : list[float]
            List of stock prices at which to compute equivalent stock shares.
        interest_rate : float, optional
            Risk-free rate to use in delta calculation (default 0.0)

        Returns
        -------
        list[float]
            List of equivalent stock shares for each price in the grid.
        """
        equivalent_shares = []
        for price in stock_price_grids:
            delta = self.calculate_bs_delta(price, interest_rate)
            # We intend to buy OTM put and sell OTM call
            trading_factor = 100
            if self.option_type == OptionType.CALL:
                eq_shares = int(round(-delta * self.number_contracts * trading_factor))
            else:
                eq_shares = int(round(delta * self.number_contracts * trading_factor))
            equivalent_shares.append(eq_shares)
        return equivalent_shares
