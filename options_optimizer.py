"""
Options Analytics Module

This module contains classes and functions for equity options analytics.
"""

from enum import Enum
from math import log, sqrt
from typing import Iterable, Optional
from scipy.stats import norm

EQ_OPTION_PRICING_FACTOR = 100


class OptionType(Enum):
    CALL = "call"
    PUT = "put"


class EquityOption:
    """
    A class to represent and analyze equity options.
    """

    def __init__(
        self,
        strike: float,
        expiry_in_years: float,
        option_type: OptionType,
        implied_vol: float,
        option_price: float,
    ):
        """
        Initialize the EquityOption instance.

        Parameters:
        -----------
        strike : float
            The strike price of the option.
        expiry_in_years : float
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
        self.expiry_in_years = expiry_in_years
        self.option_type = option_type
        self.implied_vol = implied_vol
        self.option_price = option_price
        self.number_contracts = 1

    def __repr__(self):
        """String representation of the EquityOption instance."""
        return f"EquityOption(strike={self.strike}, expiration_in_years={self.expiration_in_years}, option_type={self.option_type.value}, implied_vol={self.implied_vol}, option_price={self.option_price})"

    def set_number_contracts(self, number_contracts: int):
        """
        Set the number of contracts for this option.

        Parameters:
        -----------
        number_contracts : int
            The number of contracts.
        """
        self.number_contracts = number_contracts

    def calculate_bs_delta(
        self, stock_price: Optional[float] = None, interest_rate: float = 0.0
    ):
        """
        Calculate the delta of the option.

        Parameters:
        -----------
        stock_price : float
            The current stock price.
        interest_rate : float, optional
            The risk-free interest rate (default is 0.0).
        """

        S = (
            stock_price if stock_price is not None else self.strike
        )  # should be the ATM forward, but meh
        K = self.strike
        T = self.expiry_in_years
        r = interest_rate
        sigma = self.implied_vol

        d1 = (
            (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
            if sigma > 0 and T > 0
            else 0.0
        )

        if self.option_type == OptionType.CALL:
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1

    def equivalent_stock_shares(
        self, stock_price_grids: Iterable[float], interest_rate: float = 0.0
    ) -> dict[float, float]:
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
        dict[float, float]
            Dictionary mapping stock prices to equivalent stock shares.
        """
        equivalent_shares = {}
        for price in stock_price_grids:
            delta = self.calculate_bs_delta(price, interest_rate)
            # We intend to buy OTM put and sell OTM call
            if self.option_type == OptionType.CALL:
                eq_shares = int(
                    round(-delta * self.number_contracts * EQ_OPTION_PRICING_FACTOR)
                )
            else:
                eq_shares = int(
                    round(delta * self.number_contracts * EQ_OPTION_PRICING_FACTOR)
                )
            equivalent_shares[price] = eq_shares
        return equivalent_shares
