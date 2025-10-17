"""
Delta Hedging Strategy for CDS-Stock Trading
Provides tools to generate delta hedge tables and calculate required trading volumes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional, List, Tuple, Union
from implied_vol_solver import CDSImpliedVolatilitySolver
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeltaHedgeStrategy:
    """
    Class for CDS-Stock delta hedging strategy.
    Provides tools to generate delta hedge tables and calculate required trading volumes
    for maintaining delta neutrality in a CDS-Stock trading strategy.
    """

    def __init__(self, solver: CDSImpliedVolatilitySolver, notional: float):
        """
        Initialize the delta hedge strategy.

        Parameters:
        -----------
        solver : CDSImpliedVolatilitySolver
            Solver instance for CDS calculations
        notional : float
            Notional amount of the CDS contract
        """
        self.solver = solver
        self.notional = notional
        self.logger = logging.getLogger(__name__)

    def generate_hedge_table(
        self,
        current_price: float,
        D: float,
        implied_vol: float,
        R: float,
        price_range_pct: float = 0.20,
        price_step_pct: float = 0.01,
    ) -> Dict[float, int]:
        """
        Generate a dictionary mapping stock prices to required hedge positions.

        Parameters:
        -----------
        current_price : float
            Current stock price
        D : float
            Debt per share
        implied_vol : float
            Implied volatility
        R : float
            Recovery rate
        price_range_pct : float
            Price range percentage (default 20%)
        price_step_pct : float
            Price step percentage (default 1%)

        Returns:
        --------
        Dict[float, int] : Dictionary mapping prices to share quantities
        """
        # Calculate price range
        min_price = current_price * (1 - price_range_pct)
        max_price = current_price * (1 + price_range_pct)

        # Generate price points
        price_steps = int(price_range_pct * 2 / price_step_pct) + 1
        price_points = np.linspace(min_price, max_price, price_steps)

        # Initialize hedge dictionary
        hedge_dict = {}

        # Calculate current delta for reference
        current_delta = self.solver.delta_calculation(current_price, D, implied_vol, R)
        current_position = self.calculate_required_shares(current_delta)

        # For each price point, calculate required position
        for price in price_points:
            # Calculate delta at this price
            delta = self.solver.delta_calculation(price, D, implied_vol, R)

            # Calculate required shares
            required_shares = self.calculate_required_shares(delta)

            # Calculate trade size (positive for buy, negative for sell)
            trade_size = required_shares - current_position

            # Store in dictionary with rounded price as key
            hedge_dict[round(price, 2)] = round(trade_size)

        return hedge_dict

    def calculate_required_shares(self, delta: float) -> int:
        """
        Calculate required number of shares for delta hedging at given price.

        Parameters:
        -----------
        price : float
            Stock price
        delta : float
            Delta value

        Returns:
        --------
        int : Number of shares required for delta neutrality
        """
        # For sell CDS buy stock strategy, take negative of delta
        return int(-1 * delta * self.notional / 1.0)

    def calculate_trade_size(
        self,
        current_price: float,
        new_price: float,
        D: float,
        implied_vol: float,
        R: float,
        current_position: int = 0,
    ) -> int:
        """
        Calculate the trade size needed to rebalance the hedge.

        Parameters:
        -----------
        current_price : float
            Current stock price
        new_price : float
            New stock price
        D : float
            Debt per share
        implied_vol : float
            Implied volatility
        R : float
            Recovery rate
        current_position : int
            Current position in number of shares

        Returns:
        --------
        int : Number of shares to trade (positive for buy, negative for sell)
        """
        # Calculate delta at new price
        new_delta = self.solver.delta_calculation(new_price, D, implied_vol, R)

        # Calculate required position at new price
        new_position = self.calculate_required_shares(new_price, new_delta)

        # Calculate trade size
        trade_size = new_position - current_position

        return int(trade_size)

    def plot_hedge_table(
        self,
        current_price: float,
        D: float,
        implied_vol: float,
        R: float,
        price_range_pct: float = 0.20,
        price_step_pct: float = 0.01,
    ) -> None:
        """
        Plot the hedge table for visualization.

        Parameters:
        -----------
        current_price : float
            Current stock price
        D : float
            Debt per share
        implied_vol : float
            Implied volatility
        R : float
            Recovery rate
        price_range_pct : float
            Price range percentage (default 20%)
        price_step_pct : float
            Price step percentage (default 1%)
        """
        # Generate hedge table
        hedge_dict = self.generate_hedge_table(
            current_price, D, implied_vol, R, price_range_pct, price_step_pct
        )

        # Convert to DataFrame for plotting
        df = pd.DataFrame(
            {"price": list(hedge_dict.keys()), "trade_size": list(hedge_dict.values())}
        )

        # Sort by price
        df = df.sort_values("price")

        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Plot trade sizes
        ax1.bar(
            df["price"], df["trade_size"], width=current_price * price_step_pct * 0.8
        )
        ax1.set_title("Required Trade Size for Delta Hedging")
        ax1.set_ylabel("Shares to Trade")
        ax1.axhline(y=0, color="r", linestyle="-", alpha=0.3)
        ax1.grid(True, alpha=0.3)

        # Plot cumulative position
        df["cumulative_position"] = df["trade_size"].cumsum()
        ax2.plot(df["price"], df["cumulative_position"], "b-")
        ax2.set_title("Cumulative Position Size")
        ax2.set_xlabel("Stock Price")
        ax2.set_ylabel("Total Shares Held")
        ax2.grid(True, alpha=0.3)

        # Mark current price
        for ax in [ax1, ax2]:
            ax.axvline(
                x=current_price, color="g", linestyle="--", label="Current Price"
            )
            ax.legend()

        plt.tight_layout()
        plt.show()

    def export_to_csv(
        self, hedge_dict: Dict[float, int], filename: str = "hedge_table.csv"
    ) -> None:
        """
        Export the hedge table to a CSV file.

        Parameters:
        -----------
        hedge_dict : Dict[float, int]
            Dictionary containing price -> shares mapping
        filename : str
            Filename to save the CSV to
        """
        df = pd.DataFrame(
            {"price": list(hedge_dict.keys()), "trade_size": list(hedge_dict.values())}
        )
        df = df.sort_values("price")
        df.to_csv(filename, index=False)
        self.logger.info(f"Hedge table exported to {filename}")

    def intraday_hedge_simulation(
        self, price_series: List[float], D: float, implied_vol: float, R: float
    ) -> pd.DataFrame:
        """
        Simulate intraday hedging based on a series of prices.

        Parameters:
        -----------
        price_series : List[float]
            List of prices throughout the day
        D : float
            Debt per share
        implied_vol : float
            Implied volatility
        R : float
            Recovery rate

        Returns:
        --------
        pd.DataFrame : DataFrame containing simulation results
        """
        # Initialize results
        results = []
        current_position = 0

        # Process each price point
        for i, price in enumerate(price_series):
            # Calculate delta at this price
            delta = self.solver.delta_calculation(price, D, implied_vol, R)

            # Calculate required position
            required_position = self.calculate_required_shares(price, delta)

            # Calculate trade size
            trade_size = required_position - current_position

            # Update current position
            current_position = required_position

            # Store results
            results.append(
                {
                    "time_step": i,
                    "price": price,
                    "delta": delta,
                    "required_position": required_position,
                    "trade_size": trade_size,
                    "current_position": current_position,
                }
            )

        return pd.DataFrame(results)


# Example usage
if __name__ == "__main__":
    # Parameters
    t = 5.0  # CDS tenor
    r = 0.05  # risk-free rate
    L = 0.5  # loss given default
    lamb = 0.3  # barrier deviation
    notional = 10_000_000  # CDS notional amount
    cds_coupon = 0.01

    # Create solver and strategy objects
    solver = CDSImpliedVolatilitySolver(t, r, L, lamb, notional)
    strategy = DeltaHedgeStrategy(solver, notional)

    # Example parameters
    current_price = 50.0
    D = 75.0
    implied_vol = 0.3
    R = 0.4

    # Generate hedge table
    hedge_table = strategy.generate_hedge_table(current_price, D, implied_vol, R)

    # Print example entries
    print("Stock Price -> Required Trade Size:")
    for price in sorted(list(hedge_table.keys()))[::5]:  # Print every 5th entry
        print(f"${price:.2f} -> {hedge_table[price]:+,d} shares")

    # Plot hedge table
    strategy.plot_hedge_table(current_price, D, implied_vol, R)
