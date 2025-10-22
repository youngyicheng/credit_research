"""
Generate daily delta hedge maps for historical CDS data.
"""

from typing import Optional
import pandas as pd
import numpy as np
from datetime import datetime
import os
from tqdm import tqdm
from io import StringIO
from contextlib import redirect_stdout
import pickle

from deltaHedge import DeltaHedgeStrategy
from implied_vol_solver import CDSImpliedVolatilitySolver, CDSQuoteType


def generate_daily_stock_share_maps(
    df: pd.DataFrame,
    t: float = 5.0,
    r: float = 0.05,
    L: float = 0.5,
    lamb: float = 0.3,
    notional: float = 10_000_000,
    price_step_pct: float = 0.01,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    output_dir: str = "hedge_maps",
):
    """
    Generate daily stock share maps from historical data.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing historical CDS and stock data
    t : float
        CDS tenor in years
    r : float
        Risk-free rate
    L : float
        Loss given default
    lamb : float
        Barrier deviation
    notional : float
        CDS notional amount
    price_range_pct : float
        Stock price range percentage (default 20%)
    price_step_pct : float
        Stock price step percentage (default 1%)
    output_dir : str
        Directory to save hedge maps

    Returns:
    --------
    dict : Dictionary mapping dates to hedge dictionaries
    """
    # Create solver and strategy objects
    solver = CDSImpliedVolatilitySolver(t, r, L, lamb)
    hedge_strategy = DeltaHedgeStrategy(solver, notional)

    # Make sure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create a dictionary to store hedge maps by date
    hedge_maps = {}

    # Process each row in the DataFrame
    with redirect_stdout(StringIO()):  # Suppress output
        for i, row in tqdm(df.iterrows(), total=len(df), desc="Generating hedge maps"):
            try:
                # Extract required parameters
                date = i  # Assuming index is date
                current_price = row["stock_price"]
                D = row["financial_debt_ratio"]
                implied_vol = row["implied_vol"]
                R = row["cdsassumedrecovery"]

                # Skip if any required parameter is missing
                if (
                    pd.isna(current_price)
                    or pd.isna(D)
                    or pd.isna(implied_vol)
                    or pd.isna(R)
                ):
                    continue

                # Generate hedge table for this date
                hedge_dict = hedge_strategy.generate_stock_share_table(
                    current_price=current_price,
                    D=D,
                    implied_vol=implied_vol,
                    R=R,
                    min_price=min_price,
                    max_price=max_price,
                    price_step_pct=price_step_pct,
                )

                # Store in dictionary
                hedge_maps[date] = hedge_dict

                # Save individual hedge map to CSV
                date_str = date if isinstance(date, str) else date.strftime("%Y%m%d")
                filename = f"{output_dir}/hedge_map_{date_str}.csv"

                # Convert to DataFrame and save
                hedge_df = pd.DataFrame(
                    {
                        "price": list(hedge_dict.keys()),
                        "trade_size": list(hedge_dict.values()),
                    }
                ).sort_values("price")

                hedge_df.to_csv(filename, index=False)

            except Exception as e:
                print(f"Error processing row for date {i}: {e}")
                continue

    # Save the complete hedge map dictionary
    with open(f"{output_dir}/all_hedge_maps.pkl", "wb") as f:
        pickle.dump(hedge_maps, f)

    return hedge_maps


if __name__ == "__main__":
    # Example usage
    df = pd.read_csv("res/output_data.csv", index_col="date", parse_dates=True)

    # Parameters
    t = 5.0  # CDS tenor
    r = 0.05  # risk-free rate
    L = 0.5  # loss given default
    lamb = 0.3  # barrier deviation
    notional = -100_000_000  # negative for sell CDS protection/buy stock

    # Generate hedge maps
    hedge_maps = generate_daily_hedge_maps(df, t, r, L, lamb, notional)

    # Print summary
    print(f"Generated hedge maps for {len(hedge_maps)} trading days")

    # Example: Access the hedge map for a specific date
    first_date = list(hedge_maps.keys())[0]
    first_hedge_map = hedge_maps[first_date]
    print(f"\nExample hedge map for {first_date}:")
    prices = sorted(list(first_hedge_map.keys()))[:5]  # Show first 5 price points
    for price in prices:
        print(f"${price:.2f} -> {int(first_hedge_map[price]):+,d} shares")
