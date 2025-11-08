from dataclasses import dataclass
import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm

BPS = 10000


@dataclass(frozen=True)
class TradeType:
    """Trade type labels for tracking different types of trades"""
    INITIAL_POSITION: str = 'initial_position'
    CLOSE_POSITION: str = 'close_position'
    REBALANCING: str = 'rebalancing'
    NO_TRADE: str = 'no_trade'


def calculate_pnl_w_static_stock_price_map(
    stock_data: pd.DataFrame,
    stock_price_map: dict,
    transaction_cost_bps: float = 5.0,
    stk_px_col: str = 'close',
) -> dict:
    """
    Calculate PnL based on trading map. Assumption is we are flat before and after.

    Parameters:
    -----------
    stock_data : pd.DataFrame
        Minute-level stock data with columns: date, time, close, etc. (already filtered)
    stock_price_map : dict
        Dictionary mapping stock prices to required share positions
    transaction_cost_bps : float
        Transaction cost in basis points (default: 5 bps)
    stk_px_col : str
        Column name for stock price (default: 'close')

    Returns:
    --------
    dict : Dictionary containing PnL analysis results
    """

    if stock_data.empty:
        return {"error": "No data found in provided stock_data"}

    # Sort by time to ensure chronological order
    stock_data = stock_data.sort_values(["date", "time"]).reset_index(drop=True)

    # Get sorted price levels from stock_price_map for interpolation
    price_levels = sorted(stock_price_map.keys())
    target_positions = [stock_price_map[price] for price in price_levels]

    def get_target_position(current_price: float) -> float:
        """Interpolate target position based on current stock price"""
        if current_price <= price_levels[0]:
            return target_positions[0]
        elif current_price >= price_levels[-1]:
            return target_positions[-1]
        else:
            # Linear interpolation
            return np.interp(current_price, price_levels, target_positions)

    def append_trade_log(trade_log: list, date: datetime.date, time: str, price: float, position_before: float, position_after: float, trade_type: str) -> float:
        """Helper function to append trade information to the trade log"""

        trade_size = position_after - position_before
        if trade_size == 0:
            trade_log.append(
                {
                    "date": date,
                    "time": time,
                    "price": price,
                    "trade_size": trade_size,
                    "trade_value": 0,
                    "transaction_cost": 0,
                    "position_after": position_after,
                    "cash_flow": 0,
                    "trade_type": TradeType.NO_TRADE,
                }
            )
            return position_after
        
        # Update positions and cash flow for non-zero trade size
        trade_value = trade_size * price
        transaction_cost = abs(trade_value) * (transaction_cost_bps / BPS)
        cash_flow = -trade_value  # Negative for buying, positive for selling

        trade_log.append(
            {
                "date": date,
                "time": time,
                "price": price,
                "trade_size": trade_size,
                "trade_value": trade_value,
                "transaction_cost": transaction_cost,
                "position_after": position_after,
                "cash_flow": cash_flow,
                "trade_type": trade_type,
            }
        )
        return position_after

    # Establish initial position based on opening price
    # Get opening price and establish initial position
    opening_price = stock_data.iloc[0][stk_px_col]
    opening_date = stock_data.iloc[0]["date"]
    opening_time = stock_data.iloc[0]["time"]
    current_position = 0
    initial_position = int(get_target_position(opening_price))
    trade_log = []
    
    current_position = append_trade_log(
        trade_log,
        opening_date,
        opening_time,
        opening_price,
        0,
        initial_position,
        TradeType.INITIAL_POSITION,
    )

    # Process each subsequent minute (skip the first one as it's already processed)
    for _, row in stock_data.iloc[1:].iterrows():
        current_price = row[stk_px_col]
        current_date = row["date"]
        current_time = row["time"]
        # Get target position based on current price
        target_position = int(get_target_position(current_price))
        current_position = append_trade_log(
            trade_log,
            current_date,
            current_time,
            current_price,
            current_position,
            target_position,
            TradeType.REBALANCING,
        )

    # Flatten the position at end of period
    final_price = stock_data.iloc[-1][stk_px_col]
    final_date = stock_data.iloc[-1]["date"]
    final_time = stock_data.iloc[-1]["time"]
    current_position = append_trade_log(
        trade_log,
        final_date,
        final_time,
        final_price,
        current_position,
        0,
        TradeType.CLOSE_POSITION,
    )
    current_position = 0

    # Calculate final PnL
    if not stock_data.empty:
        final_price = stock_data.iloc[-1][stk_px_col]

        # Calculate total cash flow from all trades
        total_cash_flow = sum(trade["cash_flow"] for trade in trade_log)
        
        # Calculate total transaction costs
        total_transaction_cost = sum(trade["transaction_cost"] for trade in trade_log)

        # Note on the sign: total_cash_flow is the raw pnl, while total_transaction_cost is a absolute value.
        total_pnl = total_cash_flow - total_transaction_cost

        # Calculate some statistics
        total_trades = len(trade_log)
        rebalance_trades = len([t for t in trade_log if t["trade_type"] == "rebalance"])
        total_volume = sum(abs(trade["trade_size"]) for trade in trade_log)
        avg_trade_size = total_volume / total_trades if total_trades > 0 else 0

        price_change = final_price - opening_price
        price_change_pct = (price_change / opening_price) * 100

    else:
        final_price = opening_price
        total_pnl = 0
        total_trades = rebalance_trades = total_volume = avg_trade_size = 0
        price_change = price_change_pct = 0

    # Return comprehensive results
    return {
        "opening_price": opening_price,
        "final_price": final_price,
        "price_change": price_change,
        "price_change_pct": price_change_pct,
        "total_trades": total_trades,
        "rebalance_trades": rebalance_trades,
        "total_volume": total_volume,
        "avg_trade_size": avg_trade_size,
        "trade_log": trade_log,
        "total_cash_flow": total_cash_flow,
        "total_transaction_cost": total_transaction_cost,
        "total_pnl": total_pnl,
    }


def analyze_multiple_days_pnl_w_changing_stock_map(
    stock_data: pd.DataFrame,
    stock_share_maps: dict,
    start_date: str = None,
    end_date: str = None,
    stk_px_col: str = 'close',
    transaction_cost_bps: float = 5.0,
) -> pd.DataFrame:
    """
    Analyze PnL for multiple days, passing in a changing stock map for each day.

    Parameters:
    -----------
    stock_data : pd.DataFrame
        Minute-level stock data
    hedge_maps : dict
        Dictionary mapping dates to sample_maps
    start_date : str, optional
        Start date for analysis
    end_date : str, optional
        End date for analysis
    stk_px_col : str
        Column name for stock price (default: 'close')
    transaction_cost_bps : float
        Transaction cost in basis points

    Returns:
    --------
    pd.DataFrame : DataFrame containing daily PnL results
    """

    results = []

    # Get date range
    available_dates = sorted(stock_share_maps.keys())
    if start_date:
        available_dates = [d for d in available_dates if d >= start_date]
    if end_date:
        available_dates = [d for d in available_dates if d <= end_date]

    for date in tqdm(available_dates, desc="Calculating daily PnL"):
        if date in stock_share_maps:
            # Calculate PnL for this day
            daily_result = calculate_pnl_w_static_stock_price_map(
                stock_data=stock_data,
                stock_price_map=stock_share_maps[date],
                transaction_cost_bps=transaction_cost_bps,
                stk_px_col=stk_px_col,
            )

            if "error" not in daily_result:
                results.append(daily_result)

    # Convert to DataFrame
    if results:
        df_results = pd.DataFrame(results)

        # Calculate cumulative metrics
        df_results["cumulative_cash_flow"] = df_results["total_cash_flow"].cumsum()
        df_results["cumulative_transaction_cost"] = df_results[
            "total_transaction_cost"
        ].cumsum()
        df_results["cumulative_pnl"] = df_results["cumulative_cash_flow"] - df_results["cumulative_transaction_cost"]

        return df_results
    else:
        return pd.DataFrame()
