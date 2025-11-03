import pandas as pd
import numpy as np
from tqdm import tqdm


def calculate_daily_pnl(
    stock_data: pd.DataFrame,
    sample_map: dict,
    target_date: str,
    transaction_cost_bps: float = 5.0,
) -> dict:
    """
    Calculate daily PnL based on delta hedging strategy using minute-level stock data.
    Starting from opening price with initial PnL = 0.

    Parameters:
    -----------
    stock_data : pd.DataFrame
        Minute-level stock data with columns: date, time, close, etc.
    sample_map : dict
        Dictionary mapping stock prices to required share positions
    target_date : str
        Target date for PnL calculation (format: 'YYYY-MM-DD')
    transaction_cost_bps : float
        Transaction cost in basis points (default: 5 bps)

    Returns:
    --------
    dict : Dictionary containing PnL analysis results
    """

    # Filter data for target date
    daily_data = stock_data[stock_data["date"] == target_date].copy()

    if daily_data.empty:
        return {"error": f"No data found for date {target_date}"}

    # Sort by time to ensure chronological order
    daily_data = daily_data.sort_values("time").reset_index(drop=True)

    # Get opening price and establish initial position
    opening_price = daily_data.iloc[0]["close"]

    # Get sorted price levels from sample_map for interpolation
    price_levels = sorted(sample_map.keys())
    target_positions = [sample_map[price] for price in price_levels]

    def get_target_position(current_price):
        """Interpolate target position based on current stock price"""
        if current_price <= price_levels[0]:
            return target_positions[0]
        elif current_price >= price_levels[-1]:
            return target_positions[-1]
        else:
            # Linear interpolation
            return np.interp(current_price, price_levels, target_positions)

    # Establish initial position based on opening price
    initial_position = int(get_target_position(opening_price))
    current_position = initial_position

    # Initialize tracking variables
    cash_flow = (
        -initial_position * opening_price
    )  # Initial cash outflow for establishing position
    total_transaction_cost = abs(initial_position * opening_price) * (
        transaction_cost_bps / 10000
    )
    cash_flow -= total_transaction_cost
    trade_log = []

    # Log the initial trade
    if initial_position != 0:
        trade_log.append(
            {
                "time": daily_data.iloc[0]["time"],
                "price": opening_price,
                "trade_size": initial_position,
                "trade_value": initial_position * opening_price,
                "transaction_cost": total_transaction_cost,
                "position_after": current_position,
                "cumulative_cash_flow": cash_flow,
                "trade_type": "initial_position",
            }
        )

    # Process each subsequent minute (skip the first one as it's already processed)
    for idx, row in daily_data.iloc[1:].iterrows():
        current_price = row["close"]
        current_time = row["time"]

        # Get target position based on current price
        target_position = int(get_target_position(current_price))

        # Calculate required trade
        trade_size = target_position - current_position

        # Execute trade if needed
        if trade_size != 0:
            trade_value = trade_size * current_price
            transaction_cost = abs(trade_value) * (transaction_cost_bps / 10000)

            # Update positions and cash flow
            current_position += trade_size
            cash_flow -= trade_value  # Negative for buying, positive for selling
            cash_flow -= transaction_cost
            total_transaction_cost += transaction_cost

            # Log the trade
            trade_log.append(
                {
                    "time": current_time,
                    "price": current_price,
                    "trade_size": trade_size,
                    "trade_value": trade_value,
                    "transaction_cost": transaction_cost,
                    "position_after": current_position,
                    "cumulative_cash_flow": cash_flow,
                    "trade_type": "rebalance",
                }
            )

    # Calculate final PnL
    if not daily_data.empty:
        final_price = daily_data.iloc[-1]["close"]

        # Mark-to-market the final position
        final_position_value = current_position * final_price

        # Total PnL = cash flow + final position value
        # Since we started with PnL = 0, this represents the total P&L for the day
        total_pnl = cash_flow + final_position_value

        # Calculate some statistics
        total_trades = len(trade_log)
        rebalance_trades = len([t for t in trade_log if t["trade_type"] == "rebalance"])
        total_volume = sum(abs(trade["trade_size"]) for trade in trade_log)
        avg_trade_size = total_volume / total_trades if total_trades > 0 else 0

        price_change = final_price - opening_price
        price_change_pct = (price_change / opening_price) * 100

    else:
        final_price = opening_price
        final_position_value = total_pnl = 0
        total_trades = rebalance_trades = total_volume = avg_trade_size = 0
        price_change = price_change_pct = 0

    # Return comprehensive results
    return {
        "date": target_date,
        "opening_price": opening_price,
        "final_price": final_price,
        "price_change": price_change,
        "price_change_pct": price_change_pct,
        "initial_position": initial_position,
        "final_position": current_position,
        "position_change": current_position - initial_position,
        "final_position_value": final_position_value,
        "cash_flow": cash_flow,
        "total_pnl": total_pnl,
        "total_transaction_cost": total_transaction_cost,
        "net_pnl": total_pnl - total_transaction_cost,
        "total_trades": total_trades,
        "rebalance_trades": rebalance_trades,
        "total_volume": total_volume,
        "avg_trade_size": avg_trade_size,
        "trade_log": trade_log,
        "pnl_breakdown": {
            "cash_flow": cash_flow,
            "position_value": final_position_value,
            "total_pnl": total_pnl,
            "transaction_costs": -total_transaction_cost,
            "net_pnl": total_pnl - total_transaction_cost,
        },
    }


def analyze_multiple_days_pnl(
    stock_data: pd.DataFrame,
    hedge_maps: dict,
    start_date: str = None,
    end_date: str = None,
    transaction_cost_bps: float = 5.0,
) -> pd.DataFrame:
    """
    Analyze PnL for multiple days.

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
    transaction_cost_bps : float
        Transaction cost in basis points

    Returns:
    --------
    pd.DataFrame : DataFrame containing daily PnL results
    """

    results = []
    current_position = 0

    # Get date range
    available_dates = sorted(hedge_maps.keys())
    if start_date:
        available_dates = [d for d in available_dates if d >= start_date]
    if end_date:
        available_dates = [d for d in available_dates if d <= end_date]

    for date in tqdm(available_dates, desc="Calculating daily PnL"):
        if date in hedge_maps:
            sample_map = hedge_maps[date]

            # Calculate PnL for this day
            daily_result = calculate_daily_pnl(
                stock_data=stock_data,
                sample_map=sample_map,
                target_date=date,
                transaction_cost_bps=transaction_cost_bps,
            )

            if "error" not in daily_result:
                results.append(daily_result)
                # Update position for next day
                current_position = daily_result["final_position"]

    # Convert to DataFrame
    if results:
        df_results = pd.DataFrame(results)

        # Calculate cumulative metrics
        df_results["cumulative_pnl"] = df_results["net_pnl"].cumsum()
        df_results["cumulative_transaction_cost"] = df_results[
            "total_transaction_cost"
        ].cumsum()

        return df_results
    else:
        return pd.DataFrame()
