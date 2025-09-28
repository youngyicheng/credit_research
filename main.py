
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import yaml
import json
import time
from datetime import datetime
import multiprocessing as mp
import statsmodels.api as sm
from functools import partial
# import dolphindb as ddb
from itertools import chain, product 
from threadpoolctl import threadpool_limits 
from sklearn.linear_model import LinearRegression
import asyncpg
# from PNL import *
from flib.utils import read_supplementary_data, read_target_return, transform_column_type
from flib import DailyReader
from utils import *
import flib
import random
# pd.set_option('display.max_rows', None)  # 显示所有行
# from PNL import PNL_STOCK
warnings.filterwarnings("ignore")
# from pilot import *
from tqdm import tqdm
from implied_vol_solver import CDSImpliedVolatilitySolver
import sys
from io import StringIO
from contextlib import redirect_stdout
import logging
logging.getLogger('implied_vol_solver').setLevel(logging.WARNING)  # 只显示警告和错误



# 
def load_cds_data(path):
    """
    load cds data from parquet file.
    Args:
        path (str): The path of the parquet file.
        
    Returns:
    """
    path_lst = os.listdir(path)
    df_all = pd.DataFrame()
    for path  in tqdm(path_lst):
        a = pd.read_parquet(f'/home/yicheng/credit/data/{path}')
        use_lst = ['ticker','date','tenor','parspread','convspreard','upfront', 'runningcoupon','primarycoupon', 'cdsrealrecovery','cdsassumedrecovery','carriedforward', 'compositedepth5y',]
        b = a.loc[a['tenor']=='5Y']
        temp = b[use_lst]
        # temp['path'] = path
        df_all = pd.concat([df_all,temp])

    # this place we can not simply use first() we need to adjust some something according to other columns in data
    df_use = df_all.sort_values(by=['date',],inplace=False).groupby('date').first()
    return df_use


def read_financial_data(field_name, table_name ,start_date, end_date):
    """
    Read financial data from DolphinDB database.
    
    Args:
        field_name (str): The name of the field to read.
        start_date (str): The start date of the data to read.
        end_date (str): The end date of the data to read.
        
    Returns:
    """ 
    data = DailyReader(rf'sp_financials_{table_name}').get_data(field_name, column_type='tradingitemid').loc[start_date:end_date]
    return data


def load_financial_data(df_use,start_date, end_date):
    """
    load financial data and modify some index
    Args:
        df_use (pd.DataFrame): The DataFrame to use.
        start_date (str): The start date of the data to read.
        end_date (str): The end date of the data to read.
        
    Returns:
    """
    long_term_debt = read_financial_data('financials_1049_ytd', 'ytd', start_date, end_date)
    current_asset = read_financial_data('financials_1008_ytd', 'ytd', start_date, end_date)
    current_asset_short_term_ratio = read_financial_data('financials_43901_ytd', 'ytd', start_date, end_date)
    shares = read_financial_data('financials_1070_ytd', 'ytd', start_date, end_date)
    short_term_debt = current_asset / current_asset_short_term_ratio
    financial_debt = long_term_debt + short_term_debt
    financial_debt_ratio = financial_debt / shares

    close = DailyReader('sp_stock_daily_2010').get_data('close')
    close_ticker = transform_column_type(df = close, column_type='ticker')
    financial_debt_ratio = transform_column_type(df = financial_debt_ratio, column_type='ticker')
    df_use.index = pd.to_datetime(df_use.index)

    # 使用reindex替代loc，自动处理缺失索引
    close_ticker_aligned = close_ticker.reindex(df_use.index)
    financial_debt_aligned = financial_debt_ratio.reindex(df_use.index)
    close_ticker_aligned['XRX'].name = 'stock_price'
    financial_debt_aligned['XRX'].name = 'financial_debt_ratio'

    return close_ticker_aligned['XRX'] , financial_debt_aligned['XRX']



def data_concat(df_use,start_date, end_date):
    """
    concat cds data and financial data.
    Args:
        df_use (pd.DataFrame): The DataFrame to use.
        start_date (str): The start date of the data to read.
        end_date (str): The end date of the data to read.
        
    Returns:
    """
    close_ticker_aligned, financial_debt_aligned = load_financial_data(df_use, start_date, end_date)
    df = pd.concat([df_use,close_ticker_aligned,financial_debt_aligned],axis=1).dropna()
    return df



def MultiDay_CDSImpliedVolatilitySolver(df):
    """
    solve the implied volatility of the cds data.
    Args:
        df (pd.DataFrame): The DataFrame to use.
        
    Returns:
    """

    # D = 150.0        # debt per share
    D = 150.0        # debt per share
    t = 5.0          # CDS tenor
    r = 0.05         # risk free rate
    R = 0.4          # recovery rate
    L = 0.5          # loss given default
    lamb = 0.3       # barrier deviation
    notional = 100000000        
    cds_coupon = 0.01

    # create new columns
    df['implied_vol'] = np.nan 
    df['par_spread_error'] = np.nan
    solver = CDSImpliedVolatilitySolver(t, r, L, lamb, notional, cds_coupon)

    with redirect_stdout(StringIO()):
        for i, row in tqdm(df.iterrows()):
            # vol calculation by upfront price
            # this place we should use upfront instead of parspread considering the coupon payment 
            market_cds_spread = row['upfront']  
            implied_vol,error = solver.solve_implied_volatility(row['stock_price'], row['financial_debt_ratio'],market_cds_spread, row['cdsassumedrecovery'], method='brent');
            df.at[i, 'implied_vol'] = implied_vol
            df.at[i, 'par_spread_error'] = error


    return df
            # break

def greek_calc(df):
    """
    calculate the greek of the cds data.
    Args:
        df (pd.DataFrame): The DataFrame to use.
        
    Returns:
    """
    # return df
    # delta calc


    t = 5.0          # CDS期限
    r = 0.05         # 无风险利率
    R = 0.4          # 回收率
    L = 0.5          # 债务回收率
    lamb = 0.3       # 违约壁垒参数
    notional = 100000000        
    cds_coupon = 0.01


    solver = CDSImpliedVolatilitySolver(t, r, L, lamb, notional, cds_coupon)
    df['delta'] = np.nan
    df['gamma'] = np.nan

    with redirect_stdout(StringIO()):
        for i, row in tqdm(df.iterrows()):
            # param setting
            S = row['stock_price']
            h = max(1e-4 * S, 1.0)   # 例如：价格的万分之一与 1 取大者

            f_up = solver.calculate_cds_spread_continuous(S + h, row['financial_debt_ratio'],row['implied_vol'], row['cdsassumedrecovery'])

            # f(S) —— 你已有的市场或基准值
            f_mid = row['upfront']

            # f(S-h)
            f_dn = solver.calculate_cds_spread_continuous(S - h, row['financial_debt_ratio'],row['implied_vol'], row['cdsassumedrecovery'])

            # 中心差分 Delta / Gamma
            delta = (f_up - f_dn) / (2.0 * h)
            gamma = (f_up - 2.0 * f_mid + f_dn) / (h ** 2)

            df.at[i, 'delta'] = delta
            df.at[i, 'gamma'] = gamma

    return df


if __name__ == '__main__':
    df = load_cds_data('/home/yicheng/credit/data')
    df = data_concat(df, '2021-01-01', '2025-01-01')
    df = MultiDay_CDSImpliedVolatilitySolver(df)
    df = greek_calc(df)
    # print(df)
    ticker = str(df['ticker'].iloc[0])
    df.to_csv(f'res/{ticker}.csv')