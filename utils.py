import pandas as pd
from flib import DailyReader



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