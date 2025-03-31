import os
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import yfinance as yf

class DataLoader:
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir
        self.raw_data_patch = os.path.join(data_dir, "raw")
        self.processed_data_patch = os.path.join(data_dir, "processed")

        # create directories if they don't exist
        os.makedirs(self.raw_data_patch, exist_ok = True)
        os.makedirs(self.processed_data_patch, exist_ok = True)

        self.scaler = MinMaxScaler(feature_range = (0, 1))

    def fetch_stock_data(self, ticker: str, start_date: str, end_date: str, interval: str = "1d", save_raw: bool = True) -> pd.DataFrame:
        try:
            # fetch data using yfinance
            df = yf.download(ticker, start = start_date, end = end_date, interval = interval)
            if df.empty:
                raise ValueError(f"No data found for {ticker} in the specified date range")
            
            # save data if requested
            if save_raw:
                raw_file_path = os.path.join(self.raw_data_patch, f"{ticker}_{start_date}_{end_date}")
                df.to_csv(raw_file_path)
                print(f"Raw data saved to {raw_file_path}")

        except Exception as e:
            raise ValueError(f"Error fetching data for {ticker}: {str(e)}")