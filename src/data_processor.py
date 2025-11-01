# import pandas as pd
# import numpy as np
# from typing import Tuple, Dict, List
# import re

# class DataProcessor:
#     def __init__(self):
#         self.cleaning_strategies = {
#             'mean' : lambda x : x.fillna(x.mean()),
#             'median' : lambda x : x.fillna(x.median()),
#             'forward_fill' : lambda x : x.ffill().bfill(),
#             'drop': lambda x: x.dropna()
#         }
        
#     def load_and_clean_data(self, file_path : str) -> pd.DataFrame:
#         df = pd.read_csv(file_path)
#         df_clean = self._handle_missing_values(df)
#         df_clean = self._handle_corrupt_data(df_clean)
#         df_clean = self._remove_outliers(df_clean)
#         return df_clean
    
#     def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
#         strategies = {
#             'Disk Write Speed (MB/s)': 'mean',
#             'Disk Read Speed (MB/s)': 'mean',
#             'CPU Usage (%)': 'mean',
#             'CPU Temperature (°C)': 'mean',
#             'Clock Speed (GHz)': 'mean',
#             'Cache Miss Rate (%)': 'mean',
#             'Power Consumption (W)': 'mean'
#         }
#         for col, strategy in strategies.items():
#             if col in df.columns:
#                 df[col] = self.cleaning_strategies[strategy](df[col])
#         return df
    
#     def _handle_corrupt_data(self, df: pd.DataFrame) -> pd.DataFrame:
#         corrupt_indicators = ['ERROR', 'CORRUPT', 'NaN', 'Infinity']
        
#         for col in df.columns:
#             if df[col].dtype == 'object':
#                 # Convert to numeric, coercing errors to NaN
#                 df[col] = pd.to_numeric(df[col], errors='coerce')
#         return df
    
#     def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
#         df_clean = df.copy()
        
#         for col in df_clean.select_dtypes(include=[np.number]).columns:
#             Q1 = df_clean[col].quantile(0.25)
#             Q3 = df_clean[col].quantile(0.75)
#             IQR = Q3 - Q1
#             lower_bound = Q1 - 1.5 * IQR
#             upper_bound = Q3 + 1.5 * IQR
            
#             # Cap outliers instead of removing
#             df_clean[col] = np.where(df_clean[col] < lower_bound, lower_bound, df_clean[col])
#             df_clean[col] = np.where(df_clean[col] > upper_bound, upper_bound, df_clean[col])
#         return df_clean
    
#     def create_derived_features(self, df : pd.DataFrame) -> pd.DataFrame:
#         df_enhanced = df.copy()
#         # deriving new features by formulating
#         df_enhanced['Performance_Efficiency'] =  (
#             df_enhanced['Disk Read Speed (MB/s)'] / 
#             (df_enhanced['CPU Usage (%)'] + 1) * 
#             df_enhanced['Clock Speed (GHz)']
#         )
#         df_enhanced['Power_efficiency'] = (
#             df_enhanced['Disk Write Speed (MB/s)'] / 
#             (df_enhanced['Power Consumption (W)'] + 1)
#         )
#         df_enhanced['Thermal_Efficiency'] = (
#             df_enhanced['CPU Usage (%)'] / 
#             (df_enhanced['CPU Temperature (°C)'] + 1)
#         )
#         df_enhanced['System_Stress_Index'] = (
#             df_enhanced['CPU Usage (%)'] * 0.4 +
#             (df_enhanced['CPU Temperature (°C)'] / 100) * 0.3 +
#             (df_enhanced['Power Consumption (W)'] / 1000) * 0.3
#         )
#         return df_enhanced
    
#     def prepare_time_series_data(self, df: pd.DataFrame) -> pd.DataFrame:
#         # """Prepare data for time series analysis"""
#         # df_ts = df.copy()
#         # df_ts['Timestamp'] = pd.date_range(
#         #     start='2024-01-01', 
#         #     periods=len(df_ts), 
#         #     freq='1min'
#         # )
#         # df['timestamp'] = pd.to_datetime(df['timestamp'])
#         # df_ts.set_index('Timestamp', inplace=True)
#         # return df_ts
        
#         df_ts = df.copy()
        
#         # Create and assign a new DatetimeIndex directly
#         df_ts.index = pd.date_range(
#             start='2024-01-01', 
#             periods=len(df_ts), 
#             freq='1min'  # You can change '1min' to '5min', '1H' etc. if needed
#         )
        
#         # The buggy line that tried to convert a non-existent column is removed.
#         return df_ts
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import re

class DataProcessor:
    def __init__(self):
        self.cleaning_strategies = {
            'mean' : lambda x : x.fillna(x.mean()),
            'median' : lambda x : x.fillna(x.median()),
            'forward_fill' : lambda x : x.ffill().bfill(),
            'drop': lambda x: x.dropna()
        }
        
    def load_and_clean_data(self, uploaded_file) -> pd.DataFrame:
        """
        Loads, cleans, and processes the uploaded CSV file.
        The file_path argument is actually a Streamlit UploadedFile object.
        """
        
        # Check for and fix the 'str + int' bug if it's a debug print
        # For example, if you had print("File size: " + uploaded_file.size),
        # change it to:
        # print(f"File size: {uploaded_file.size}")
        
        df = pd.read_csv(uploaded_file)
        
        # --- LOGIC FIX: SWAPPED ORDER ---
        
        # 1. Handle corrupt data FIRST
        # This converts all 'object' (string) columns to numeric,
        # turning "ERROR" or other strings into NaN.
        df_clean = self._handle_corrupt_data(df)
        
        # 2. Handle missing values SECOND
        # Now that columns are numeric, .mean() and .median() will work
        # to fill the NaNs created in the step above.
        df_clean = self._handle_missing_values(df_clean)
        
        # 3. Remove/cap outliers
        df_clean = self._remove_outliers(df_clean)
        
        return df_clean
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fills missing values based on predefined strategies."""
        strategies = {
            'Disk Write Speed (MB/s)': 'mean',
            'Disk Read Speed (MB/s)': 'mean',
            'CPU Usage (%)': 'mean',
            'CPU Temperature (°C)': 'mean',
            'Clock Speed (GHz)': 'mean',
            'Cache Miss Rate (%)': 'mean',
            'Power Consumption (W)': 'mean'
        }
        for col, strategy in strategies.items():
            if col in df.columns:
                # This check is important. Only fill if the column is numeric.
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = self.cleaning_strategies[strategy](df[col])
                else:
                    # This could happen if a column is all strings
                    print(f"Warning: Column '{col}' is not numeric. Skipping fillna.")
        return df
    
    def _handle_corrupt_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts object-type columns that should be numeric.
        Turns any non-numeric string (e.g., "ERROR") into NaN.
        """
        for col in df.columns:
            # Check if column is object type AND is not all-string by nature
            # (like a 'Hostname' column, which we don't have here)
            # We assume all columns in this app *should* be numeric.
            if df[col].dtype == 'object':
                # Convert to numeric, coercing errors to NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Caps outliers using the 1.5*IQR rule."""
        df_clean = df.copy()
        
        for col in df_clean.select_dtypes(include=[np.number]).columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers instead of removing
            df_clean[col] = np.where(df_clean[col] < lower_bound, lower_bound, df_clean[col])
            df_clean[col] = np.where(df_clean[col] > upper_bound, upper_bound, df_clean[col])
        return df_clean
    
    def create_derived_features(self, df : pd.DataFrame) -> pd.DataFrame:
        """Engineers new features from existing metrics."""
        df_enhanced = df.copy()
        
        # Ensure columns exist before trying to use them
        required_cols = {
            'perf': ['Disk Read Speed (MB/s)', 'CPU Usage (%)', 'Clock Speed (GHz)'],
            'power': ['Disk Write Speed (MB/s)', 'Power Consumption (W)'],
            'thermal': ['CPU Usage (%)', 'CPU Temperature (°C)'],
            'stress': ['CPU Usage (%)', 'CPU Temperature (°C)', 'Power Consumption (W)']
        }
        
        # Use .get(col, pd.Series(0, index=df.index)) as a safe way to handle missing cols
        # Although ideally, the cleaning ensures they are present and numeric.
        
        if all(col in df_enhanced.columns for col in required_cols['perf']):
            df_enhanced['Performance_Efficiency'] =  (
                df_enhanced['Disk Read Speed (MB/s)'] / 
                (df_enhanced['CPU Usage (%)'] + 1) * df_enhanced['Clock Speed (GHz)']
            )
        
        if all(col in df_enhanced.columns for col in required_cols['power']):
            df_enhanced['Power_efficiency'] = (
                df_enhanced['Disk Write Speed (MB/s)'] / 
                (df_enhanced['Power Consumption (W)'] + 1)
            )
            
        if all(col in df_enhanced.columns for col in required_cols['thermal']):
            df_enhanced['Thermal_Efficiency'] = (
                df_enhanced['CPU Usage (%)'] / 
                (df_enhanced['CPU Temperature (°C)'] + 1)
            )
            
        if all(col in df_enhanced.columns for col in required_cols['stress']):
            df_enhanced['System_Stress_Index'] = (
                df_enhanced['CPU Usage (%)'] * 0.4 +
                (df_enhanced['CPU Temperature (°C)'] / 100) * 0.3 +
                (df_enhanced['Power Consumption (W)'] / 1000) * 0.3
            )
            
        return df_enhanced.fillna(0) # Fill any NaNs from division by zero
    
    def prepare_time_series_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepares data for time series analysis by adding a DatetimeIndex.
        This assumes the CSV data is in chronological order.
        """
        df_ts = df.copy()
        
        # Check if index is already datetime
        if not pd.api.types.is_datetime64_any_dtype(df_ts.index):
            # Create and assign a new DatetimeIndex directly
            df_ts.index = pd.date_range(
                start='2024-01-01', 
                periods=len(df_ts), 
                freq='1min'  # Assumes 1-minute frequency
            )
        
        return df_ts
