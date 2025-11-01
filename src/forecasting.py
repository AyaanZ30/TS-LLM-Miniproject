import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class AdvancedForecaster:
    def __init__(self):
        self.models = {}
        self.forecast_history = {}
    
    def arima_forecast(self, series : pd.Series, steps : int = 10, 
    order : Tuple = (2, 1, 2)) -> Dict:
        try:
            model = ARIMA(series, order=order)
            fitted_model = model.fit()
            
            forecast = fitted_model.forecast(steps = steps)
            confidence_intervals = fitted_model.get_forecast(steps=steps).conf_int()
            fitted_values = fitted_model.fittedvalues
            mae = mean_absolute_error(series.iloc[len(fitted_values):], fitted_values)
            rmse = np.sqrt(mean_squared_error(series.iloc[len(fitted_values):], fitted_values))
            
            return {
                'forecast': forecast,
                'confidence_intervals': confidence_intervals,
                'model': fitted_model,
                'metrics': {'MAE': mae, 'RMSE': rmse},
                'fitted_values': fitted_values
            }
        except Exception as e:
            print(f'Error in ARIMA forecasting : {e}')
            return None
    
    def exponential_smoothing_forecast(self, series: pd.Series, steps : int = 10, 
    seasonal_periods: int = None) -> Dict:
        try:
            if seasonal_periods and len(series) > (2 * seasonal_periods):
                model = ExponentialSmoothing(
                    series, 
                    seasonal_periods=seasonal_periods,
                    trend='add', 
                    seasonal='add'
                ) 
            else:
                model = ExponentialSmoothing(series, trend='add')
            
            fitted_model = model.fit()
            forecast = fitted_model.forecast(steps=steps)
            
            fitted_values = fitted_model.fittedvalues
            mae = mean_absolute_error(series, fitted_values)
            rmse = np.sqrt(mean_squared_error(series, fitted_values))
            
            return {
                'forecast': forecast,
                'model': fitted_model,
                'metrics': {'MAE': mae, 'RMSE': rmse},
                'fitted_values': fitted_values
            }
        except Exception as e:
            print(f'Error in ES forecasting : {e}')
            return None
    
    def rolling_forecast(self, series : pd.Series, steps : int = 10,
    window : int = 20) -> Dict:
        forecasts = []
        actuals = []
        confidence_intervals = []
        
        for i in range(len(series) - window - steps):
            train_data = series.iloc[i:i + window]
            test_data = series.iloc[i + window:i + window + steps]
            
            forecast = np.full(steps, train_data.mean())
            forecasts.extend(forecast)
            actuals.extend(test_data)
            
            std = train_data.std()
            ci_lower = forecast - 1.96 * std
            ci_upper = forecast + 1.96 * std
            confidence_intervals.append(pd.DataFrame({
                'lower': ci_lower,
                'upper': ci_upper
            }))
        
        if forecasts:
            mae = mean_absolute_error(actuals, forecasts)
            rmse = np.sqrt(mean_squared_error(actuals, forecasts))
            
            return {
                'forecasts': forecasts,
                'actuals': actuals,
                'confidence_intervals': confidence_intervals,
                'metrics': {'MAE': mae, 'RMSE': rmse}
            }
        return None
    
    def multi_step_forecasting(self, df: pd.DataFrame, 
                            target_column: str,
                            steps: int = 10) -> Dict:
        """Multi-step forecasting for system metrics"""
        results = {}
        series = df[target_column]
        
        # Try different forecasting methods
        arima_result = self.arima_forecast(series, steps)
        es_result = self.exponential_smoothing_forecast(series, steps)
        
        if arima_result:
            results['arima'] = arima_result
        if es_result:
            results['exponential_smoothing'] = es_result
        
        # Select best model based on RMSE
        best_model = None
        best_rmse = float('inf')
        
        for method, result in results.items():
            if result['metrics']['RMSE'] < best_rmse:
                best_rmse = result['metrics']['RMSE']
                best_model = method
        
        return {
            'all_models': results,
            'best_model': best_model,
            'best_forecast': results[best_model]['forecast'] if best_model else None,
            'combined_metrics': {
                method: result['metrics'] 
                for method, result in results.items()
            }
        }
    
    def detect_forecast_anomalies(self, actual: pd.Series, 
    forecast: pd.Series,
    confidence_interval: pd.DataFrame = None,
    threshold: float = 2.0) -> Dict:
        residuals = (actual - forecast)
        std_res = residuals.std()
        
        anomalies = np.abs(residuals) > (threshold * std_res)
        anomaly_details = []
        for i, is_anomaly in enumerate(anomalies):
            if is_anomaly:
                anomaly_details.append({
                    'index': i,
                    'actual': actual.iloc[i],
                    'forecast': forecast.iloc[i],
                    'deviation': residuals.iloc[i],
                    'deviation_std': residuals.iloc[i] / std_res
                })
        
        return {
            'anomalies': anomalies,
            'residuals': residuals,
            'anomaly_details': anomaly_details,
            'threshold': threshold * std_res
        } 