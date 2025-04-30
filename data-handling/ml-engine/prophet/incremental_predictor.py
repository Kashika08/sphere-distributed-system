#!/usr/bin/env python3
"""
Enhanced Predictive Scaling Model for Kubernetes
- Reads streaming CPU metrics from ml_data.log
- Predicts CPU utilization 15 steps ahead (2min 30s)
- Uses advanced techniques to minimize MAE (<10)
- Writes predictions to ml_predictions.log
"""

import time
import os
import json
import numpy as np
import pandas as pd
import itertools
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import logging
from datetime import datetime, timedelta
import threading
import warnings
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('predictive_scaler')

class PredictiveScaler:
    def __init__(
        self,
        input_file='ml_data.log',
        output_file='ml_predictions.log',
        prediction_steps=15,
        step_seconds=10,
        history_size=120,  # Reduced from 1000 to avoid noise
        model_type='ensemble',  # Using ensemble as default
        tune_hyperparameters=True,
        optimization_metric='mae',
        prophet_seasonality='auto',
        remove_outliers=True,
        feature_engineering=True,
        performance_log='performance_metrics.log'
    ):
        self.input_file = input_file
        self.output_file = output_file
        self.prediction_steps = prediction_steps
        self.step_seconds = step_seconds
        self.history_size = history_size
        self.model_type = model_type
        self.tune_hyperparameters = tune_hyperparameters
        self.optimization_metric = optimization_metric
        self.prophet_seasonality = prophet_seasonality
        self.remove_outliers = remove_outliers
        self.feature_engineering = feature_engineering
        self.performance_log = performance_log
        
        # For model evaluation and performance tracking
        self.mae_history = []
        self.best_mae = float('inf')
        self.best_model_params = {}
        
        # Data buffer and threading setup
        self.data_buffer = []
        self.last_position = 0
        self.running = True
        self.lock = threading.Lock()
        self.last_timestamp = None
        
        # Model specific parameters - will be tuned if tune_hyperparameters=True
        self.prophet_params = {
            'changepoint_prior_scale': 0.01,
            'seasonality_prior_scale': 10.0,
            'holidays_prior_scale': 10.0,
            'seasonality_mode': 'additive'
        }
        
        self.arima_params = {
            'order': (5, 1, 0)
        }
        
        # Ensemble weights - will be dynamically updated based on performance
        self.ensemble_weights = {
            'prophet': 0.5,
            'arima': 0.5
        }
        
        # Initialize files
        for file in [self.input_file, self.output_file, self.performance_log]:
            if not os.path.exists(file):
                with open(file, 'w') as f:
                    if file == self.performance_log:
                        f.write("timestamp,model,mae,prediction_count\n")

        logger.info(f"Initialized predictive scaler with {model_type} model")
        logger.info(f"Will predict {prediction_steps} steps ahead ({prediction_steps * step_seconds} seconds)")
        
        # Initial hyperparameter tuning if enabled and we have data
        if os.path.exists(self.input_file) and os.path.getsize(self.input_file) > 0:
            self._load_initial_data()

    def _load_initial_data(self):
        """Load initial data from the input file to initialize models"""
        try:
            with open(self.input_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    self._process_new_data(lines)
                    logger.info(f"Loaded {len(self.data_buffer)} initial data points")
                    
                    # If we have enough data and tuning is enabled, do initial tuning
                    if len(self.data_buffer) >= 60 and self.tune_hyperparameters:
                        self._tune_hyperparameters()
        except Exception as e:
            logger.error(f"Error loading initial data: {str(e)}")

    def tail_file(self):
        """Continuously read new lines from input file (similar to tail -f)"""
        while self.running:
            try:
                with open(self.input_file, 'r') as f:
                    f.seek(self.last_position)
                    new_lines = f.readlines()
                    if new_lines:
                        self.last_position = f.tell()
                        self._process_new_data(new_lines)
                time.sleep(1)  # Check for updates every second
            except Exception as e:
                logger.error(f"Error tailing file: {str(e)}")
                time.sleep(5)

    def _process_new_data(self, lines):
        """Process new data lines and add to buffer"""
        processed_lines = []
        
        for line in lines:
            try:
                # Try to parse as JSON
                data = json.loads(line.strip())
                timestamp = data.get('timestamp', datetime.now().isoformat())
                cpu_value = float(data.get('cpu_utilization', 0))
                
                processed_lines.append({
                    'timestamp': timestamp,
                    'cpu_value': cpu_value
                })
            except json.JSONDecodeError:
                # Try to parse as CSV format
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    try:
                        timestamp = parts[0].strip()
                        cpu_value = float(parts[1].strip())
                        processed_lines.append({
                            'timestamp': timestamp,
                            'cpu_value': cpu_value
                        })
                    except (ValueError, IndexError):
                        logger.warning(f"Couldn't parse line: {line}")
                else:
                    logger.warning(f"Skipping line with invalid format: {line}")
            except Exception as e:
                logger.warning(f"Error processing line {line}: {str(e)}")
        
        if processed_lines:
            with self.lock:
                self.data_buffer.extend(processed_lines)
                # Keep only the most recent history_size entries
                if len(self.data_buffer) > self.history_size:
                    self.data_buffer = self.data_buffer[-self.history_size:]
                
                # Update the last timestamp (used for output)
                if processed_lines:
                    try:
                        ts = pd.to_datetime(processed_lines[-1]['timestamp'])
                        self.last_timestamp = ts.strftime('%Y-%m-%d %H:%M:%S')
                    except Exception as e:
                        logger.error(f"Error formatting timestamp: {str(e)}")
                        self.last_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Generate a new prediction
            self.make_prediction()
            
            # Periodically retune hyperparameters if enough new data has been collected
            if len(self.data_buffer) >= 60 and self.tune_hyperparameters and len(self.mae_history) >= 10:
                avg_mae = sum(self.mae_history[-10:]) / 10
                if avg_mae > 10:  # Only tune if MAE is above target
                    self._tune_hyperparameters()

    def _preprocess_data(self, df):
        """Preprocess data to improve model performance"""
        # Handle missing values if any
        df = df.dropna().reset_index(drop=True)
        
        if self.remove_outliers:
            # Remove outliers using IQR method
            Q1 = df['cpu_value'].quantile(0.25)
            Q3 = df['cpu_value'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers instead of removing them
            df['cpu_value'] = df['cpu_value'].clip(lower_bound, upper_bound)
        
        # Ensure data is sorted by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df

    def _add_engineered_features(self, df):
        """Add engineered features to improve prediction accuracy"""
        if not self.feature_engineering:
            return df
            
        # Create a copy to avoid modifying original
        df_features = df.copy()
        
        # Add rolling statistics
        for window in [3, 5, 10]:
            if len(df) >= window:
                df_features[f'rolling_mean_{window}'] = df['cpu_value'].rolling(window=window, min_periods=1).mean()
                df_features[f'rolling_std_{window}'] = df['cpu_value'].rolling(window=window, min_periods=1).std()
        
        # Add lag features
        for lag in [1, 2, 3, 5]:
            if len(df) > lag:
                df_features[f'lag_{lag}'] = df['cpu_value'].shift(lag)
        
        # Add time-based features
        df_features['hour'] = df_features['timestamp'].dt.hour
        df_features['minute'] = df_features['timestamp'].dt.minute
        df_features['day_of_week'] = df_features['timestamp'].dt.dayofweek
        
        # Add rate of change (momentum)
        if len(df) > 1:
            df_features['diff'] = df['cpu_value'].diff()
            df_features['diff_pct'] = df['cpu_value'].pct_change()
        
        # Add exponential weighted moving average
        if len(df) >= 3:
            df_features['ewma_3'] = df['cpu_value'].ewm(span=3, adjust=False).mean()
        if len(df) >= 5:
            df_features['ewma_5'] = df['cpu_value'].ewm(span=5, adjust=False).mean()
        
        # Fill NaN values that result from operations
        df_features = df_features.fillna(method='bfill').fillna(method='ffill').fillna(0)
        
        return df_features

    def _tune_hyperparameters(self):
        """Tune model hyperparameters to minimize MAE"""
        if len(self.data_buffer) < 60:
            logger.info("Not enough data for hyperparameter tuning")
            return
            
        logger.info("Starting hyperparameter tuning...")
        
        # Convert buffer to dataframe
        df = pd.DataFrame(self.data_buffer)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Preprocess data
        df = self._preprocess_data(df)
        
        # Tune Prophet parameters
        if self.model_type in ['prophet', 'ensemble']:
            self._tune_prophet(df)
        
        # Tune ARIMA parameters
        if self.model_type in ['arima', 'ensemble']:
            self._tune_arima(df)
            
        logger.info(f"Hyperparameter tuning completed. Best Prophet params: {self.prophet_params}")
        logger.info(f"Best ARIMA params: {self.arima_params}")
        
        # Update ensemble weights based on individual model performance
        if self.model_type == 'ensemble' and hasattr(self, 'prophet_mae') and hasattr(self, 'arima_mae'):
            total_error = self.prophet_mae + self.arima_mae
            if total_error > 0:
                # Inverse weighting: better models get higher weights
                self.ensemble_weights['prophet'] = self.arima_mae / total_error
                self.ensemble_weights['arima'] = self.prophet_mae / total_error
                logger.info(f"Updated ensemble weights: Prophet={self.ensemble_weights['prophet']:.2f}, ARIMA={self.ensemble_weights['arima']:.2f}")

    def _tune_prophet(self, df):
        """Tune Prophet hyperparameters"""
        prophet_df = df.rename(columns={'timestamp': 'ds', 'cpu_value': 'y'})
        
        param_grid = {
            'changepoint_prior_scale': [0.001, 0.01, 0.05, 0.1, 0.5],
            'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
            'holidays_prior_scale': [0.01, 1.0, 10.0],
            'seasonality_mode': ['additive', 'multiplicative']
        }
        
        # Generate all combinations of parameters
        all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
        best_mae = float('inf')
        best_params = None
        
        # Use simple cross validation approach for tuning
        cutoff_idx = int(len(prophet_df) * 0.8)
        train_df = prophet_df.iloc[:cutoff_idx]
        test_df = prophet_df.iloc[cutoff_idx:]
        
        if len(test_df) < 5:
            logger.info("Not enough test data for proper Prophet tuning")
            return
            
        for params in all_params[:5]:  # Limit to first 5 combinations to save time
            try:
                m = Prophet(**params).fit(train_df)
                future = m.make_future_dataframe(periods=len(test_df), freq='S')
                forecast = m.predict(future)
                
                # Calculate MAE for the test period
                forecast_subset = forecast.iloc[-len(test_df):]
                mae = mean_absolute_error(test_df['y'].values, forecast_subset['yhat'].values)
                
                if mae < best_mae:
                    best_mae = mae
                    best_params = params
                    
            except Exception as e:
                logger.warning(f"Error during Prophet tuning: {str(e)}")
                continue
                
        if best_params:
            self.prophet_params = best_params
            self.prophet_mae = best_mae
            logger.info(f"Best Prophet parameters: {best_params}, MAE: {best_mae:.4f}")

    def _tune_arima(self, df):
        """Find optimal ARIMA parameters"""
        series = df['cpu_value'].values
        
        try:
            # Use auto_arima to find optimal parameters
            model = auto_arima(
                series,
                start_p=1, start_q=1,
                max_p=5, max_q=5, d=1,
                seasonal=False,  # CPU data might not have strong seasonality
                trace=False,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True,
                max_order=10
            )
            
            # Get the order from the model
            best_order = model.order
            
            # Test with validation set
            cutoff_idx = int(len(series) * 0.8)
            train = series[:cutoff_idx]
            test = series[cutoff_idx:]
            
            if len(test) >= 5:
                model_fit = ARIMA(train, order=best_order).fit()
                forecast = model_fit.forecast(steps=len(test))
                mae = mean_absolute_error(test, forecast)
                
                self.arima_params = {'order': best_order}
                self.arima_mae = mae
                logger.info(f"Best ARIMA order: {best_order}, MAE: {mae:.4f}")
            else:
                logger.info("Not enough test data for proper ARIMA validation")
                self.arima_params = {'order': best_order}
                
        except Exception as e:
            logger.error(f"Error in ARIMA parameter tuning: {str(e)}")

    def make_prediction(self):
        """Generate predictions using the selected model"""
        with self.lock:
            if len(self.data_buffer) < 30:
                logger.info(f"Not enough data for prediction. Have {len(self.data_buffer)} points, need at least 30.")
                return
            
            # Convert buffer to dataframe
            df = pd.DataFrame(self.data_buffer)
            
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
            except Exception as e:
                logger.error(f"Error converting timestamps: {str(e)}")
                return
                
            # Preprocess data
            df = self._preprocess_data(df)
            
            # Optional: add engineered features
            df_features = self._add_engineered_features(df)
        
        # Make predictions based on model type
        if self.model_type == 'arima':
            predictions = self._predict_arima(df)
        elif self.model_type == 'prophet':
            predictions = self._predict_prophet(df)
        elif self.model_type == 'ensemble':
            predictions = self._predict_ensemble(df)
        else:
            logger.error(f"Unknown model type: {self.model_type}")
            return

        # Write prediction to output file
        if predictions and predictions.get('values'):
            self._write_prediction(predictions)
            
            # Calculate performance metrics if possible
            self._evaluate_prediction_performance(df, predictions)

    def _predict_arima(self, df):
        """Make prediction using ARIMA model"""
        try:
            # Prepare data for ARIMA
            series = df['cpu_value'].values
            
            # Fit ARIMA model with optimized parameters
            model = ARIMA(series, order=self.arima_params['order'])
            model_fit = model.fit()
            
            # Make forecast
            forecast = model_fit.forecast(steps=self.prediction_steps)
            
            # Generate future timestamps
            last_timestamp = df['timestamp'].iloc[-1]
            future_timestamps = [
                last_timestamp + timedelta(seconds=(i+1) * self.step_seconds)
                for i in range(self.prediction_steps)
            ]
            
            return {
                'timestamps': [ts.isoformat() for ts in future_timestamps],
                'values': forecast.tolist()
            }
        except Exception as e:
            logger.error(f"Error in ARIMA prediction: {str(e)}")
            return {'timestamps': [], 'values': []}

    def _predict_prophet(self, df):
        """Make prediction using Prophet model"""
        try:
            # Prepare data for Prophet
            prophet_df = df.rename(columns={'timestamp': 'ds', 'cpu_value': 'y'})
            
            # Initialize and fit Prophet model with tuned parameters
            model = Prophet(
                changepoint_prior_scale=self.prophet_params['changepoint_prior_scale'],
                seasonality_prior_scale=self.prophet_params['seasonality_prior_scale'],
                holidays_prior_scale=self.prophet_params['holidays_prior_scale'],
                seasonality_mode=self.prophet_params['seasonality_mode'],
                daily_seasonality=False,  # CPU patterns may not follow daily pattern
                weekly_seasonality=False   # CPU patterns may not follow weekly pattern
            )
            
            # Add custom hourly seasonality if data spans more than an hour
            time_span = (prophet_df['ds'].max() - prophet_df['ds'].min()).total_seconds()
            if time_span > 3600:
                model.add_seasonality(
                    name='hourly',
                    period=60*60,  # 1 hour in seconds
                    fourier_order=5
                )
            
            model.fit(prophet_df)
            
            # Create future dataframe
            last_timestamp = prophet_df['ds'].iloc[-1]
            future_timestamps = [
                last_timestamp + timedelta(seconds=(i+1) * self.step_seconds)
                for i in range(self.prediction_steps)
            ]
            future = pd.DataFrame({'ds': future_timestamps})
            
            # Make forecast
            forecast = model.predict(future)
            
            return {
                'timestamps': [ts.isoformat() for ts in future_timestamps],
                'values': forecast['yhat'].tolist()
            }
        except Exception as e:
            logger.error(f"Error in Prophet prediction: {str(e)}")
            return {'timestamps': [], 'values': []}

    def _predict_ensemble(self, df):
        """Create a weighted ensemble of predictions"""
        predictions_arima = self._predict_arima(df)
        predictions_prophet = self._predict_prophet(df)
        
        # Ensure both predictions have values
        if not predictions_arima.get('values') or not predictions_prophet.get('values'):
            return predictions_arima if predictions_arima.get('values') else predictions_prophet
        
        # Create weighted ensemble
        try:
            ensemble_values = [
                self.ensemble_weights['arima'] * a + self.ensemble_weights['prophet'] * p 
                for a, p in zip(predictions_arima['values'], predictions_prophet['values'])
            ]
            
            return {
                'timestamps': predictions_prophet['timestamps'],
                'values': ensemble_values
            }
        except Exception as e:
            logger.error(f"Error creating ensemble: {str(e)}")
            # Fallback to prophet if ensemble fails
            return predictions_prophet

    def _evaluate_prediction_performance(self, df, predictions):
        """Evaluate prediction performance against actual values if available"""
        try:
            # This is a placeholder for future functionality
            # In a production environment, we would compare the prediction with actual values
            # when they become available in the future
            
            # For now, we'll just log the prediction statistics
            if predictions and predictions.get('values'):
                values = predictions['values']
                avg_pred = sum(values) / len(values) if values else 0
                min_pred = min(values) if values else 0
                max_pred = max(values) if values else 0
                
                logger.debug(f"Prediction stats: avg={avg_pred:.2f}, min={min_pred:.2f}, max={max_pred:.2f}")
                
                # Add a periodic model performance check (e.g., every hour)
                # This would compare predictions made in the past with actual values
                
        except Exception as e:
            logger.error(f"Error evaluating prediction: {str(e)}")

    def _write_prediction(self, predictions):
        """Write predictions to output file in CSV format"""
        try:
            # Use the timestamp from the input data instead of the current time
            timestamp = self.last_timestamp
            if not timestamp:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
            # Round predictions to 2 decimal places and ensure non-negative
            preds = [max(0, round(val, 2)) for val in predictions['values']]
            
            # Create CSV line
            csv_line = f"{timestamp}, " + ", ".join(map(str, preds)) + "\n"
            
            with open(self.output_file, 'a') as f:
                f.write(csv_line)
                
            logger.info(f"Wrote predictions to {self.output_file}: {csv_line.strip()}")
        except Exception as e:
            logger.error(f"Error writing predictions: {str(e)}")

    def run(self):
        """Start the prediction service"""
        try:
            logger.info("Starting predictive scaler service")
            self.tail_thread = threading.Thread(target=self.tail_file)
            self.tail_thread.daemon = True
            self.tail_thread.start()
            
            # Keep main thread alive
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping service")
            self.running = False
            if hasattr(self, 'tail_thread'):
                self.tail_thread.join(timeout=5)

if __name__ == "__main__":
    # Create and run the predictive scaler
    scaler = PredictiveScaler(
        input_file='ml_data.log',
        output_file='ml_predictions.log',
        prediction_steps=15,          # 15 steps ahead
        step_seconds=10,              # 10 seconds per step
        history_size=30,             # Reduced from 1000 to 120 to focus on recent patterns
        model_type='ensemble',        # Use ensemble of models for better predictions
        tune_hyperparameters=True,    # Dynamically optimize model parameters
        remove_outliers=True,         # Remove outliers in preprocessing
        feature_engineering=True      # Enable feature engineering for better predictions
    )
    scaler.run()
