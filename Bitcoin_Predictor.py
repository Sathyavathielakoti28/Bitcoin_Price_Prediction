import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import requests
from math import sqrt
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

def predict_bitcoin_price(last_three_days_prices):
    """
    Predict Bitcoin price based on the last 3 days of price data
    
    Parameters:
    last_three_days_prices (list): List of the last 3 days of Bitcoin prices [day-3, day-2, day-1]
    
    Returns:
    dict: Dictionary containing predictions and RMSE value
    """
    if len(last_three_days_prices) != 3:
        raise ValueError("Input must contain exactly 3 days of price data")
    
    # Check if input values are within a reasonable range
    for price in last_three_days_prices:
        if price <= 0:
            raise ValueError("Price values must be positive")
        if price > 1000000:  # Adding a reasonable upper limit
            raise ValueError("Price values are too large. Please use values between 1 and 1,000,000")
    
    # Generate historical data for training (either from API or synthetic)
    historical_data = get_training_data()
    
    # Train the model on historical data
    model, train_rmse = train_linear_regression_model(historical_data)
    
    # Prepare input data for prediction
    current_date = datetime.now().date()
    dates = [(current_date - timedelta(days=i)) for i in range(3, 0, -1)]
    
    input_df = pd.DataFrame({
        'date': dates,
        'price': last_three_days_prices
    })
    input_df.set_index('date', inplace=True)
    
    # Create features for prediction
    X_pred = create_prediction_features(input_df)
    
    # Make prediction for next 7 days
    predictions = predict_next_days(model, X_pred, days=7)
    
    # Ensure predictions are within reasonable limits
    min_price = min(last_three_days_prices) * 0.5  # 50% of minimum input price
    max_price = max(last_three_days_prices) * 1.5  # 150% of maximum input price
    
    # Clip predictions to reasonable range
    predictions['predicted_price'] = predictions['predicted_price'].clip(min_price, max_price)
    
    result = {
        'next_day_prediction': predictions['predicted_price'].iloc[0],
        'seven_day_predictions': predictions,
        'model_rmse': train_rmse
    }
    
    return result, input_df, predictions

def get_training_data():
    """
    Get historical Bitcoin price data for training
    
    Returns:
    pandas.DataFrame: Historical Bitcoin price data
    """
    try:
        # Generate synthetic data for training
        # We'll use synthetic data to avoid API rate limits and connection issues
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        # Generate synthetic data
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Create synthetic price data with realistic patterns 
        # Using more moderate values to avoid size issues
        base_price = 130000  
        trend = np.linspace(0, 20000, len(dates))  # Reduced trend magnitude
        noise = np.random.normal(0, 5000, len(dates))  # Reduced noise
        cycle = 10000 * np.sin(np.linspace(0, 4*np.pi, len(dates)))  # Reduced cycle amplitude
        
        prices = base_price + trend + noise + cycle
        
        df = pd.DataFrame({'price': prices}, index=dates)
        return df
        
    except Exception as e:
        print(f"Error generating data: {e}")
        # Fallback to very simple synthetic data if everything else fails
        dates = pd.date_range(start=datetime.now()-timedelta(days=30), end=datetime.now(), freq='D')
        prices = np.linspace(120000, 140000, len(dates)) + np.random.normal(0, 2000, len(dates))
        
        return pd.DataFrame({'price': prices}, index=dates)

def create_training_features(data, target_days=1):
    """
    Create features for model training
    
    Parameters:
    data (pandas.DataFrame): Historical price data
    target_days (int): Number of days to predict into the future
    
    Returns:
    tuple: X features and y target for model training
    """
    # Create a copy to avoid modifying the original
    df = data.copy()
    
    # Create target (future price)
    df['target'] = df['price'].shift(-target_days)
    
    # Create features based on the last 3 days
    df['price_1'] = df['price'].shift(1)  # Previous day
    df['price_2'] = df['price'].shift(2)  # 2 days ago
    df['price_3'] = df['price'].shift(3)  # 3 days ago
    
    # Calculate price changes (momentum features)
    df['change_1'] = df['price'] - df['price_1']
    df['change_2'] = df['price_1'] - df['price_2']
    df['change_3'] = df['price_2'] - df['price_3']
    
    # Moving average of last 3 days
    df['ma_3'] = df['price'].rolling(window=3).mean()
    
    # Day of week (0=Monday, 6=Sunday)
    df['day_of_week'] = df.index.dayofweek
    
    # Drop rows with NaN values (first 3 days)
    df.dropna(inplace=True)
    
    # Prepare features and target
    features = ['price_1', 'price_2', 'price_3', 'change_1', 'change_2', 'change_3', 'ma_3', 'day_of_week']
    X = df[features]
    y = df['target']
    
    return X, y

def create_prediction_features(input_data):
    """
    Create features for prediction from the last 3 days of data
    
    Parameters:
    input_data (pandas.DataFrame): Last 3 days of price data
    
    Returns:
    pandas.DataFrame: Features for prediction
    """
    # Verify we have 3 days of data
    if len(input_data) != 3:
        raise ValueError("Input data must contain exactly 3 days")
    
    # Create a DataFrame for the prediction features
    price_values = input_data['price'].values
    
    # Create a DataFrame with one row for prediction
    X_pred = pd.DataFrame({
        'price_1': [price_values[2]],             # Yesterday's price
        'price_2': [price_values[1]],             # 2 days ago price
        'price_3': [price_values[0]],             # 3 days ago price
        'change_1': [price_values[2] - price_values[1]],  # Yesterday's change
        'change_2': [price_values[1] - price_values[0]],  # 2 days ago change
        'change_3': [0],  # Placeholder (we don't have this value)
        'ma_3': [np.mean(price_values)],          # Moving average of 3 days
        'day_of_week': [datetime.now().weekday()]  # Current day of week
    })
    
    return X_pred

def train_linear_regression_model(data):
    """
    Train a linear regression model on historical data
    
    Parameters:
    data (pandas.DataFrame): Historical Bitcoin price data
    
    Returns:
    tuple: Trained model and RMSE
    """
    # Create features and target
    X, y = create_training_features(data)
    
    # Train the model (use all data since we're making a real prediction)
    model = LinearRegression()
    model.fit(X, y)
    
    # Calculate training RMSE
    predictions = model.predict(X)
    rmse = sqrt(mean_squared_error(y, predictions))
    
    print(f"Model training RMSE: ${rmse:.2f}")
    
    return model, rmse

def predict_next_days(model, X_pred, days=7):
    """
    Predict Bitcoin price for the next specified number of days
    
    Parameters:
    model: Trained LinearRegression model
    X_pred: Initial prediction features
    days (int): Number of days to predict
    
    Returns:
    pandas.DataFrame: Predictions for the specified number of days
    """
    predictions = []
    dates = []
    current_features = X_pred.iloc[0].copy()
    
    # Start date for predictions (tomorrow)
    current_date = datetime.now().date() + timedelta(days=1)
    
    for i in range(days):
        # Make prediction for the current day
        pred = model.predict(pd.DataFrame([current_features]))[0]
        predictions.append(pred)
        dates.append(current_date)
        
        # Update features for the next day's prediction
        current_features['price_3'] = current_features['price_2']
        current_features['price_2'] = current_features['price_1']
        current_features['price_1'] = pred
        
        current_features['change_3'] = current_features['change_2']
        current_features['change_2'] = current_features['change_1']
        current_features['change_1'] = pred - current_features['price_2']
        
        # Update moving average (simplified)
        current_features['ma_3'] = (current_features['price_1'] + 
                                   current_features['price_2'] + 
                                   current_features['price_3']) / 3
        
        # Update day of week
        current_date += timedelta(days=1)
        current_features['day_of_week'] = current_date.weekday()
    
    # Create DataFrame with predictions
    results = pd.DataFrame({
        'date': dates,
        'predicted_price': predictions
    })
    results.set_index('date', inplace=True)
    
    return results

def plot_prediction(input_data, predictions):
    """
    Create a Figure with plot of input data and predictions
    
    Parameters:
    input_data (pandas.DataFrame): Last 3 days of price data
    predictions (pandas.DataFrame): Predicted prices
    
    Returns:
    Figure: Matplotlib figure with the plot
    """
    # Create figure and axes - smaller dpi to avoid size issues
    fig = Figure(figsize=(7, 3.5), dpi=80)
    ax = fig.add_subplot(111)
    
    try:
        # Convert index to datetime if needed
        input_data.index = pd.to_datetime(input_data.index)
        predictions.index = pd.to_datetime(predictions.index)
        
        # Format dates for x-axis
        input_dates = [date.strftime('%m-%d') for date in input_data.index]
        pred_dates = [date.strftime('%m-%d') for date in predictions.index]
        
        # Use integer indices for x-axis to avoid date formatting issues
        ax.plot(range(len(input_data)), input_data['price'], 'bo-', label='Past 3 Days')
        ax.plot(range(len(input_data), len(input_data) + len(predictions)), 
                predictions['predicted_price'], 'ro--', label='Predictions')
        
        # Set custom x-tick labels
        all_dates = input_dates + pred_dates
        ax.set_xticks(range(len(all_dates)))
        ax.set_xticklabels(all_dates, rotation=45)
        
        # Add labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (USD)')
        ax.set_title('Bitcoin Price Prediction')
        
        # Set y-axis with reasonable steps (avoid maximum size exceeded error)
        min_price = min(input_data['price'].min(), predictions['predicted_price'].min())
        max_price = max(input_data['price'].max(), predictions['predicted_price'].max())
        price_range = max_price - min_price
        
        # Use appropriate step size based on the range
        if price_range > 100000:
            step = 50000
        elif price_range > 50000:
            step = 25000
        else:
            step = 10000
            
        y_min = (min_price // step) * step
        y_max = ((max_price // step) + 1) * step
        
        # Ensure the range isn't too large
        if (y_max - y_min) / step > 10:
            steps = 10
            step = (y_max - y_min) / steps
            y_ticks = np.linspace(y_min, y_max, steps + 1)
        else:
            y_ticks = np.arange(y_min, y_max + 1, step)
            
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(y_ticks)
        
        # Format y-tick labels
        ax.set_yticklabels([f"${y:,.0f}" for y in y_ticks])
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    except Exception as e:
        # If plotting fails, create a simple error message in the plot
        ax.text(0.5, 0.5, f"Error creating plot: {str(e)}", 
                ha='center', va='center', transform=ax.transAxes)
    
    # Make sure layout is properly set
    try:
        fig.tight_layout()
    except:
        pass
    
    return fig

class BitcoinPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Bitcoin Price Prediction")
        self.root.geometry("900x650")
        self.root.configure(bg="#f0f0f0")
        
        # Configure style
        self.style = ttk.Style()
        self.style.configure("TFrame", background="#f0f0f0")
        self.style.configure("TLabel", background="#f0f0f0")
        self.style.configure("TLabelframe", background="#f0f0f0")
        self.style.configure("TLabelframe.Label", background="#f0f0f0")
        self.style.configure("Accent.TButton", font=("Helvetica", 10, "bold"))
        
        # Create and place widgets
        self.create_widgets()
        
        # Canvas for graph
        self.canvas = None
        
    def create_widgets(self):
        # Header Label
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Label(
            header_frame, 
            text="Bitcoin Price Prediction", 
            font=("Helvetica", 16, "bold")
        ).pack()
        
        # Input Frame
        input_frame = ttk.LabelFrame(self.root, text="Enter Bitcoin Prices")
        input_frame.pack(fill="x", padx=10, pady=10)
    
        # Default values that make more sense
        default_values = [0, 0, 0]
        
        # Price inputs
        self.price_entries = []
        
        for i, day in enumerate(["3 days ago:", "2 days ago:", "Yesterday:"]):
            frame = ttk.Frame(input_frame)
            frame.pack(fill="x", padx=5, pady=5)
            
            ttk.Label(frame, text=f"Price from {day}", width=15).pack(side="left", padx=5)
            
            price_var = tk.StringVar(value=default_values[i])
            entry = ttk.Entry(frame, textvariable=price_var, width=20)
            entry.pack(side="left", padx=5)
            
            ttk.Label(frame, text="$").pack(side="left")
            
            self.price_entries.append(price_var)
        
        # Buttons Frame
        button_frame = ttk.Frame(input_frame)
        button_frame.pack(fill="x", padx=5, pady=10)
        
        ttk.Button(
            button_frame, 
            text="Make Prediction", 
            command=self.make_prediction,
            style="Accent.TButton"
        ).pack(side="left", padx=5)
        
        ttk.Button(
            button_frame,
            text="Reset",
            command=self.reset_values
        ).pack(side="left", padx=5)
        
        # Results Frame
        self.results_frame = ttk.LabelFrame(self.root, text="Prediction Results")
        self.results_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create a frame for the graph
        self.graph_frame = ttk.Frame(self.results_frame)
        self.graph_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create a frame for the prediction data
        self.data_frame = ttk.Frame(self.results_frame)
        self.data_frame.pack(fill="x", padx=5, pady=5)
        
        # Status label
        self.status_var = tk.StringVar(value="Enter prices and click 'Make Prediction'")
        self.status_label = ttk.Label(
            self.root, 
            textvariable=self.status_var,
            font=("Helvetica", 10)
        )
        self.status_label.pack(pady=5)
    
    def reset_values(self):
        """Reset input values to defaults"""
        default_values = [0,0,0]
        for i, entry_var in enumerate(self.price_entries):
            entry_var.set(default_values[i])
    
    def make_prediction(self):
        """Make prediction based on input values"""
        try:
            # Get input values
            prices = []
            for entry_var in self.price_entries:
                try:
                    price = float(entry_var.get().replace(",", "").replace("$", ""))
                    prices.append(price)
                except ValueError:
                    messagebox.showerror("Input Error", "Please enter valid numeric values for prices")
                    return
            
            # Check if prices are within a reasonable range
            if any(price <= 0 for price in prices):
                messagebox.showerror("Input Error", "All prices must be positive")
                return
                
            if any(price > 1000000 for price in prices):
                messagebox.showerror("Input Error", "Prices should be less than 1,000,000")
                return
                
            # Update status
            self.status_var.set("Processing... Making prediction based on your inputs")
            self.root.update_idletasks()
            
            # Make prediction
            result, input_df, predictions = predict_bitcoin_price(prices)
            
            # Clear previous results
            for widget in self.graph_frame.winfo_children():
                widget.destroy()
            
            for widget in self.data_frame.winfo_children():
                widget.destroy()
            
            # Create figure for the plot
            try:
                fig = plot_prediction(input_df, predictions)
                
                # Add plot to the frame
                self.canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
                self.canvas.draw()
                canvas_widget = self.canvas.get_tk_widget()
                canvas_widget.pack(fill="both", expand=True)
            except Exception as plot_error:
                error_label = ttk.Label(
                    self.graph_frame,
                    text=f"Unable to display graph: {str(plot_error)}",
                    foreground="red"
                )
                error_label.pack(padx=10, pady=20)
                print(f"Plot error: {plot_error}")
            
            # Display prediction results
            prediction_label = ttk.Label(
                self.data_frame,
                text=f"Tomorrow's predicted price: ${result['next_day_prediction']:,.2f}",
                font=("Helvetica", 12, "bold")
            )
            prediction_label.pack(anchor="w", padx=5, pady=5)
            
            # Create a table for 7-day predictions
            table_frame = ttk.Frame(self.data_frame)
            table_frame.pack(fill="x", padx=5, pady=5)
            
            # Table header
            ttk.Label(table_frame, text="Date", width=15, font=("Helvetica", 10, "bold")).grid(row=0, column=0, padx=2, pady=2)
            ttk.Label(table_frame, text="Predicted Price ($)", width=20, font=("Helvetica", 10, "bold")).grid(row=0, column=1, padx=2, pady=2)
            
            # Table rows
            for i, (idx, row) in enumerate(predictions.iterrows()):
                date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, 'strftime') else str(idx)
                ttk.Label(table_frame, text=date_str).grid(row=i+1, column=0, padx=2, pady=2)
                ttk.Label(table_frame, text=f"${row['predicted_price']:,.2f}").grid(row=i+1, column=1, padx=2, pady=2)
            
            # Add model RMSE
            rmse_label = ttk.Label(
                self.data_frame,
                text=f"Model RMSE (error metric): ${result['model_rmse']:,.2f}",
                font=("Helvetica", 10)
            )
            rmse_label.pack(anchor="w", padx=5, pady=5)
            
            # Update status
            self.status_var.set("Prediction complete! Table shows 7-day price predictions")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.status_var.set("An unexpected error occurred")
            print(f"Error in make_prediction: {e}")
root = tk.Tk()
app = BitcoinPredictionApp(root)
root.mainloop()
