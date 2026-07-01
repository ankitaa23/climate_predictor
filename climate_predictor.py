"""
climate_predictor.py

Single-file demo project:
- generates synthetic historical data (optional)
- trains a RandomForestRegressor to predict temperature
- saves model to climate_model.joblib
- supports real-time prediction using OpenWeatherMap API
- shows Actual vs Predicted scatter plot

Usage examples:
    python climate_predictor.py --mode train --use_synthetic
    python climate_predictor.py --mode predict --country Germany --api_key YOUR_API_KEY
    python climate_predictor.py --mode train --datafile historical_weather.csv
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
import random
import math

import numpy as np # pyright: ignore[reportMissingImports]
import pandas as pd # type: ignore
from sklearn.preprocessing import LabelEncoder # pyright: ignore[reportMissingModuleSource]
from sklearn.ensemble import RandomForestRegressor # pyright: ignore[reportMissingModuleSource]
from sklearn.model_selection import train_test_split # pyright: ignore[reportMissingModuleSource]
from sklearn.metrics import mean_squared_error, r2_score # pyright: ignore[reportMissingModuleSource]
import joblib # pyright: ignore[reportMissingImports]
import requests # pyright: ignore[reportMissingModuleSource]
import matplotlib.pyplot as plt # pyright: ignore[reportMissingModuleSource]
import matplotlib # pyright: ignore[reportMissingModuleSource]
matplotlib.use('TkAgg')  # or 'Qt5Agg', depending on your system


MODEL_FILENAME = "climate_model.joblib"
ENCODER_FILENAME = "label_encoder.joblib"

# ----------------------------
# Synthetic data generator
# ----------------------------
def generate_synthetic_data(num_countries=8, years=5, seed=42):
    """
    Generate a synthetic historical climate dataset.
    Columns: Date, Country, Temperature, Humidity, Rainfall, WindSpeed
    """
    random.seed(seed)
    np.random.seed(seed)

    country_names = [
        "India", "Germany", "Canada", "Australia", "Brazil", "SouthAfrica", "Japan", "UK"
    ][:num_countries]

    start_date = datetime.now() - timedelta(days=365 * years)
    records = []
    for country in country_names:
        # define country base temperature & seasonality amplitude
        if country in ("India", "Brazil"):
            base_temp = 25
            amp = 8
        elif country in ("Germany", "UK", "Canada"):
            base_temp = 8
            amp = 12
        elif country == "Australia":
            base_temp = 18
            amp = 10
        elif country == "Japan":
            base_temp = 12
            amp = 10
        else:
            base_temp = 15
            amp = 8

        for day_offset in range(365 * years):
            dt = start_date + timedelta(days=day_offset)
            # seasonal pattern: sine wave by day-of-year
            doy = dt.timetuple().tm_yday
            seasonal = amp * math.sin(2 * math.pi * (doy / 365.0))
            trend = 0.02 * (day_offset / 365.0)  # small warming trend
            noise = np.random.normal(0, 1.8)
            temp = base_temp + seasonal + trend + noise

            # humidity inversely related slightly to temperature (simplified)
            humidity = min(max(40 + (10 - seasonal / 2) + np.random.normal(0, 6), 5), 100)

            # rainfall: random chance with monthly variation
            month = dt.month
            if country in ("India", "Brazil"):
                rain_factor = 0.5 if month in (6,7,8,9) else 0.15
            elif country in ("UK", "Germany"):
                rain_factor = 0.25
            elif country == "Australia":
                rain_factor = 0.30
            else:
                rain_factor = 0.2

            rainfall = np.random.exponential(5) if random.random() < rain_factor else 0.0
            wind_speed = max(0.5, np.random.normal(4.5, 1.8))

            records.append({
                "Date": dt.strftime("%Y-%m-%d"),
                "Country": country,
                "Temperature": round(temp, 2),
                "Humidity": round(humidity, 2),
                "Rainfall": round(rainfall, 2),
                "WindSpeed": round(wind_speed, 2)
            })

    df = pd.DataFrame.from_records(records)
    return df

# ----------------------------
# Data loading & preprocessing
# ----------------------------
def load_data(path=None, use_synthetic=False):
    if use_synthetic:
        print("Generating synthetic data (for demo)...")
        df = generate_synthetic_data()
    else:
        if not path or not os.path.exists(path):
            raise FileNotFoundError("No historical file found. Use --use_synthetic or pass --datafile <path>.")
        df = pd.read_csv(path)
    # basic checks and normalization of column names
    expected_cols = {"Date", "Country", "Temperature", "Humidity", "Rainfall", "WindSpeed"}
    # try to map if user used lowercase or slightly different names
    cols = set(df.columns)
    col_map = {}
    for field in expected_cols:
        if field in cols:
            col_map[field] = field
        else:
            # try lower-case match
            for c in cols:
                if c.lower() == field.lower():
                    col_map[field] = c
                    break
    missing = [f for f in expected_cols if f not in col_map]
    if missing:
        raise ValueError(f"Missing columns in data: {missing}. Expected columns: {list(expected_cols)}")
    df = df.rename(columns={col_map[k]: k for k in col_map})
    # ensure Date parsed
    df['Date'] = pd.to_datetime(df['Date'])
    # add month/day as features
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    return df

def prepare_features(df, label_encoder=None, fit_encoder=False):
    df_copy = df.copy()
    # handle categorical Country
    if label_encoder is None and fit_encoder:
        le = LabelEncoder()
        df_copy['Country_Code'] = le.fit_transform(df_copy['Country'])
        return df_copy, le
    elif label_encoder is not None:
        df_copy['Country_Code'] = label_encoder.transform(df_copy['Country'])
        return df_copy, label_encoder
    else:
        raise ValueError("Label encoder handling: pass encoder or set fit_encoder=True")

# ----------------------------
# Model training & saving
# ----------------------------
def train_and_save(df, target_col="Temperature", model_filename=MODEL_FILENAME, encoder_filename=ENCODER_FILENAME):
    print("Preparing features...")
    df_feat, le = prepare_features(df, fit_encoder=True)
    features = ['Country_Code', 'Month', 'Day', 'Humidity', 'Rainfall', 'WindSpeed']
    X = df_feat[features].values
    y = df_feat[target_col].values
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    print("Training RandomForestRegressor...")
    model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("Evaluating on test set...")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Test MSE: {mse:.4f}, R2: {r2:.4f}")
    print(f"Saving model to {model_filename} and encoder to {encoder_filename} ...")
    joblib.dump(model, model_filename)
    joblib.dump(le, encoder_filename)
    # return test split for plotting if needed
    return model, le, (X_test, y_test, y_pred, features)

# ----------------------------
# Real-time fetch
# ----------------------------
def get_weather_real_time(country_name, api_key):
    """
    Get current weather from OpenWeatherMap by city/country string.
    Note: openweathermap expects city name usually like "Berlin" or "London". 
    For country-level simplicity we query by country name; results may vary.
    """
    if not api_key:
        raise ValueError("API key required for real-time weather. Provide --api_key.")
    # try multiple query formats to increase chance of match
    query_options = [
        country_name,
        f"{country_name}",
    ]
    base = "http://api.openweathermap.org/data/2.5/weather"
    for q in query_options:
        params = {"q": q, "appid": api_key, "units": "metric"}
        try:
            r = requests.get(base, params=params, timeout=10)
            data = r.json()
        except Exception as e:
            data = {"cod": "error", "message": str(e)}
        if str(data.get("cod")) == "200":
            main = data.get("main", {})
            wind = data.get("wind", {})
            info = {
                "temperature": main.get("temp"),
                "humidity": main.get("humidity"),
                "pressure": main.get("pressure"),
                "wind_speed": wind.get("speed", 0.0)
            }
            return info
    # if failed:
    return None

# ----------------------------
# Prediction wrapper
# ----------------------------
def predict_for_country(country_name, api_key, model=None, le=None):
    # load model/encoder if not provided
    if model is None or le is None:
        if not os.path.exists(MODEL_FILENAME) or not os.path.exists(ENCODER_FILENAME):
            raise FileNotFoundError("Model or encoder not found. Train first using --mode train")
        model = joblib.load(MODEL_FILENAME)
        le = joblib.load(ENCODER_FILENAME)
    # fetch real-time data
    real = get_weather_real_time(country_name, api_key)
    if real is None:
        raise ValueError(f"Could not fetch real-time weather for '{country_name}'. Try a city name or check API key.")
    # build input vector
    today = datetime.now()
    try:
        country_code = le.transform([country_name])[0]
    except Exception:
        # if the country isn't in encoder, map to nearest known one (fallback: 0)
        print(f"Warning: '{country_name}' not found in training countries. Using fallback encoding 0.")
        country_code = 0
    X = np.array([[country_code, today.month, today.day, real["humidity"], 0.0, real["wind_speed"]]])
    # NOTE: we set rainfall=0 for current prediction; can be adapted if API returned rain volume.
    pred_temp = model.predict(X)[0]
    return {
        "predicted_temperature": float(round(pred_temp, 2)),
        "current_humidity": real["humidity"],
        "current_wind_speed": real["wind_speed"],
        "current_pressure": real.get("pressure")
    }

# ----------------------------
# Plotting
# ----------------------------
def plot_actual_vs_pred(y_test, y_pred):
    plt.figure(figsize=(7,7))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle="--")
    plt.xlabel("Actual Temperature")
    plt.ylabel("Predicted Temperature")
    plt.title("Actual vs Predicted Temperature")
    plt.grid(True)
    plt.show(block=True)


# ----------------------------
# CLI main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Climate prediction demo: train or predict (real-time).")
    parser.add_argument("--mode", choices=["train", "predict", "both"], default="both",
                        help="train: train model; predict: run real-time prediction; both: train then predict")
    parser.add_argument("--datafile", type=str, default=None, help="Path to historical CSV (optional)")
    parser.add_argument("--use_synthetic", action="store_true", help="Generate synthetic historical data and train on it")
    parser.add_argument("--country", type=str, default="Germany", help="Country (or city) name for real-time prediction")
    parser.add_argument("--api_key", type=str, default=None, help="OpenWeatherMap API key for real-time data")
    parser.add_argument("--show_plot", action="store_true", help="Show Actual vs Predicted scatter plot after training")
    args = parser.parse_args()

    if args.mode in ("train", "both"):
        # data loading
        try:
            df = load_data(path=args.datafile, use_synthetic=args.use_synthetic)
        except Exception as e:
            print("Error loading data:", e)
            sys.exit(1)
        print("Dataset sample:")
        print(df.head().to_string(index=False))
        model, le, test_info = train_and_save(df)
        X_test, y_test, y_pred, features = test_info
        if args.show_plot:
            plot_actual_vs_pred(y_test, y_pred)

    if args.mode in ("predict", "both"):
        if args.api_key is None:
            print("Real-time prediction requires --api_key. Provide your OpenWeatherMap API key.")
            # if mode was both and training just happened, still allow predict if API key provided
            if args.mode == "predict":
                sys.exit(1)
        try:
            mdl = None
            encoder = None
            if os.path.exists(MODEL_FILENAME) and os.path.exists(ENCODER_FILENAME):
                mdl = joblib.load(MODEL_FILENAME)
                encoder = joblib.load(ENCODER_FILENAME)
            elif args.mode == "predict":
                print("Model not found. Train first using --mode train")
                sys.exit(1)
            result = predict_for_country(args.country, args.api_key, model=mdl, le=encoder)
            print(f"Real-time prediction for '{args.country}':")
            print(f" Predicted Temperature: {result['predicted_temperature']} °C")
            print(f" Current Humidity: {result['current_humidity']}")
            print(f" Current Wind Speed: {result['current_wind_speed']}")
            print(f" Current Pressure: {result.get('current_pressure')}")
        except Exception as e:
            print("Prediction error:", e)
            sys.exit(1)

if __name__ == "__main__":
    main()


    