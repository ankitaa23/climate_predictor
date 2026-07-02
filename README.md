# Climate Predictor

This repository contains a Python demo for training a climate prediction model and performing real-time weather-based temperature predictions.

## Project files

- `climate_predictor.py` - Main script to train a `RandomForestRegressor` model and perform prediction using OpenWeatherMap data.
- `historical_weather.csv.csv` - Optional historical weather dataset for training.
- `climate_model.joblib` - Saved trained model artifact.
- `label_encoder.joblib` - Saved label encoder artifact.

## Features

- Synthetic dataset generation for demo training
- Data preprocessing with categorical encoding and date-based features
- Model training with test evaluation metrics
- Real-time prediction using OpenWeatherMap API
- Optional scatter plot visualization of actual vs predicted values

## Usage

Train using synthetic data:

```bash
d:/AIML_PROJECTS/.venv/Scripts/python.exe d:/AIML_PROJECTS/climate_predictor.py --mode train --use_synthetic
```

Train using an existing CSV dataset:

```bash
d:/AIML_PROJECTS/.venv/Scripts/python.exe d:/AIML_PROJECTS/climate_predictor.py --mode train --datafile historical_weather.csv.csv
```

Predict in real time:

```bash
d:/AIML_PROJECTS/.venv/Scripts/python.exe d:/AIML_PROJECTS/climate_predictor.py --mode predict --country Germany --api_key YOUR_API_KEY
```

Train and then predict in one command:

```bash
d:/AIML_PROJECTS/.venv/Scripts/python.exe d:/AIML_PROJECTS/climate_predictor.py --mode both --use_synthetic --api_key YOUR_API_KEY
```

## Notes

- Replace `YOUR_API_KEY` with a valid OpenWeatherMap API key.
- The repository is linked to https://github.com/ankitaa23/climate_predictor.
