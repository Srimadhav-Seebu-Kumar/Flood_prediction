# Flood Prediction API

## Overview
The Flood Prediction API is a Flask-based web service that predicts the likelihood of flooding based on real-time weather data and a pre-trained machine learning model.

## Features
- Fetches real-time geographical coordinates using OpenCage API
- Retrieves weather parameters from OpenWeather API
- Uses a trained Decision Tree model for flood prediction
- Returns flood prediction results with probability percentage

## Requirements
- Python 3.7+
- Flask
- Requests
- Joblib
- Pandas
- NumPy
- Scikit-learn

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/flood-prediction-api.git
   cd flood-prediction-api
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Add your API keys in the script:
   - Replace `API_KEY` in `OPENCAGE_API_KEY` and `OPENWEATHER_API_KEY` with valid keys.
4. Ensure `flood_prediction_decision_tree_model.pkl` and `scaler.pkl` are present in the project directory.

## Running the API
Run the Flask application:
```bash
python app.py
```

## API Endpoint
### `GET /predict_flood`
#### Description
Predicts flood probability for a given location.

#### Query Parameters
- `place_name` (string, required): Name of the location (e.g., `Chennai`).

#### Example Request
```bash
curl "http://127.0.0.1:5000/predict_flood?place_name=Chennai"
```

#### Example Response
```json
{
  "place_name": "Chennai",
  "prediction": "Flood predicted",
  "prediction_percentage": 78.5,
  "parameters": {
    "Max_Temp": 35.2,
    "Min_Temp": 27.4,
    "Rainfall": 12.5,
    "Relative_Humidity": 85,
    "Wind_Speed": 4.2,
    "Cloud_Coverage": 75,
    "Latitude": 13.0827,
    "Longitude": 80.2707
  }
}
```

## License
This project is licensed under the MIT License. Modify and distribute as needed.

## Authors
Developed by Srimadhav Seebu Kumar and Contributors.

