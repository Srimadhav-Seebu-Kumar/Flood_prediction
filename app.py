from flask import Flask, request, jsonify
import requests
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

model, model_sklearn_version = joblib.load('flood_prediction_decision_tree_model.pkl')
scaler = joblib.load('scaler.pkl')

OPENCAGE_API_KEY = 'API_KEY'
OPENWEATHER_API_KEY = 'API_KEY'

@app.route('/predict_flood', methods=['GET'])
def predict_flood():
    PLACE_NAME = request.args.get('place_name')
    
    if not PLACE_NAME:
        return jsonify({"error": "Please provide a valid PLACE_NAME"}), 400

    opencage_url = f'https://api.opencagedata.com/geocode/v1/json?q={PLACE_NAME}&key={OPENCAGE_API_KEY}'
    opencage_response = requests.get(opencage_url)
    
    if opencage_response.status_code != 200 or not opencage_response.text:
        return jsonify({"error": "Failed to fetch data from OpenCage API"}), 500

    opencage_data = opencage_response.json()

    if not opencage_data['results']:
        return jsonify({"error": f"No results found for the location: {PLACE_NAME}"}), 404

    location = opencage_data['results'][0]['geometry']
    latitude = float(location['lat'])
    longitude = float(location['lng'])

    weather_url = f'http://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={OPENWEATHER_API_KEY}&units=metric'
    weather_response = requests.get(weather_url)
    weather_data = weather_response.json()

    if 'main' not in weather_data:
        return jsonify({"error": f"No weather data available for the location: {PLACE_NAME}"}), 404

    temp_max = weather_data['main']['temp_max']
    temp_min = weather_data['main']['temp_min']
    humidity = weather_data['main']['humidity']
    wind_speed = weather_data['wind']['speed']
    cloud_coverage = weather_data['clouds']['all']
    rainfall = weather_data.get('rain', {}).get('1h', 0)

    sample_data = {
        'Year': [2024],  
        'Month': [8],  
        'Max_Temp': [temp_max],
        'Min_Temp': [temp_min],
        'Rainfall': [rainfall],
        'Relative_Humidity': [humidity],
        'Wind_Speed': [wind_speed],
        'Cloud_Coverage': [cloud_coverage],
        'Bright_Sunshine': [0], 
        'X_COR': [latitude],  
        'Y_COR': [longitude], 
        'LATITUDE': [latitude],
        'LONGITUDE': [longitude],
        'ALT': [0]  
    }

    sample_df = pd.DataFrame(sample_data)
    sample_df_scaled = scaler.transform(sample_df)

    prediction = model.predict(sample_df_scaled)
    prediction_proba = model.predict_proba(sample_df_scaled)

    prediction_percentage = round(prediction_proba[0][prediction[0]] * 100, 2)

    result = "Flood predicted" if prediction[0] == 1 else "No flood predicted"
    response = {
        "place_name": PLACE_NAME,
        "prediction": result,
        "prediction_percentage": prediction_percentage,
        "parameters": {
            "Max_Temp": temp_max,
            "Min_Temp": temp_min,
            "Rainfall": rainfall,
            "Relative_Humidity": humidity,
            "Wind_Speed": wind_speed,
            "Cloud_Coverage": cloud_coverage,
            "Latitude": latitude,
            "Longitude": longitude
        }
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
