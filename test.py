import requests
import joblib
import pandas as pd

model, model_sklearn_version = joblib.load('flood_prediction_decision_tree_model.pkl')
scaler = joblib.load('scaler.pkl')

PLACE_NAME = 'Vandalur'  

OPENCAGE_API_KEY = 'API_KEY' 
opencage_url = f'https://api.opencagedata.com/geocode/v1/json?q={PLACE_NAME}&key={OPENCAGE_API_KEY}'

opencage_response = requests.get(opencage_url)
opencage_data = opencage_response.json()

print(f"Status Code: {opencage_response.status_code}")
print(f"Response Content: {opencage_response.text}")

if opencage_response.status_code != 200 or not opencage_response.text:
    print(f"Failed to fetch data from OpenCage API. Status Code: {opencage_response.status_code}")
    exit()

if not opencage_data['results']:
    print(f"No results found for the location: {PLACE_NAME}")
    exit()

location = opencage_data['results'][0]['geometry']
latitude = float(location['lat'])
longitude = float(location['lng'])

OPENWEATHER_API_KEY = 'API_KEY' 
weather_url = f'http://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={OPENWEATHER_API_KEY}&units=metric'

weather_response = requests.get(weather_url)
weather_data = weather_response.json()

print(f"Weather API Response: {weather_data}")

if 'main' not in weather_data:
    print(f"No weather data available for the location: {PLACE_NAME}")
    exit()

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

if prediction[0] == 1:
    print("Flood predicted ")
else:
    print("No flood predicted ")
