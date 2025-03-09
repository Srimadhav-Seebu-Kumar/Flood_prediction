import joblib
import sklearn
import pandas as pd

model, model_sklearn_version = joblib.load('flood_prediction_decision_tree_model.pkl')


sample_data = {
    'Year': [1950],
    'Month': [6],
    'Max_Temp': [34.4],
    'Min_Temp': [25.7],
    'Rainfall': [512],
    'Relative_Humidity': [80],
    'Wind_Speed': [1.63],
    'Cloud_Coverage': [5.6],
    'Bright_Sunshine': [4.07],
    'X_COR': [536809.8],
    'Y_COR': [510151.9],
    'LATITUDE': [22.7],
    'LONGITUDE': [90.36],
    'ALT': [4]
}

sample_df = pd.DataFrame(sample_data)

scaler = joblib.load('scaler.pkl') 
sample_df_scaled = scaler.transform(sample_df)

prediction = model.predict(sample_df_scaled)

if prediction[0] == 1:
    print("Flood predicted")
else:
    print("No flood predicted")
