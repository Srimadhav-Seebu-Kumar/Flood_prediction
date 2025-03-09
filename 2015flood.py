import joblib
import sklearn
import pandas as pd

model, model_sklearn_version = joblib.load('flood_prediction_decision_tree_model.pkl')

if model_sklearn_version != sklearn.__version__:
    print(f"Warning: Model was trained with scikit-learn {model_sklearn_version} but you are using {sklearn.__version__}. Consider using the same version.")

sample_data = {
    'Year': [2015],
    'Month': [11],
    'Max_Temp': [32],
    'Min_Temp': [23],
    'Rainfall': [163],
    'Relative_Humidity': [89],
    'Wind_Speed': [10.8],
    'Cloud_Coverage': [7],
    'Bright_Sunshine': [5.7],
    'X_COR': [8928079.589],
    'Y_COR': [1450221.806],
    'LATITUDE': [13.09],
    'LONGITUDE': [80.19],
    'ALT': [6]
}

sample_df = pd.DataFrame(sample_data)

scaler = joblib.load('scaler.pkl')  
sample_df_scaled = scaler.transform(sample_df)

prediction = model.predict(sample_df_scaled)

if prediction[0] == 1:
    print("Flood predicted")
else:
    print("No flood predicted")
