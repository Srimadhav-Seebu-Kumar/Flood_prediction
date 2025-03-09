import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import sklearn

def main():
    # Load the dataset
    data = pd.read_csv('FloodPrediction.csv')

    # Preprocessing
    data['Flood?'] = data['Flood?'].apply(lambda x: 1 if x == 1 else 0)
    data = data.drop(['Sl', 'Station_Names', 'Station_Number', 'Period'], axis=1)
    data = data.dropna()

    # Feature and target separation
    X = data.drop('Flood?', axis=1)
    y = data['Flood?']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Save the scaler
    joblib.dump(scaler, 'scaler.pkl')

    # Model training using DecisionTreeClassifier
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Prediction and evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    print('Classification Report:')
    print(classification_report(y_test, y_pred))
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))

    # Save the model with sklearn version
    joblib.dump((model, sklearn.__version__), 'flood_prediction_decision_tree_model.pkl')
    print("Model and scaler saved.")

if __name__ == "__main__":
    main()
