import pandas as pd
import numpy as np
import joblib
import sys

#load model and scaler
model = joblib.load('random_forest_classifier.pkl')
scaler = joblib.load('rf_scaler.pkl')

#feature column used during training(copied from data.py output)
FEATURE_COLUMNS = [
    'bedrooms','bathrooms','sqft_living','floors','age',
    'location_Downtown','location_Suburb'
]

#Helper to get all possible location columns from training
def get_location_columns():
    return['Downtown','Suburb']

location_columns = get_location_columns()

#prepare input features
def prepare_features(input_dict):
    features = {col: 0 for col in FEATURE_COLUMNS}
    for key, value in input_dict.items():
        if key in features:
            features[key] = value
        elif key == 'location':
            loc_col = f"location_{value}"
            if loc_col in features:
                features[loc_col] = 1
    return pd.DataFrame([features])


def manual_input():
    print("Enter the following features:")
    input_dict = {}
    for col in FEATURE_COLUMNS:
        if col.startswith('location_'):
            continue
        val = input(f"{col}: ")
        try:
            input_dict[col] = float(val)
        except ValueError:
            input_dict[col] = val
    print(f"Available locations: {location_columns}")
    location = input("location: ")
    input_dict['location'] = location
    X = prepare_features(input_dict)
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)
    print(f"Predicted price category: {pred[0]}")

def batch_predict(csv_path):
    df = pd.read_csv(csv_path)
    df = pd.get_dummies(df, columns=['location'], drop_first=True)
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0
    df = df[FEATURE_COLUMNS]
    X_scaled = scaler.transform(df)
    preds = model.predict(X_scaled)
    df['predicted_price_category'] = preds
    print(df[['predicted_price_category']])
    df.to_csv('predictions.csv', index=False)
    print("Predictions saved to predictions.csv")

if __name__ == "__main__":
    if not FEATURE_COLUMNS:
        print("Please fill FEATURE_COLUMNS with the list printed in data.py output.")
        sys.exit(1)
    mode = input("Choose mode: (1) Manual input, (2) Batch CSV prediction. Enter 1 or 2: ")
    if mode == '1':
        manual_input()
    elif mode == '2':
        csv_path = input("Enter path to CSV file: ")
        batch_predict(csv_path)
    else:
        print("Invalid option.")