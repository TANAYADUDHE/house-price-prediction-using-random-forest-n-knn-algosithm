import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import numpy as np
import pandas as pd

# Load KNN model and scaler
model = joblib.load('knn_classifier.pkl')
scaler = joblib.load('knnscaler.pkl')

FEATURE_COLUMNS = [
    'bedrooms', 'bathrooms', 'sqft_living', 'floors', 'age',
    'location_Downtown', 'location_Suburb'
]

LOCATIONS = ['Countryside', 'Downtown', 'Suburb']

# Prepare input features for prediction
def prepare_features_tk(input_dict):
    features = {col: 0 for col in FEATURE_COLUMNS}
    for key, value in input_dict.items():
        if key in features:
            features[key] = value
        elif key == 'location':
            if value == 'Downtown':
                features['location_Downtown'] = 1
            elif value == 'Suburb':
                features['location_Suburb'] = 1
            # Countryside: both dummies remain 0
    return pd.DataFrame([features])


def predict():
    try:
        input_dict = {
            'bedrooms': float(entry_bedrooms.get()),
            'bathrooms': float(entry_bathrooms.get()),
            'sqft_living': float(entry_sqft_living.get()),
            'floors': float(entry_floors.get()),
            'age': float(entry_age.get()),
            'location': location_var.get()
        }
    except ValueError:
        messagebox.showerror('Input Error', 'Please enter valid numeric values for all features.')
        return
    X = prepare_features_tk(input_dict)
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)
    result_var.set(f'Predicted price category: {pred[0]}')

# Tkinter GUI setup
root = tk.Tk()
root.title('KNN House Price Category Predictor')
root.geometry('350x350')

frame = ttk.Frame(root, padding=20)
frame.pack(fill='both', expand=True)

# Feature entries
labels = ['Bedrooms', 'Bathrooms', 'Sqft Living', 'Floors', 'Age']
entries = []

entry_bedrooms = ttk.Entry(frame)
entry_bathrooms = ttk.Entry(frame)
entry_sqft_living = ttk.Entry(frame)
entry_floors = ttk.Entry(frame)
entry_age = ttk.Entry(frame)

for i, (label, entry) in enumerate(zip(labels, [entry_bedrooms, entry_bathrooms, entry_sqft_living, entry_floors, entry_age])):
    ttk.Label(frame, text=label+':').grid(row=i, column=0, sticky='e', pady=5)
    entry.grid(row=i, column=1, pady=5)

# Location dropdown
ttk.Label(frame, text='Location:').grid(row=5, column=0, sticky='e', pady=5)
location_var = tk.StringVar(value=LOCATIONS[0])
location_menu = ttk.OptionMenu(frame, location_var, LOCATIONS[0], *LOCATIONS)
location_menu.grid(row=5, column=1, pady=5)

# Predict button
predict_btn = ttk.Button(frame, text='Predict', command=predict)
predict_btn.grid(row=6, column=0, columnspan=2, pady=15)

# Result label
result_var = tk.StringVar()
result_label = ttk.Label(frame, textvariable=result_var, font=('Arial', 12, 'bold'))
result_label.grid(row=7, column=0, columnspan=2, pady=10)

root.mainloop()