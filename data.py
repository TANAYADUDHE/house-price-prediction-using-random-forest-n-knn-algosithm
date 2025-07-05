import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib

#load the dataset
df = pd.read_csv('house_prices.csv')

#create price category
def price_category(price):
    if price < 200000:
        return 'Cheap'
    elif price < 400000:
        return "Medium"
    else:
        return 'Expensive'
    
df['price_category'] = df['price'].apply(price_category)
#porice class distribution
print('Class distribution:')
print(df['price_category'].value_counts())

# Convert categorical columns to numeric (one-hot encoding)
df = pd.get_dummies(df, columns=['location'], drop_first=True)

# Features and target
X = df.drop(['price', 'price_category'], axis=1)
y = df['price_category']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Stratified split to ensure all classes are represented
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize and train the Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Save the trained Random Forest model and scaler
joblib.dump(clf, 'random_forest_classifier.pkl')
joblib.dump(scaler, 'rf_scaler.pkl')
print("Random Forest model saved as random_forest_classifier.pkl")

# Predict on the test set
y_pred = clf.predict(X_test)

# Confusion matrix and classification report
cm = confusion_matrix(y_test, y_pred, labels=['Cheap', 'Medium', 'Expensive'])
cr = classification_report(y_test, y_pred)

print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(cr)

print("Feature columns used for training:")
print(X.columns.tolist())

print("FEATURE ORDER FOR PREDICTION:")
print(X.columns.tolist())