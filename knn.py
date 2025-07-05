import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib

# Load the dataset
df = pd.read_csv('house_prices.csv')

# Create price category
def price_category(price):
    if price < 200000:
        return 'Cheap'
    elif price < 400000:
        return 'Medium'
    else:
        return 'Expensive'

df['price_category'] = df['price'].apply(price_category)

# Print class distribution
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

# Initialize and train the KNN Classifier
best_k = 5  # You can change this value to try different k
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)

# Save the trained KNN model and scaler
joblib.dump(knn, 'knn_classifier.pkl')
joblib.dump(scaler, 'knnscaler.pkl')
print(f"KNN model saved as knn_classifier.pkl (k={best_k})")

# Predict on the test set
y_pred = knn.predict(X_test)

# Confusion matrix and classification report
cm = confusion_matrix(y_test, y_pred, labels=['Cheap', 'Medium', 'Expensive'])
cr = classification_report(y_test, y_pred)

print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(cr)