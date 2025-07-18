import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import os
from datetime import datetime

# Path to the dataset
csv_path = 'data/soilmoisture_dataset_improved.csv'
model_path = 'ml/soil_moisture_model.pkl'

# 1. Load the dataset
try:
    df = pd.read_csv(csv_path)
    print('First 5 rows:')
    print(df.head())
    print('\nData summary:')
    print(df.info())
    print('\nMissing values per column:')
    print(df.isnull().sum())
    
    # Check data quality
    print('\nData Quality Check:')
    print(f'Soil moisture range: {df["soil_moisture_percent"].min():.2f}% - {df["soil_moisture_percent"].max():.2f}%')
    print(f'Temperature range: {df["temperature_celsius"].min():.2f}°C - {df["temperature_celsius"].max():.2f}°C')
    print(f'Humidity range: {df["humidity_percent"].min():.2f}% - {df["humidity_percent"].max():.2f}%')
    
except Exception as e:
    print(f'Error loading dataset: {e}')
    exit(1)

# 2. Data Cleaning and Preprocessing
print('\n=== Data Preprocessing ===')

# Drop duplicates and handle missing values
df = df.drop_duplicates().dropna()

# Remove outliers using IQR method for soil moisture
Q1 = df['soil_moisture_percent'].quantile(0.25)
Q3 = df['soil_moisture_percent'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f'Removing soil moisture outliers below {lower_bound:.2f}% and above {upper_bound:.2f}%')
df = df[(df['soil_moisture_percent'] >= lower_bound) & (df['soil_moisture_percent'] <= upper_bound)]

# 3. Feature Engineering
print('\n=== Feature Engineering ===')

# Extract time-based features from timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month

# Create interaction features
df['temp_humidity_interaction'] = df['temperature_celsius'] * df['humidity_percent']
df['temp_squared'] = df['temperature_celsius'] ** 2
df['humidity_squared'] = df['humidity_percent'] ** 2

# 4. Feature Selection
required_columns = ['location', 'temperature_celsius', 'humidity_percent', 'rainfall_24h', 
                   'days_since_irrigation', 'soil_type', 'hour', 'day_of_week', 
                   'month', 'temp_humidity_interaction', 'temp_squared', 'humidity_squared', 'soil_moisture_percent']
missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    print(f"Missing required columns: {missing_cols}")
    exit(1)

# Select features and target
feature_cols = ['location', 'temperature_celsius', 'humidity_percent', 'rainfall_24h', 
                'days_since_irrigation', 'soil_type', 'hour', 'day_of_week', 
                'month', 'temp_humidity_interaction', 'temp_squared', 'humidity_squared']
X = df[feature_cols]
y = df['soil_moisture_percent']

print(f'Features used: {feature_cols}')
print(f'Total samples after cleaning: {len(df)}')

# 5. Encode categorical features
label_encoders = {}
cat_cols = ['location', 'soil_type']
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# 6. Feature Scaling
scaler = RobustScaler()  # More robust to outliers
X_scaled = scaler.fit_transform(X)

# 7. Split Data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 8. Model Training with Multiple Algorithms
print('\n=== Model Training ===')

models = {
    'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=6, random_state=42),
    'Linear Regression': LinearRegression()
}

best_model = None
best_score = -float('inf')
best_model_name = ''

for name, model in models.items():
    print(f'\nTraining {name}...')
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
    cv_mean = cv_scores.mean()
    
    print(f'{name} Results:')
    print(f'  R² Score: {r2:.4f}')
    print(f'  RMSE: {rmse:.2f}')
    print(f'  MAE: {mae:.2f}')
    print(f'  Cross-validation R²: {cv_mean:.4f} (+/- {cv_scores.std() * 2:.4f})')
    
    # Select best model based on R² score
    if r2 > best_score:
        best_score = r2
        best_model = model
        best_model_name = name

# 9. Final Model Evaluation
print(f'\n=== Best Model: {best_model_name} ===')
y_pred_final = best_model.predict(X_test)
mse_final = mean_squared_error(y_test, y_pred_final)
rmse_final = np.sqrt(mse_final)
r2_final = r2_score(y_test, y_pred_final)
mae_final = mean_absolute_error(y_test, y_pred_final)

# Calculate accuracy percentage (R² score as percentage)
accuracy_percentage = r2_final * 100

print(f'\nFinal Model Evaluation:')
print(f'Mean Squared Error: {mse_final:.4f}')
print(f'R^2 Score: {r2_final:.4f}')
print(f'RMSE: {rmse_final:.4f}')
print(f'MAE: {mae_final:.4f}')
print(f'Accuracy: {accuracy_percentage:.1f}%')
print(f'Training samples: {len(X_train)}')
print(f'Test samples: {len(X_test)}')
print(f'Total samples: {len(df)}')

# 10. Save the model and preprocessors
os.makedirs(os.path.dirname(model_path), exist_ok=True)
model_bundle = {
    'model': best_model,
    'scaler': scaler,
    'label_encoders': label_encoders,
    'feature_cols': feature_cols,
    'training_date': datetime.now().isoformat(),
    'model_name': best_model_name,
    'metrics': {
        'mse': mse_final,
        'rmse': rmse_final,
        'r2': r2_final,
        'mae': mae_final,
        'accuracy_percentage': accuracy_percentage,
        'n_samples': len(df),
        'n_train': len(X_train),
        'n_test': len(X_test)
    }
}

with open(model_path, 'wb') as f:
    pickle.dump(model_bundle, f)
print(f'\nModel saved to {model_path}')

# 11. Feature Importance (for tree-based models)
if hasattr(best_model, 'feature_importances_'):
    print('\nFeature Importance:')
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(10).iterrows():
        print(f'  {row["feature"]}: {row["importance"]:.4f}') 