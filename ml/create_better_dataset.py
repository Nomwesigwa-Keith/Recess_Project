import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def create_realistic_soil_dataset(n_samples=1000):
    """
    Create a realistic soil moisture dataset with proper feature-target relationships
    """
    np.random.seed(42)
    random.seed(42)
    
    # Generate realistic data
    locations = ['Farm A', 'Farm B', 'Farm C', 'Farm D', 'Farm E']
    crop_types = ['Corn', 'Wheat', 'Soybeans', 'Cotton', 'Rice']
    
    data = []
    base_date = datetime(2024, 1, 1)
    
    for i in range(n_samples):
        # Generate realistic features
        location = random.choice(locations)
        crop_type = random.choice(crop_types)
        
        # Temperature (realistic range: 10-35°C)
        temperature = np.random.normal(25, 8)
        temperature = max(10, min(35, temperature))
        
        # Humidity (realistic range: 30-90%)
        humidity = np.random.normal(60, 15)
        humidity = max(30, min(90, humidity))
        
        # Rainfall in last 24 hours (mm)
        rainfall_24h = np.random.exponential(5)  # Most days have little rain
        rainfall_24h = min(rainfall_24h, 50)  # Cap at 50mm
        
        # Days since last irrigation
        days_since_irrigation = np.random.exponential(3)
        days_since_irrigation = min(days_since_irrigation, 14)  # Cap at 14 days
        
        # Soil type factor (affects moisture retention)
        soil_types = {'Sandy': 0.7, 'Loamy': 1.0, 'Clay': 1.3}
        soil_type = random.choice(list(soil_types.keys()))
        soil_factor = soil_types[soil_type]
        
        # Calculate realistic soil moisture based on features
        # Base moisture from rainfall and irrigation
        base_moisture = (rainfall_24h * 0.3) + (max(0, 20 - days_since_irrigation * 2))
        
        # Temperature effect (higher temp = lower moisture)
        temp_effect = -0.5 * (temperature - 20) / 15
        
        # Humidity effect (higher humidity = higher moisture)
        humidity_effect = 0.3 * (humidity - 50) / 40
        
        # Soil type effect
        soil_effect = (soil_factor - 1.0) * 10
        
        # Calculate final soil moisture
        soil_moisture = base_moisture + temp_effect + humidity_effect + soil_effect
        
        # Add some realistic noise
        soil_moisture += np.random.normal(0, 3)
        
        # Ensure realistic bounds (5-60%)
        soil_moisture = max(5, min(60, soil_moisture))
        
        # Generate timestamp
        timestamp = base_date + timedelta(days=i//10, hours=i%24)
        
        # Determine status based on moisture
        if soil_moisture < 15:
            status = 'Critical Low'
        elif soil_moisture < 25:
            status = 'Dry'
        elif soil_moisture < 40:
            status = 'Normal'
        elif soil_moisture < 50:
            status = 'Wet'
        else:
            status = 'Critical High'
        
        # Determine irrigation action
        if soil_moisture < 20:
            irrigation_action = 'Irrigate'
        elif soil_moisture > 45:
            irrigation_action = 'Reduce Irrigation'
        else:
            irrigation_action = 'None'
        
        # Generate sensor data
        sensor_id = f'SENSOR_{random.randint(1, 20)}'
        battery_voltage = round(random.uniform(3.0, 4.2), 2)
        
        data.append({
            'record_id': i + 1,
            'sensor_id': sensor_id,
            'location': location,
            'crop_type': crop_type,
            'soil_moisture_percent': round(soil_moisture, 2),
            'temperature_celsius': round(temperature, 1),
            'humidity_percent': round(humidity, 1),
            'rainfall_24h': round(rainfall_24h, 1),
            'days_since_irrigation': round(days_since_irrigation, 1),
            'soil_type': soil_type,
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'status': status,
            'battery_voltage': battery_voltage,
            'irrigation_action': irrigation_action
        })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    print("Creating realistic soil moisture dataset...")
    
    # Create the dataset
    df = create_realistic_soil_dataset(1000)
    
    # Save to file
    output_path = 'data/soil_moisture_dataset_improved.csv'
    df.to_csv(output_path, index=False)
    
    print(f"Dataset created with {len(df)} samples")
    print(f"Saved to: {output_path}")
    
    # Show correlations
    print("\nFeature correlations with soil moisture:")
    correlation_cols = ['soil_moisture_percent', 'temperature_celsius', 'humidity_percent', 
                       'rainfall_24h', 'days_since_irrigation']
    correlations = df[correlation_cols].corr()['soil_moisture_percent']
    print(correlations)
    
    # Show statistics
    print("\nSoil moisture statistics:")
    print(df['soil_moisture_percent'].describe())
    
    print("\nSample data:")
    print(df.head()) 