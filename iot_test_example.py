#!/usr/bin/env python3
"""
IoT Device Test Script
This script demonstrates how to use the IoT API endpoints for soil moisture monitoring.
"""

import requests
import json
import time
import random
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:8000/soil/api"
DEVICE_ID = "test_sensor_001"
LOCATION_ID = 1  # Make sure this location exists in your database

def test_api_status():
    """Test the API status endpoint"""
    print("Testing API status...")
    try:
        response = requests.get(f"{BASE_URL}/iot-status/")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Status: {data['status']}")
            print(f"Message: {data['message']}")
            return True
        else:
            print(f"❌ API Status failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

def submit_soil_reading(moisture, temperature=None, humidity=None, crop_type_id=None):
    """Submit a soil moisture reading to the IoT API"""
    url = f"{BASE_URL}/iot-reading/"
    
    data = {
        "device_id": DEVICE_ID,
        "location_id": LOCATION_ID,
        "moisture": moisture,
        "temperature": temperature,
        "humidity": humidity,
        "crop_type_id": crop_type_id,
        "notes": f"Test reading from {DEVICE_ID} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    }
    
    # Remove None values
    data = {k: v for k, v in data.items() if v is not None}
    
    try:
        print(f"Submitting reading: moisture={moisture}%, temperature={temperature}°C, humidity={humidity}%")
        response = requests.post(url, json=data)
        
        if response.status_code == 201:
            result = response.json()
            print(f"✅ Reading submitted successfully!")
            print(f"Record ID: {result['record_id']}")
            print(f"Recorded at: {result['recorded_at']}")
            return True
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

def get_latest_reading():
    """Get the latest soil moisture reading"""
    print("Getting latest reading...")
    try:
        response = requests.get(f"{BASE_URL}/latest-soil-moisture/")
        if response.status_code == 200:
            data = response.json()
            if data['moisture'] is not None:
                print(f"✅ Latest reading: {data['moisture']}% at {data['location']}")
                print(f"Temperature: {data['temperature']}°C, Humidity: {data['humidity']}%")
                return True
            else:
                print("No readings available yet.")
                return False
        else:
            print(f"❌ Error getting latest reading: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

def simulate_iot_device():
    """Simulate an IoT device sending periodic readings"""
    print("\n🚀 Starting IoT device simulation...")
    print(f"Device ID: {DEVICE_ID}")
    print(f"Location ID: {LOCATION_ID}")
    print("Press Ctrl+C to stop\n")
    
    reading_count = 0
    
    try:
        while True:
            # Simulate sensor readings with realistic values
            moisture = random.uniform(20, 80)  # 20-80% moisture
            temperature = random.uniform(15, 35)  # 15-35°C
            humidity = random.uniform(40, 90)  # 40-90% humidity
            
            print(f"\n--- Reading #{reading_count + 1} ---")
            success = submit_soil_reading(moisture, temperature, humidity)
            
            if success:
                reading_count += 1
                print(f"Total readings submitted: {reading_count}")
            
            # Wait 10 seconds before next reading
            print("Waiting 10 seconds before next reading...")
            time.sleep(10)
            
    except KeyboardInterrupt:
        print(f"\n\n📊 Simulation stopped. Total readings submitted: {reading_count}")

def main():
    """Main function to run the IoT test"""
    print("🌱 Soil Moisture IoT API Test")
    print("=" * 40)
    
    # Test 1: Check API status
    if not test_api_status():
        print("❌ API is not available. Make sure the Django server is running.")
        return
    
    # Test 2: Submit a single reading
    print("\n" + "=" * 40)
    print("Testing single reading submission...")
    success = submit_soil_reading(
        moisture=45.5,
        temperature=25.3,
        humidity=60.2
    )
    
    if not success:
        print("❌ Failed to submit reading. Check your configuration.")
        return
    
    # Test 3: Get latest reading
    print("\n" + "=" * 40)
    get_latest_reading()
    
    # Test 4: Start simulation (optional)
    print("\n" + "=" * 40)
    print("IoT Device Simulation")
    print("This will continuously submit readings every 10 seconds.")
    response = input("Start simulation? (y/n): ").lower().strip()
    
    if response == 'y':
        simulate_iot_device()
    else:
        print("Simulation skipped. Test completed.")

if __name__ == "__main__":
    main() 