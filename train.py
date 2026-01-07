import pandas as pd
import joblib
import random
import os
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

MODEL_FILE = "parking_model.pkl"

def train_model():
    print("🤖 Generating training data...")
    data = []
    start_date = datetime.now() - timedelta(days=365)

    for _ in range(2000):
        random_days = random.randint(0, 365)
        current_date = start_date + timedelta(days=random_days)
        
        day_of_week = current_date.weekday()
        hour = random.randint(6, 23)
        is_weekend = 1 if day_of_week >= 5 else 0
        
        # Ground Truth Logic
        occupancy = random.randint(10, 30)
        if 17 <= hour <= 20: occupancy += random.randint(30, 50)
        if is_weekend: occupancy += random.randint(10, 20)
        occupancy = min(100, occupancy)

        data.append([day_of_week, hour, is_weekend, occupancy])

    df = pd.DataFrame(data, columns=['day', 'hour', 'weekend', 'occupancy'])
    
    X = df[['day', 'hour', 'weekend']]
    y = df['occupancy']

    print("🧠 Training Model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    joblib.dump(model, MODEL_FILE)
    print(f"✅ Model saved to {MODEL_FILE}")

if __name__ == "__main__":
    train_model()