from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from datetime import datetime
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# --- LOAD MODELS ON STARTUP ---
PARKING_MODEL_PATH = "parking_model.pkl"
parking_model = None

@app.on_event("startup")
def load_models():
    global parking_model
    if os.path.exists(PARKING_MODEL_PATH):
        parking_model = joblib.load(PARKING_MODEL_PATH)
        print("✅ Parking Model Loaded")
    else:
        print("⚠️ Warning: Parking Model not found. Run train.py first.")

# --- DATA STRUCTURES ---
class ParkingRequest(BaseModel):
    datetime: str

class RecommendationRequest(BaseModel):
    # We receive the TEXT content, not IDs, because this server 
    # doesn't have access to the Django Database.
    target_content: str  # The description of the event user booked
    candidates: list     # List of all other events [{'id': 1, 'content': '...'}, ...]

# --- ROUTES ---

@app.get("/")
def home():
    return {"status": "ML Service Online"}

@app.post("/predict/parking/")
def predict_parking(req: ParkingRequest):
    if not parking_model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        dt = datetime.fromisoformat(req.datetime.replace("Z", ""))
        day = dt.weekday()
        hour = dt.hour
        is_weekend = 1 if day >= 5 else 0
        
        # Predict
        prediction = parking_model.predict([[day, hour, is_weekend]])[0]
        
        occupancy = round(prediction)
        level = "Low"
        if occupancy > 85: level = "Critical"
        elif occupancy > 60: level = "High"
        elif occupancy > 30: level = "Moderate"

        return {
            "predicted_occupancy": occupancy,
            "level": level,
            "hour": hour,
            "day": dt.strftime("%A"),
            "is_weekend": bool(is_weekend)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/recommend/")
def get_recommendations(req: RecommendationRequest):
    """
    Generic Recommendation Engine (Works for Events AND Facilities).
    """
    if not req.candidates:
        return []

    # Prepare Data
    contents = [req.target_content] + [item['content'] for item in req.candidates]
    
    # Vectorize
    tfidf = TfidfVectorizer(stop_words='english')
    matrix = tfidf.fit_transform(contents)
    
    # Calculate Similarity with the Target (Index 0)
    cosine_sim = cosine_similarity(matrix[0:1], matrix[1:])
    
    # Get Scores
    scores = list(enumerate(cosine_sim[0]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    
    # Return Top 3 IDs
    top_indices = [i[0] for i in scores[:3]]
    recommended_ids = [req.candidates[i]['id'] for i in top_indices]
    
    return recommended_ids