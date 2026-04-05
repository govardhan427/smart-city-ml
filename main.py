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

# UPGRADED: Added target_location
class RecommendationRequest(BaseModel):
    target_content: str  
    target_location: str 
    candidates: list     # [{'id': 1, 'content': '...', 'location': '...'}, ...]

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

# UPGRADED: Hybrid Scoring System
@app.post("/recommend/")
def get_recommendations(req: RecommendationRequest):
    """
    Hybrid Recommendation Engine: Blends TF-IDF Text Similarity with Geographic Weighting.
    """
    if not req.candidates:
        return []

    # 1. Prepare Text Data for TF-IDF
    contents = [req.target_content] + [item['content'] for item in req.candidates]
    
    # Vectorize
    tfidf = TfidfVectorizer(stop_words='english')
    matrix = tfidf.fit_transform(contents)
    
    # Calculate Text Similarity with the Target (Index 0)
    # This gives us an array of base scores from 0.0 to 1.0
    cosine_sim = cosine_similarity(matrix[0:1], matrix[1:])[0]
    
    # 2. Apply Location Weighting Algorithm
    LOCATION_WEIGHT = 0.3 # Adjust this to make location more or less important!
    
    final_scores = []
    
    for idx, candidate in enumerate(req.candidates):
        base_score = cosine_sim[idx]
        
        # Check if the candidate's location contains the user's target city
        location_boost = 0.0
        if req.target_location and req.target_location.lower() in candidate.get('location', '').lower():
            location_boost = LOCATION_WEIGHT
            
        # Blend the scores
        total_score = base_score + location_boost
        final_scores.append((candidate['id'], total_score))
    
    # 3. Sort by the new Hybrid Score
    final_scores = sorted(final_scores, key=lambda x: x[1], reverse=True)
    
    # Return Top 3 IDs
    top_indices = [item[0] for item in final_scores[:3]]
    
    return top_indices