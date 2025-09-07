import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# 1. Initialize FastAPI app
app = FastAPI(title="Smart Drainage API")

# 2. Define the structure of the data the app will send
class FieldData(BaseModel):
    soil_moisture_percent: int
    rainfall_last_7_days_mm: int
    soil_type: str
    slope_degrees: float
    historical_complaints: int

# 3. Load your trained model and the column names
# These files must be in the same directory as this script.
try:
    model = joblib.load('drainage_risk_classifier.joblib')
    training_columns = joblib.load('training_columns.joblib')
except FileNotFoundError:
    # This provides a helpful error if the model files are missing
    print("ERROR: Model or training column files not found. Make sure they are in the same directory.")
    model = None
    training_columns = None


# 4. Create the prediction endpoint
@app.post("/predict")
def predict_risk_level(data: FieldData):
    if not model or not training_columns:
        return {"error": "Model not loaded. Check server logs."}

    # Convert incoming data to a DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    # One-hot encode the data exactly as in training
    input_df_encoded = pd.get_dummies(input_df)
    
    # Align columns to match the model's training data
    # This ensures any missing dummy columns are added with a value of 0
    final_df = input_df_encoded.reindex(columns=training_columns, fill_value=0)
    
    # Make the prediction
    prediction = model.predict(final_df)
    
    # Return the result in a JSON format
    return {"predicted_risk_level": prediction[0]}

