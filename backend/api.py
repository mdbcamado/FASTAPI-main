from fastapi import FastAPI, HTTPException, Depends, status, Request, Form, File, UploadFile
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from jose import JWTError, jwt
from passlib.context import CryptContext
import joblib
import numpy as np
from pydantic import BaseModel
import logging
import os
import shutil
import subprocess
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from backend.database import SessionLocal, create_user, get_user, verify_password
import redis
from datetime import datetime, timedelta
import sys
import json

# Load environment variables
load_dotenv()

# Logging setup
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# FastAPI instance
app = FastAPI()

# Jinja2 Templates
templates = Jinja2Templates(directory="templates")

# Secret key for JWT, typically stored in .env
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Ensure SECRET_KEY is set
if not SECRET_KEY:
    raise ValueError("❌ SECRET_KEY is not set in the .env file!")

# Redis Cache setup
cache = redis.StrictRedis(host="localhost", port=6379, db=0, decode_responses=True)

# OAuth2 Password Bearer
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# File paths
UPLOAD_FOLDER = "data/"
MODEL_PATH = "backend/models/fruit_classifier.joblib"
SCALER_PATH = "backend/models/scaler.joblib"

# Load the model and scaler with error handling
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    logger.info("✅ Model and scaler loaded successfully!")
except Exception as e:
    logger.error(f"❌ Error loading model or scaler: {e}")
    model, scaler = None, None

# Fruit label mapping
fruit_labels = {1: "Apple", 2: "Mandarin", 3: "Orange", 4: "Lemon"}

# Utility function to create JWT token
def create_access_token(data: dict, expires_delta: timedelta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Verify JWT token
def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None

# Dependency to get the current user from the token
def get_current_user(token: str = Depends(oauth2_scheme)):
    payload = verify_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials"
        )
    return payload

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Home Page - Login Form
@app.get("/")
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

# Registration Page
@app.get("/register")
def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

# Process Registration
@app.post("/register")
def register_user(request: Request, username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    existing_user = get_user(db, username)
    if existing_user:
        return templates.TemplateResponse("register.html", {"request": request, "error": "Username already exists"})

    create_user(db, username, password)
    return RedirectResponse("/", status_code=303)

# Process Login
@app.post("/token")
def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = get_user(db, form_data.username)
    
    if not user or not verify_password(form_data.password, user.hashed_password):
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid username or password"})
    
    access_token = create_access_token(data={"sub": form_data.username})
    response = RedirectResponse(url="/dashboard", status_code=303)
    response.set_cookie(key="access_token", value=access_token)
    return response

# Dashboard (After Login)
@app.get("/dashboard")
def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/logout")
def logout():
    response = RedirectResponse(url="/")
    response.delete_cookie("access_token")  # Remove authentication token
    return response

# File Upload Route
@app.post("/upload/")
async def upload_file(request: Request, file: UploadFile = File(...)):
    # Ensure authentication
    token = request.cookies.get("access_token")
    if not token or not verify_token(token):
        return RedirectResponse(url="/", status_code=303)

    # Check file format
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are allowed")

    file_path = os.path.join(UPLOAD_FOLDER, "fruit_data_with_colors.txt")

    # Save the file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return templates.TemplateResponse("dashboard.html", {"request": request, "message": "File uploaded successfully"})

# Model Retraining Route

@app.post("/retrain/")
async def retrain_model(request: Request):
    token = request.cookies.get("access_token")
    if not token or not verify_token(token):
        return RedirectResponse(url="/", status_code=303)

    try:
        # Run model training script
        result = subprocess.run([sys.executable, "scripts/train_model.py"], check=True, capture_output=True, text=True)
        logger.info(f"Retrain output: {result.stdout}")

        # Reload updated model and scaler
        global model, scaler
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)

        # Load metrics
        with open("backend/models/metrics.json", "r") as f:
            metrics = json.load(f)

        return templates.TemplateResponse(
            "dashboard.html",
            {
                "request": request,
                "metrics": metrics,
                "message": "✅ Model and scaler retrained successfully!"
            }
        )

    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Retrain subprocess error: {e.stderr}")
        return templates.TemplateResponse(
            "dashboard.html",
            {
                "request": request,
                "message": f"❌ Error retraining model: {e.stderr}"
            }
        )

    except Exception as e:
        logger.error(f"❌ Unexpected error during retraining: {e}")
        return templates.TemplateResponse(
            "dashboard.html",
            {
                "request": request,
                "message": f"❌ Unexpected error: {str(e)}"
            }
        )


# @app.post("/retrain/")
# async def retrain_model(request: Request):
#     # Ensure authentication
#     token = request.cookies.get("access_token")
#     if not token or not verify_token(token):
#         return RedirectResponse(url="/", status_code=303)
    
#     try:
#         # Run model training script
#         result = subprocess.run([sys.executable, "scripts/train_model.py"], check=True)

#         logger.info(f"Retrain output: {result.stdout}")
#         if result.stderr:
#             logger.error(f"Retrain error: {result.stderr}")

#         # Reload the updated model and scaler
#         global model, scaler
#         model = joblib.load(MODEL_PATH)
#         scaler = joblib.load(SCALER_PATH)

#         # Evaluate new model performance
#         df = pd.read_csv(os.path.join(UPLOAD_FOLDER, "fruit_data_with_colors.txt"), delimiter="\t")
#         X = df[['mass', 'width', 'height', 'color_score']]
#         y = df['fruit_label']

#         X_scaled = scaler.transform(X)
#         y_pred = model.predict(X_scaled)

#         metrics = {
#             "accuracy": round(accuracy_score(y, y_pred), 4),
#             "precision": round(precision_score(y, y_pred, average="weighted"), 4),
#             "recall": round(recall_score(y, y_pred, average="weighted"), 4),
#         }

#         return templates.TemplateResponse(
#             "dashboard.html",
#             {
#                 "request": request,
#                 "metrics": metrics,
#                 "message": "✅ Model and scaler retrained successfully!"
#             }
#         )


#     except subprocess.CalledProcessError as e:
#         logger.error(f"❌ Retrain subprocess error: {e.stderr}")
#         return templates.TemplateResponse(
#             "dashboard.html",
#             {
#                 "request": request,
#                 "message": f"❌ Error retraining model: {e.stderr}"
#             }
#         )

#     except Exception as e:
#         logger.error(f"❌ Unexpected error during retraining: {e}")
#         return templates.TemplateResponse(
#             "dashboard.html",
#             {
#                 "request": request,
#                 "message": f"❌ Unexpected error: {str(e)}"
#             }
#         )

# Prediction Route
@app.post("/predict/")
async def predict_fruit(request: Request, mass: float = Form(...), width: float = Form(...), height: float = Form(...), color_score: float = Form(...)):
    # Ensure authentication
    token = request.cookies.get("access_token")
    if not token or not verify_token(token):
        return RedirectResponse(url="/", status_code=303)

    # Ensure model is available
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model is not available")

    # Predict
    input_data = np.array([[mass, width, height, color_score]])
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    fruit_name = fruit_labels.get(int(prediction[0]), "Unknown")

    return templates.TemplateResponse("dashboard.html", {"request": request, "prediction": fruit_name})
