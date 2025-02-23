# FASTAPI-main

 # Fruit Classifier Web App

A simple web application for fruit classification using machine learning. The app allows users to upload new data, retrain the model, and predict fruit types based on input features such as mass, width, height, and color score.

## Features

- **User Authentication**: Login and Registration functionality using JWT (JSON Web Tokens).
- **Fruit Prediction**: Predict fruit types (Apple, Mandarin, Orange, Lemon) based on the given features.
- **Model Retraining**: Retrain the model with new data uploaded by the user.
- **Dashboard**: A user-friendly dashboard for uploading files, viewing model performance metrics, and predicting fruit types.

## Requirements

- Python 3.8+
- PostgreSQL database (for user management)
- Redis (for caching)
- FastAPI
- scikit-learn
- joblib
- pandas
- SQLAlchemy
- Jinja2 (for HTML rendering)
- passlib (for password hashing)
- dotenv (for environment variables)

## Setup

### 1. Clone the repository
```bash
git clone <repository_url>
cd fruit_classifier
```

### 2. Install Dependencies
````
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
````

### 3. Setup Environment Variables
````
DATABASE_URL=postgresql://postgres:your_password@localhost:5432/fruit_classifier_db
SECRET_KEY=your_secret_key
````

### 4. Create Database
Ensure you have PostgreSQL set up and create a database fruit_classifier_db. You can use the following command to access your PostgreSQL instance:
````
psql -U postgres -d postgres
````
Then create the database:
````
CREATE DATABASE fruit_classifier_db;
````

### 5.  Redis Setup
Ensure that Redis is installed and running locally.

### 6. Run the Application
Start the FastAPI server using Uvicorn:
````
uvicorn main:app --reload
````
The app will be available at http://127.0.0.1:8000.

## File Structure
- *api.py*: Main FastAPI application file, which includes routes for authentication, file upload, retraining, and fruit prediction.
  
- *train_model.py* : Script for training and retraining the machine learning model using a dataset of fruit data.
  
- *backend/models/*: Folder containing the saved model (fruit_classifier.joblib), scaler (scaler.joblib), and model metrics (metrics.json).
  
- *data/* : Folder for storing uploaded .txt files containing fruit data.
  
- *templates/*: Folder for storing HTML templates used in the dashboard (login.html, register.html, dashboard.html).
  
- *.env*: Environment variables configuration file.
  
- *requirements.txt*: List of Python dependencies.

## Usage
- *Login/Register*: Users can register and log in to access the dashboard.

- *Upload Data*: Users can upload a .txt file containing fruit data. The fi-le must include columns mass, width, height, and color_score, and the target column fruit_label.

- *Retrain Model*: After uploading data, users can trigger model retraining. The system will automatically reload the updated model and scaler.

- *Fruit Prediction*: Once logged in, users can input fruit data (mass, width, height, color score) and get a prediction of the fruit type.

## Model Training
The model is trained using a RandomForestClassifier with the dataset fruit_data_with_colors.txt. The features used for training include:

- *mass*: Mass of the fruit.
  
- *width*: Width of the fruit.
  
- *height*: Height of the fruit.
  
- *color_score*: A color score based on the fruit's appearance.
  
The model is saved in backend/models/fruit_classifier.joblib and can be retrained by uploading new data.

## Model Metrics
After retraining, the following performance metrics are calculated and saved in metrics.json:

- *Accuracy*: The accuracy score of the model on the test set.
- *Precision*: The weighted precision score.
- *Recall*: The weighted recall score.

## Security

- *JWT Authentication*: The app uses JWT tokens to authenticate users. Tokens are stored in cookies and have a limited expiration time.
  
- *Password Hashing*: User passwords are hashed using passlib for secure storage.
Troubleshooting

- *Model Not Loading*: If the model or scaler cannot be loaded, ensure the paths are correct and that the required files (fruit_classifier.joblib and scaler.joblib) exist in the backend/models/ directory.
  
- *File Upload Issues*: Ensure that the uploaded file is in .txt format and follows the correct structure (mass, width, height, color_score, fruit_label).
