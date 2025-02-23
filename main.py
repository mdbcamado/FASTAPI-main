import uvicorn
from backend.api import app  # Correct path to FastAPI app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)  # Make sure the API starts
