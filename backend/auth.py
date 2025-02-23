from datetime import datetime, timedelta
from jose import JWTError, jwt
from pydantic import BaseModel
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Fetch the secret key
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30  # Token expiration time

class User(BaseModel):
    username: str

class UserInDB(User):
    password: str

def create_access_token(data: dict, expires_delta: timedelta = None):
    """Create JWT token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime() + expires_delta
    else:
        expire = datetime() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str):
    """Verify JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None

def get_current_user(token: str):
    """Extract current user from token."""
    return verify_token(token)
