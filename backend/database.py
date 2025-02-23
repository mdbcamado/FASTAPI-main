from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from passlib.context import CryptContext
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Database connection setup
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

# SessionLocal is a factory for new Session objects
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for declarative models
Base = declarative_base()

# Password hashing setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# User model for database
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)

# Create the database tables (if not already created)
Base.metadata.create_all(bind=engine)

# Dependency to get a session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Utility functions to handle user creation and authentication
def get_password_hash(password: str):
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str):
    return pwd_context.verify(plain_password, hashed_password)

def get_user(db, username: str):
    return db.query(User).filter(User.username == username).first()

def create_user(db, username: str, password: str):
    db_user = User(username=username, hashed_password=get_password_hash(password))
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user
