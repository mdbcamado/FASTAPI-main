# backend/db.py
from databases import Database
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get PostgreSQL credentials from environment variables
DATABASE_URL = os.getenv("DATABASE_URL")

# Connect to the database
database = Database(DATABASE_URL)

