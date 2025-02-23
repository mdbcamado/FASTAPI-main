from fastapi import Depends
from .db import database

# Dependency for the database connection
def get_db():
    try:
        yield database
    finally:
        pass
