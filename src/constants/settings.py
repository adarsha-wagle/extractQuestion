import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# Access variables
class Settings : 
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

settings = Settings()