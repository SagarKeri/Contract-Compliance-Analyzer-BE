import requests
from langchain_huggingface import HuggingFaceEmbeddings
from pymongo import MongoClient

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client["Contract-Compliance-Analyzer"]

# Embedding model setup
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# JWT Configuration
SECRET_KEY = "your-secret-key"  # Replace with a strong secret key in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# LLM Query Endpoint
def query_endpoint(param_string):
    url = "http://172.16.117.136:8001/query"
    # Preserve JSON structure, only trim outer whitespace
    cleaned_string = param_string.strip() if param_string else ""
    payload = {"prompt": cleaned_string}
    try:
        print("Payload:", payload)
        response = requests.post(url, json=payload)
        print("Status code:", response.status_code)
        print("Raw response:", response.text)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print("Request error:", str(e))
        return f"Error: {str(e)}"