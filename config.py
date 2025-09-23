"""
Configuration module for Search Lambda
"""
import os
from pymongo import MongoClient
from upstash_redis import Redis
from upstash_vector import Index
from typing import Any, Optional

# Load .env file for local development only (not needed in Lambda)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available in Lambda environment, which is fine
    pass


def get_env_var(var_name: str, required: bool = True) -> Optional[str]:
    """Get environment variable with optional requirement check"""
    value = os.getenv(var_name)
    if required and value is None:
        raise ValueError(f"Required environment variable {var_name} is not set")
    return value


# MongoDB Configuration
MONGODB_URI = get_env_var("MONGO_URI")
DB_NAME = get_env_var("MONGODB_DB_NAME", required=False) or "finalBackendDB"

# Initialize MongoDB client
mongo_client = MongoClient(MONGODB_URI)
mongo_db = mongo_client[DB_NAME]

# Upstash Configuration (for vector search and Redis caching)
UPSTASH_URL = get_env_var("UPSTASH_URL")
UPSTASH_TOKEN = get_env_var("UPSTASH_TOKEN")

# Redis Configuration (Upstash REST)
UPSTASH_REDIS_REST_URL = get_env_var("UPSTASH_REDIS_REST_URL")
UPSTASH_REDIS_REST_TOKEN = get_env_var("UPSTASH_REDIS_REST_TOKEN")

# Initialize Upstash clients
upstash_client = Index(url=UPSTASH_URL, token=UPSTASH_TOKEN)  # For vector search
redis_client = Redis(url=UPSTASH_REDIS_REST_URL, token=UPSTASH_REDIS_REST_TOKEN)  # For caching
