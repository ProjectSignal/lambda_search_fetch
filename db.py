# Minimal db.py for Fetch Lambda
from config import mongo_client, mongo_db, upstash_client
from logging_config import setup_logger

logger = setup_logger(__name__)

# MongoDB client setup - using configured client from config.py
client = mongo_client
db = mongo_db

# Vector database client (Upstash)
upstash_redis = upstash_client

# MongoDB collections actually used by Fetch
searchOutputCollection = db["searchOutput"]
nodes_collection = db["node"]
webpageCollection = db["webpage"]

# Create only the essential indexes that Fetch actually needs
searchOutputCollection.create_index([
    ("userId", 1),
    ("createdAt", -1)
])

nodes_collection.create_index([
    ("userId", 1),
    ("_id", 1)
])

webpageCollection.create_index([
    ("url", 1)
])