# <db.py>
from config import mongo_client, mongo_db, redis_client
from upstash_redis.asyncio import Redis  # For async operations
import json
from bson import ObjectId
from datetime import timedelta, datetime
from logging_config import setup_logger
from pymongo import ASCENDING, DESCENDING

logger = setup_logger(__name__)

# MongoDB client setup - using configured client from config.py
client = mongo_client
db = mongo_db

# Redis client setup - using configured client from config.py
r = redis_client
async_redis = Redis.from_env()  # For async operations

# Export the users collection
users_collection = db["user"]
nodes_collection = db["node"]

activityCollection = db["activity"]
feedCollection = db["feed"]
logsCollection = db["logs"]
webpageCollection = db["webpage"]

# Renamed from goalsCollection to groupsCollection:
groupsCollection = db["groups"]
stageHistoryCollection = db["stageHistory"]

workflowCollection = db["workflowTasks"]

nodeNotesCollection = db["notes"]
cadenceActivityCollection = db["cadenceActivity"]
cadenceCollection = db["cadence"]
highLevelProfileInsightsCollection = db["highLevelProfileInsights"]
searchOutputCollection = db["searchOutput"]
processedWebhooksCollection = db["processedWebhooks"]

# Add new collection for smart filters
smartFiltersCollection = db["smartFilters"]

# Add new collection for feedback
feedbackCollection = db["feedback"]

# Add new collection for invite codes
invitesCollection = db["invites"]

# --- NEW: Collection for Platform Votes ---
platformVotesCollection = db["platformVotes"]

# --- NEW: Collection for Shared Insights ---
sharedCollection = db["shared"]
# -----------------------------------------

# Collections for subscription management
subscriptionsCollection = db["subscriptions"]
invoicesCollection = db["invoices"]

# Collections for LinkedIn session management
linkedin_sessions_collection = db["linkedin_sessions"]
session_events_collection = db["session_events"]
# Add new collection
task_runs_collection = db["taskruns"]

task_runs_collection.create_index([("status", 1), ("startTime", -1)])
task_runs_collection.create_index([("userId", 1), ("status", 1)])
task_runs_collection.create_index([("expectedEndTime", 1), ("status", 1)])
    
# Users indexes
users_collection.create_index([("dailyUsage.date", 1)])
users_collection.create_index([("processingLock", 1)])

# Nodes indexes
nodes_collection.create_index([("userId", 1), ("deepScan", 1), ("lastDeepScan", 1)])
nodes_collection.create_index([("userId", 1), ("connectionLevel", 1), ("scrapped", 1)])

# Create indexes for subscriptionsCollection
subscriptionsCollection.create_index([
    ("userId", ASCENDING),
    ("subscription_id", ASCENDING)
])

subscriptionsCollection.create_index([
    ("customer_id", ASCENDING),
    ("subscription_id", ASCENDING)
])

subscriptionsCollection.create_index([
    ("product_id", ASCENDING),
    ("status", ASCENDING)
])

# Create indexes for invoicesCollection
invoicesCollection.create_index([
    ("payment_id", ASCENDING),
    ("customer_id", ASCENDING)
])

nodes_collection.create_index({"userId": 1, "workExperience.webpageId": 1})

# Create indexes
stageHistoryCollection.create_index(
    [("nodeId", 1), ("userId", 1), ("timestamp", -1)])

# Indexes for cadenceActivityCollection
cadenceActivityCollection.create_index([
    ("userId", ASCENDING),
    ("nodeId", ASCENDING),
    ("status", ASCENDING),
    ("updatedAt", DESCENDING)
])
cadenceActivityCollection.create_index([
    ("userId", ASCENDING),
    ("nodeId", ASCENDING),
    ("status", ASCENDING),
    ("scheduledFor", ASCENDING)
])
cadenceActivityCollection.create_index([
    ("userId", ASCENDING),
    ("nodeId", ASCENDING),
    ("cadenceId", ASCENDING),
    ("status", ASCENDING)
])

# Add an index on userId, nodeId, and scheduledFor for queries that don't include a status filter.
cadenceActivityCollection.create_index([
    ("userId", ASCENDING),
    ("nodeId", ASCENDING),
    ("scheduledFor", ASCENDING)
], name="user_node_scheduledFor_index")

# Indexes for groupsCollection
groupsCollection.create_index([
    ("userId", ASCENDING),
    ("_id", ASCENDING)
])

# Indexes for nodes_collection
nodes_collection.create_index([
    ("userId", ASCENDING),
    ("_id", ASCENDING)
])

searchOutputCollection.create_index([
    ("userId", ASCENDING),
    ("createdAt", DESCENDING)
])

nodes_collection.create_index([
    ("name", "text"),
    ("orgString", "text"),
    ("bio", "text"),
    ("about", "text")
], name="text_search_index")

# Add index for groups field
nodes_collection.create_index([("groups.groupId", 1)], name="groups_index")

# Notes
nodeNotesCollection.create_index(
    [("content", "text")], name="text_search_index")

# Groups
groupsCollection.create_index([
    ("title", "text"),
    ("description", "text")
], name="text_search_index")

# --- Recommended Indexes for Webpage Collection ---

# Compound index for faster queries on platform, scrapped status, and scrappedAt timestamp.
webpageCollection.create_index([
    ("platform", ASCENDING),
    ("scrapped", ASCENDING),
    ("scrappedAt", ASCENDING)
])

# Index on URL field to quickly fetch webpage documents by URL.
webpageCollection.create_index([("url", ASCENDING)])
# Optionally, if you expect unique URLs, you could enforce uniqueness:
# webpageCollection.create_index([("url", ASCENDING)], unique=True)

# Add these indexes to existing webpageCollection indexes
webpageCollection.create_index(
    [("_id", ASCENDING), ("platform", ASCENDING),
     ("scrapped", ASCENDING), ("scrappedAt", ASCENDING)],
    name="id_platform_scrapped_scrappedAt"
)

webpageCollection.create_index(
    [("_id", ASCENDING), ("platform", ASCENDING), ("scrapped", ASCENDING)],
    name="id_platform_scrapped"
)

# Create indexes for smartFiltersCollection
smartFiltersCollection.create_index([
    ("userId", ASCENDING),
    ("groupId", ASCENDING),
    ("createdAt", DESCENDING)
])

# Create index for feedbackCollection
feedbackCollection.create_index([
    ("userId", ASCENDING),
    ("timestamp", DESCENDING)
])

# Create indexes for invitesCollection
invitesCollection.create_index([("inviteCode", ASCENDING)], unique=True)
invitesCollection.create_index([
    ("isClaimed", ASCENDING),
    ("claimedByUserId", ASCENDING)  # Index claimedByUserId even if sparse
])

# Create index for usersCollection for hasClaimedInvite
users_collection.create_index([
    # Index for checking if user has claimed an invite
    ("hasClaimedInvite", ASCENDING)
])

# --- NEW: Index for Shared Collection ---
# Ensures only one share record per user/node combination
sharedCollection.create_index(
    [("userId", ASCENDING), ("nodeId", ASCENDING)], unique=True, name="user_node_share_unique")
# Index for fast lookup by shareId (_id) is created automatically by MongoDB
# ----------------------------------------------

# --- NEW: Indexes for Platform Votes Collection ---
platformVotesCollection.create_index([("userId", ASCENDING)])
platformVotesCollection.create_index([("platform", ASCENDING)])
platformVotesCollection.create_index(
    [("userId", ASCENDING), ("platform", ASCENDING)], unique=True, name="user_platform_vote_unique")
# ----------------------------------------------

# Helper functions for Refresh Tokens using Redis


def store_refresh_token(refresh_token: str, user_id: str, expires_in: int = 30 * 24 * 60 * 60):
    """
    Store the refresh token in Redis with an expiration time.
    Default expiration is 30 days.
    """
    logger.info(f"Storing refresh token: {refresh_token} for user: {user_id}")
    key = f"refresh_token:{refresh_token}"
    r.set(key, user_id, ex=expires_in)


def get_user_id_by_refresh_token(refresh_token: str):
    """
    Retrieve the user ID associated with the given refresh token.
    """
    logger.info(f"Getting user ID for refresh token: {refresh_token}")
    key = f"refresh_token:{refresh_token}"
    user_id = r.get(key)
    if user_id:
        return user_id.decode('utf-8') if isinstance(user_id, bytes) else user_id
    return None


def delete_refresh_token(refresh_token: str):
    """
    Delete the refresh token from Redis.
    """
    logger.info(f"Deleting refresh token: {refresh_token}")
    key = f"refresh_token:{refresh_token}"
    r.delete(key)


def extend_refresh_token_ttl(refresh_token, user_id):
    """
    Extend the TTL of an existing refresh token if it's still valid.
    Returns True if extended, False if token doesn't exist.
    """
    token_key = f"refresh_token:{refresh_token}"
    user_id_key = f"refresh_token_user:{refresh_token}"

    # Check if token exists and get current mapping
    exists = r.exists(token_key)
    current_user_id = r.get(token_key)

    if not exists or (current_user_id.decode('utf-8') if isinstance(current_user_id, bytes) else current_user_id) != user_id:
        return False

    # Extend TTL to 30 days
    thirty_days = 30 * 24 * 60 * 60
    r.expire(token_key, thirty_days)
    r.expire(user_id_key, thirty_days)

    return True


def get_refresh_token_expiry(refresh_token):
    """
    Get remaining TTL (Time To Live) for a refresh token in seconds.
    Returns None if token doesn't exist.
    """
    ttl = r.ttl(f"refresh_token:{refresh_token}")
    return ttl if ttl > 0 else None


async def cleanup_expired_refresh_tokens():
    """
    Cleanup expired refresh tokens from Redis.
    This function should be called periodically to maintain Redis storage.
    """
    try:
        deleted_tokens = []
        # Use Redis SCAN to efficiently iterate through keys
        cursor = "0"
        while True:
            cursor, keys = await async_redis.scan(cursor=cursor, match="refresh_token:*")
            for key in keys:
                # TTL returns -2 if key doesn't exist, -1 if no expiry, or seconds remaining
                ttl = await async_redis.ttl(key)
                if ttl == -2 or ttl == 0:  # Key doesn't exist or just expired
                    token = key.replace("refresh_token:", "")
                    user_id = await async_redis.get(f"refresh_token:{token}")
                    deleted_tokens.append({
                        "token": token,
                        "user_id": user_id.decode('utf-8') if isinstance(user_id, bytes) else user_id,
                        "ttl": ttl
                    })
                    await async_redis.delete(key)
            if cursor == "0":
                break
    except Exception as e:
        print(f"Error cleaning up refresh tokens: {str(e)}")
        logsCollection.insert_one({
            "type": "REFRESH_TOKEN_CLEANUP_ERROR",
            "timestamp": datetime.utcnow(),
            "error": str(e)
        })
