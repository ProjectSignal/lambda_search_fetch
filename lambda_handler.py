import json
import asyncio
import os
import traceback
from datetime import datetime, timezone
from dotenv import load_dotenv
from search_processor import SearchProcessor
from logging_config import setup_logger
from db import searchOutputCollection

# Load environment variables (for local testing)
load_dotenv()

logger = setup_logger(__name__)

class SearchStatus:
    """Search execution status tracking"""
    NEW = "NEW"
    HYDE_COMPLETE = "HYDE_COMPLETE"
    SEARCH_COMPLETE = "SEARCH_COMPLETE"
    RANK_AND_REASONING_COMPLETE = "RANK_AND_REASONING_COMPLETE"
    ERROR = "ERROR"

def get_utc_now():
    """Returns current UTC datetime in ISO format"""
    return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()

# REMOVED: Unused helper functions for Step Functions architecture
# def _extract_event_payload(event):
# def _get_event_value(event, key):

async def _run(event):
    """Main async execution logic for Search Lambda"""
    start_time = datetime.utcnow()
    search_id = None
    
    try:
        # Extract required parameters from Step Functions event
        search_id = event.get('searchId')
        user_id = event.get('userId')
        query = event.get('query')
        flags = event.get('flags', {})

        if not all([search_id, user_id, query]):
            error_msg = "Missing required fields: searchId, userId, and query"
            logger.warning(error_msg)
            return {
                "statusCode": 400,
                "body": json.dumps({
                    "error": error_msg,
                    "success": False
                })
            }

        logger.info(f"Processing Search for searchId: {search_id}, user: {user_id}, query: {query}")

        # Get search document and verify HyDE analysis is complete
        search_doc = searchOutputCollection.find_one({"_id": search_id})
        if not search_doc:
            error_msg = f"Search document not found for searchId: {search_id}"
            logger.error(error_msg)
            return {
                "statusCode": 404,
                "body": json.dumps({
                    "error": error_msg,
                    "success": False
                })
            }

        if search_doc.get("status") != SearchStatus.HYDE_COMPLETE:
            error_msg = f"HyDE analysis not complete for searchId: {search_id}, status: {search_doc.get('status')}"
            logger.error(error_msg)
            return {
                "statusCode": 400,
                "body": json.dumps({
                    "error": error_msg,
                    "success": False
                })
            }

        # Extract hyde analysis from search document
        hyde_analysis = search_doc.get("hydeAnalysis")
        if not hyde_analysis:
            error_msg = f"No HyDE analysis found in search document: {search_id}"
            logger.error(error_msg)
            return {
                "statusCode": 400,
                "body": json.dumps({
                    "error": error_msg,
                    "success": False
                })
            }

        # Convert to the format expected by FetchAndRankProcessor
        hyde_output = {
            "query_breakdown": hyde_analysis.get("queryBreakdown", {}),
            "response": hyde_analysis.get("response", {})
        }

        logger.info("Retrieved HyDE analysis from searchOutput collection")

        # Initialize the processor
        processor = SearchProcessor()

        # Process the search and prepare candidates
        result = await processor.process(
            hyde_output=hyde_output,
            user_id=user_id,
            query=query,
            flags=flags
        )

        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds()

        # Update search document with results
        now = datetime.utcnow()

        # Store full candidate data for ranking in Reasoning Lambda
        candidates = result.get("ranked_results", [])  # These are now full candidates, not ranked

        # Limit candidates to avoid document bloat
        if len(candidates) > 200:
            candidates = candidates[:200]

        update_result = searchOutputCollection.update_one(
            {"_id": search_id, "status": {"$in": [SearchStatus.HYDE_COMPLETE, SearchStatus.SEARCH_COMPLETE]}},
            {
                "$set": {
                    "results": {
                        "summary": {
                            "count": len(candidates),
                            "topK": len(candidates),
                            "idsOnly": False
                        },
                        "candidates": candidates  # Store full candidates instead of ranked results
                    },
                    "searchMetrics": result.get("search_metrics", {}),
                    "status": SearchStatus.SEARCH_COMPLETE,
                    "metrics.searchMs": processing_time * 1000,
                    "updatedAt": now
                },
                "$addToSet": {
                    "events": {
                        "id": f"SEARCH:{search_id}",
                        "stage": "SEARCH",
                        "message": f"Search completed, {len(candidates)} candidates found",
                        "timestamp": now
                    }
                }
            }
        )

        if update_result.matched_count == 0:
            # Check if document already in SEARCH_COMPLETE status (idempotent retry)
            existing_doc = searchOutputCollection.find_one({"_id": search_id})
            if existing_doc and existing_doc.get("status") == SearchStatus.SEARCH_COMPLETE:
                logger.info(f"Search document {search_id} already processed (idempotent retry)")
                # Return success since work is already done
                existing_candidates = existing_doc.get("results", {}).get("candidates", [])
                return {
                    "statusCode": 200,
                    "body": json.dumps({
                        "searchId": search_id,
                        "success": True,
                        "candidateCount": len(existing_candidates),
                        "processing_time": processing_time,
                        "processed_at": get_utc_now(),
                        "note": "Already processed (idempotent)"
                    })
                }
            else:
                error_msg = f"Failed to update search document for searchId: {search_id} - invalid status transition"
                logger.error(error_msg)
                return {
                    "statusCode": 409,  # Conflict status
                    "body": json.dumps({
                        "error": error_msg,
                        "success": False
                    })
                }

        logger.info(f"Updated search document {search_id} with {len(candidates)} candidates")

        return {
            "statusCode": 200,
            "body": json.dumps({
                "searchId": search_id,
                "success": True,
                "candidateCount": len(candidates),
                "processing_time": processing_time,
                "processed_at": get_utc_now()
            })
        }

    except Exception as e:
        logger.error(f"Error in FetchAndRank Lambda: {str(e)}", exc_info=True)
        
        # Update search document with error state if we have searchId
        if search_id:
            try:
                now = datetime.utcnow()
                searchOutputCollection.update_one(
                    {"_id": search_id},
                    {
                        "$set": {
                            "status": SearchStatus.ERROR,
                            "error": {
                                "stage": "SEARCH",
                                "message": str(e),
                                "stackTrace": traceback.format_exc(),
                                "occurredAt": now
                            },
                            "updatedAt": now
                        },
                        "$push": {
                            "events": {
                                "stage": "SEARCH",
                                "message": f"Error: {str(e)}",
                                "timestamp": now
                            }
                        }
                    }
                )
            except Exception as db_error:
                logger.error(f"Failed to update error state: {db_error}")
        
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": str(e),
                "success": False
            })
        }

def lambda_handler(event, context):
    """
    Lambda entry point - synchronous wrapper for async execution

    Expected event format from Step Functions:
    {
        "searchId": "uuid-string",
        "userId": "507f1f77bcf86cd799439012", 
        "query": "search query string",
        "flags": {...}  # Optional search flags
    }

    Reads HyDE analysis from searchOutput collection using searchId.

    Environment variables required:
    - MONGODB_URI: MongoDB connection string
    - UPSTASH_URL: Upstash Vector DB URL
    - UPSTASH_TOKEN: Upstash auth token
    - REDIS_HOST: Redis host for caching
    - REDIS_PORT: Redis port
    - REDIS_PASSWORD: Redis password
    """
    return asyncio.run(_run(event))