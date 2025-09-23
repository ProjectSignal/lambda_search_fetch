# search.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re
import time
import traceback
import asyncio
import logging
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import List, Dict
from asyncio import Semaphore
import requests
from logic.cloudflareFunctions import fetchImage as fetch_image, fetchImageBatch as fetch_image_batch

from logic.utils import format_datetime
from bson.objectid import ObjectId

from logging_config import setup_logger
logger = setup_logger(__name__)
from config import upstash_client
from db import nodes_collection, webpageCollection
from threading import BoundedSemaphore
from logic.search_config import SearchLimits

# Initialize search limits configuration
search_limits = SearchLimits()
mongoCollectionNodes = nodes_collection


# ------------------------------------------------------------------------------
# Utility function for ObjectId conversion
# ------------------------------------------------------------------------------
def convert_objectids_to_strings(obj):
    """
    Recursively convert ObjectIds to strings in nested data structures.
    Handles dictionaries, lists, and ObjectId objects.
    """
    if isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, dict):
        return {key: convert_objectids_to_strings(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_objectids_to_strings(item) for item in obj]
    else:
        return obj

# ------------------------------------------------------------------------------
# Batch processing for avatar URLs (unchanged)
# ------------------------------------------------------------------------------
_avatar_url_batch: List[str] = []
_avatar_url_results: Dict[str, str] = {}
_BATCH_SIZE = 50

def process_avatar_urls_batch():
    global _avatar_url_batch, _avatar_url_results
    if not _avatar_url_batch:
        return
    results = fetch_image_batch(_avatar_url_batch)
    _avatar_url_results.update(results)
    _avatar_url_batch.clear()

def get_avatar_url(raw_url: str) -> str:
    if not raw_url:
        return ""
    if raw_url.startswith(("http://media.licdn.com", "https://media.licdn.com")):
        return raw_url
    if raw_url in _avatar_url_results:
        return _avatar_url_results[raw_url]
    _avatar_url_batch.append(raw_url)
    if len(_avatar_url_batch) >= _BATCH_SIZE:
        process_avatar_urls_batch()
        return _avatar_url_results.get(raw_url, raw_url)
    return raw_url

def process_mutuals(mutual_ids):
    """
    Process mutual connections by fetching their details from MongoDB.
    Returns a list of objects containing personId, name, and avatarURL.
    mutual_ids can be either a list of ObjectIds or a list of dicts with $oid
    """
    if not mutual_ids:
        return []
    try:
        # If mutual_ids are already ObjectIds, use them directly
        # Otherwise, convert only if needed
        object_ids = []
        for mid in mutual_ids:
            if isinstance(mid, ObjectId):
                object_ids.append(mid)
            elif isinstance(mid, dict) and '$oid' in mid:
                # Convert $oid string to ObjectId only once
                object_ids.append(ObjectId(mid['$oid']))
            elif mid:
                object_ids.append(ObjectId(str(mid)))
        if not object_ids:
            return []
            
        # Query MongoDB directly with ObjectIds
        mutuals = list(mongoCollectionNodes.find(
            {"_id": {"$in": object_ids}},
            {"_id": 1, "name": 1, "avatarURL": 1}
        ))
        if len(mutuals) < len(object_ids):
            logger.warning(f"Some mutual IDs were not found in MongoDB. Found {len(mutuals)} out of {len(object_ids)}")
        
        # Collect all avatar URLs for batch processing
        avatar_urls = [mutual.get("avatarURL", "") for mutual in mutuals if mutual.get("avatarURL")]
        if avatar_urls:
            global _avatar_url_batch
            _avatar_url_batch.extend(avatar_urls)
            process_avatar_urls_batch()
        return [{
            "personId": str(mutual["_id"]),
            "name": mutual.get("name", ""),
            "avatarURL": _avatar_url_results.get(mutual.get("avatarURL", ""), mutual.get("avatarURL", ""))
        } for mutual in mutuals]
    except Exception as e:
        logger.error(f"Error processing mutuals: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return []

# ------------------------------------------------------------------------------
# Jina Embeddings (unchanged)
# ------------------------------------------------------------------------------
JINA_API_KEY = os.getenv("JINA_API_KEY", "")
JINA_API_URL = 'https://api.jina.ai/v1/embeddings'
JINA_HEADERS = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {JINA_API_KEY}'
}

def get_jina_embeddings(texts):
    if not texts:
        return []
    data = {
        "model": "jina-embeddings-v3",
        "task": "text-matching",
        "late_chunking": False,
        "dimensions": 1024,
        "embedding_type": "float",
        "input": texts
    }
    try:
        response = requests.post(JINA_API_URL, headers=JINA_HEADERS, json=data)
        response.raise_for_status()
        result = response.json()
        embeddings = [item["embedding"] for item in result["data"]]
        logger.info(f"Generated embeddings for {len(embeddings)} texts")
        for i, emb in enumerate(embeddings):
            if not isinstance(emb, list) or len(emb) != 1024:
                raise ValueError(
                    f"Invalid embedding at index {i}. Expected 1024-dim vector, got: {len(emb) if isinstance(emb, list) else type(emb)}"
                )
        return embeddings
    except Exception as e:
        logger.error(f"Error getting Jina embeddings: {e}")
        traceback.print_exc()
        return []

# ------------------------------------------------------------------------------
# Batching Utility: Skills (unchanged)
# ------------------------------------------------------------------------------
def batch_embed_skills(skills: list, alternative_skills: bool):
    """
    Collect all skill descriptions (and related role descriptions if alternative_skills=True)
    that do not already have "embeddings". Make one Jina call, then assign them back.
    Each skill dict has the form:
        {
          "name": "...",
          "description": "...",
          "embeddings": optional [...],
          "titleKeywords": optional [...],
          "relatedRoles": [
            {"name":"...", "description":"...", "embeddings": optional [...]} ...
          ]
        }
    """
    texts_to_embed = []
    index_map = []
    for skill_idx, skl in enumerate(skills):
            
        if "embeddings" not in skl or not skl["embeddings"]:
            desc = skl.get("description", "").strip()
            if desc:
                texts_to_embed.append(desc)
                index_map.append((skill_idx, False, None))
        if alternative_skills and isinstance(skl.get("relatedRoles"), list):
            for rr_idx, rr in enumerate(skl["relatedRoles"]):
                # Handle both dict and string related roles
                if isinstance(rr, dict):
                    if "embeddings" not in rr or not rr["embeddings"]:
                        rr_desc = rr.get("description", "").strip()
                        if rr_desc:
                            texts_to_embed.append(rr_desc)
                            index_map.append((skill_idx, True, rr_idx))
                elif isinstance(rr, str):
                    # If rr is a string, use it as the description
                    rr_desc = rr.strip()
                    if rr_desc:
                        texts_to_embed.append(rr_desc)
                        index_map.append((skill_idx, True, rr_idx))
    if not texts_to_embed:
        return
    new_embeddings = get_jina_embeddings(texts_to_embed)
    if len(new_embeddings) != len(index_map):
        logger.warning(
            f"batch_embed_skills mismatch: got {len(new_embeddings)} embeddings but expected {len(index_map)}"
        )
        return
    for emb_idx, emb in enumerate(new_embeddings):
        skill_idx, is_related, rr_idx = index_map[emb_idx]
        if not is_related:
            skills[skill_idx]["embeddings"] = emb
        else:
            # Handle both dict and string related roles
            rr = skills[skill_idx]["relatedRoles"][rr_idx]
            if isinstance(rr, dict):
                skills[skill_idx]["relatedRoles"][rr_idx]["embeddings"] = emb
            elif isinstance(rr, str):
                # Convert string to dict with embeddings
                skills[skill_idx]["relatedRoles"][rr_idx] = {
                    "name": rr,
                    "description": rr,
                    "embeddings": emb
                }

# ------------------------------------------------------------------------------
# Helper function to analyze HyDE data requirements
# ------------------------------------------------------------------------------
def analyze_hyde_data_requirements(hyde_result: dict) -> dict:
    """
    Analyze HyDE result to determine what additional MongoDB fields to fetch
    for proper reasoning/scoring.
    
    Returns:
        dict: {
            'additional_fields': list of MongoDB fields to fetch,
            'db_queries': list of database queries for context
        }
    """
    required_fields = set()
    db_queries = []
    
    if not hyde_result:
        return {'additional_fields': [], 'db_queries': []}
    
    # Check if this is the nested structure with 'response' key
    hyde_response = hyde_result.get('response', hyde_result)
    
    # Extract database queries if present
    if hyde_response.get('dbBasedQuery', False):
        db_details = hyde_response.get('dbQueryDetails', {})
        queries = db_details.get('queries', [])
        
        for query in queries:
            field = query.get('field', '')
            if field:
                db_queries.append(query)
                
                # Determine which MongoDB collections/fields to fetch
                if field.startswith('education.'):
                    required_fields.add('education')
                elif field.startswith('accomplishments.'):
                    required_fields.add('accomplishments')
                elif field.startswith('workExperience.') and not field.startswith('workExperience.0.'):
                    # Full work history, not just current (which we already fetch)
                    required_fields.add('workExperience')
                elif field.startswith('certifications.'):
                    required_fields.add('accomplishments')  # Certifications are under accomplishments
    
    return {
        'additional_fields': list(required_fields), 
        'db_queries': db_queries
    }

# ------------------------------------------------------------------------------
# MAIN RANKING FUNCTION (updated to fetch additional structured data)
# ------------------------------------------------------------------------------
async def enrich_candidates(people_data: dict, query: str, newReasoningFlag: bool = True, reasoning_model: str = 'gemini', hyde_analysis_flags: dict = None, hyde_result: dict = None) -> list:
    """
    Main function to handle the final ranking step for people_data.
    Now enhanced to fetch additional structured data based on HyDE analysis for better reasoning.
    
    Args:
        people_data (dict): Dictionary containing people data to rank
        query (str): Original search query
        newReasoningFlag (bool): Whether to use new reasoning logic
        reasoning_model (str): Model to use for reasoning (e.g. 'groq_llama')
        hyde_analysis_flags (dict): Pre-analyzed flags from Hyde {skill_present: bool, location_present: bool, entity_present: bool}
        hyde_result (dict): Full HyDE analysis result containing database queries and field requirements
    """
    logger.info("###LOGS: Starting data transformation and recommendation")
    start_time = time.time()
    total_people = len(people_data['people'])
    logger.info(f"###LOGS: Starting with {total_people} total nodes")
    
    # ========== TEMPORARY DEBUG CODE - REMOVE AFTER TESTING ==========
    # # Save input data for validation
    # import json
    # import os
    # debug_timestamp = int(time.time())
    # debug_folder = "debug_ranking"
    # os.makedirs(debug_folder, exist_ok=True)
    # 
    # # Save query and Hyde result
    # debug_data = {
    #     "timestamp": debug_timestamp,
    #     "query": query,
    #     "hyde_result": hyde_result,
    #     "people_data_summary": {
    #         "total_people": len(people_data['people']),
    #         "person_ids": list(people_data['people'].keys())[:5]  # First 5 IDs for reference
    #     },
    #     "reasoning_model": reasoning_model,
    #     "hyde_analysis_flags": hyde_analysis_flags
    # }
    # 
    # with open(f"{debug_folder}/input_data_{debug_timestamp}.json", 'w') as f:
    #     json.dump(debug_data, f, indent=2, default=str)
    # logger.info(f"###DEBUG: Saved input data to {debug_folder}/input_data_{debug_timestamp}.json")
    # ================================================================
    
    # Analyze HyDE requirements to determine additional fields to fetch
    hyde_requirements = analyze_hyde_data_requirements(hyde_result)
    additional_fields = hyde_requirements.get('additional_fields', [])
    db_queries = hyde_requirements.get('db_queries', [])
    
    if additional_fields:
        logger.info(f"###LOGS: HyDE analysis requires additional fields: {additional_fields}")
    if db_queries:
        logger.info(f"###LOGS: Found {len(db_queries)} database queries for context")
    
    try:
        person_ids = [ObjectId(person_id) for person_id in people_data["people"].keys()]
        
        # Build projection dynamically based on HyDE requirements
        base_projection = {
            "_id": 1,
            "about": 1,
            "name": 1,
            "currentLocation": 1,
            "workExperience": 1,
            "scrapped": 1,
            "connectionLevel": 1,
            "linkedinUsername": 1,
            "linkedinHeadline": 1,
            "contacts": 1,
            "avatarURL": 1,
            "mutual": 1,
            "stage": 1,  # Include stage for response
            "education": 1,  # Always include for jsonToXml
            "accomplishments": 1,  # Always include for jsonToXml
            "volunteering": 1  # Always include for jsonToXml
        }
        
        # Add additional fields based on HyDE analysis
        for field in additional_fields:
            base_projection[field] = 1
            
        docs = mongoCollectionNodes.find(
            {"_id": {"$in": person_ids}},
            base_projection
        )
        mongo_docs = {str(doc["_id"]): doc for doc in docs}
        logger.info(f"###LOGS: Fetched {len(mongo_docs)} documents from MongoDB")
        avatar_urls = [doc.get("avatarURL", "") for doc in mongo_docs.values() if doc.get("avatarURL")]
        if avatar_urls:
            global _avatar_url_batch
            _avatar_url_batch.extend(avatar_urls)
            process_avatar_urls_batch()
        missing_ids = set(people_data["people"].keys()) - set(mongo_docs.keys())
        if missing_ids:
            logger.warning(f"###LOGS: Missing documents for IDs: {missing_ids}")
    except Exception as e:
        logger.error(f"###LOGS: Error in MongoDB fetch: {str(e)}")
        return []
    reasoning_transform_people = []
    for person_id, person_data in people_data["people"].items():
        mongo_doc = mongo_docs.get(person_id)
        if not mongo_doc:
            logger.warning(f"###LOGS: No MongoDB document for person {person_id}")
            continue
        if not mongo_doc.get("scrapped"):
            logger.warning(f"###LOGS: Person {person_id} not marked as scrapped")
            continue
        try:
            transformed_person = {
                "personId": person_id,
                "userId": person_data["userId"],
                "name": mongo_doc["name"],
                "aboutMe": mongo_doc.get("about", ""),
                "currentLocation": mongo_doc.get("currentLocation", ""),
                "avatarURL": _avatar_url_results.get(mongo_doc.get("avatarURL", ""), mongo_doc.get("avatarURL", "")),
                "mutuals": process_mutuals(mongo_doc.get("mutual", []) or mongo_doc.get("contacts", {}).get("mutuals", []))
            }
            
            # Add all structured data for rich XML conversion (always include these for jsonToXml)
            transformed_person["linkedinHeadline"] = mongo_doc.get("linkedinHeadline", "")
            transformed_person["education"] = mongo_doc.get("education", [])
            transformed_person["accomplishments"] = mongo_doc.get("accomplishments", {})
            transformed_person["volunteering"] = mongo_doc.get("volunteering", [])
            transformed_person["workExperience"] = mongo_doc.get("workExperience", [])
            
            # Note: Database query context is now passed at hyde analysis level, not per person
            work_exp = mongo_doc.get("workExperience", [])
            if work_exp:
                first_exp = work_exp[0]
                transformed_person["currentWork"] = {
                    "companyName": first_exp.get("companyName", ""),
                    "duration": first_exp.get("duration", ""),
                    "description": first_exp.get("description", ""),
                    "location": first_exp.get("location", ""),
                    "title": first_exp.get("title", "")
                }
            else:
                transformed_person["currentWork"] = {
                    "companyName": "",
                    "duration": "",
                    "description": "",
                    "location": "",
                    "title": ""
                }
            reasoning_transform_people.append(transformed_person)
        except Exception as e:
            logger.error(f"###LOGS: Error processing person {person_id}: {str(e)}")
            traceback.print_exc()
            continue
    logger.info(f"###LOGS: Transformed {len(reasoning_transform_people)} people for reasoning")
    process_time = round(time.time() - start_time, 2)
    logger.info(f"###LOGS: Initial processing completed in {process_time} seconds")
    
    # ========== TEMPORARY DEBUG CODE - REMOVE AFTER TESTING ==========
    # # Save transformed people data with enhanced fields
    # transformed_debug_data = {
    #     "timestamp": debug_timestamp,
    #     "total_transformed": len(reasoning_transform_people),
    #     "additional_fields_fetched": additional_fields,
    #     "db_queries_context": db_queries,
    #     "sample_transformed_people": reasoning_transform_people[:3],  # First 3 people for validation
    #     "enhanced_fields_summary": {}
    # }
    # 
    # # Count how many people have each enhanced field
    # for field in ["education", "accomplishments", "dbQueryContext"]:
    #     count = sum(1 for person in reasoning_transform_people if field in person)
    #     transformed_debug_data["enhanced_fields_summary"][field] = count
    # 
    # with open(f"{debug_folder}/transformed_people_{debug_timestamp}.json", 'w') as f:
    #     json.dump(transformed_debug_data, f, indent=2, default=str)
    # logger.info(f"###DEBUG: Saved transformed people data to {debug_folder}/transformed_people_{debug_timestamp}.json")
    # ================================================================
    def enhance_results(base_results):
        enhanced = []
        avatar_urls = []
        for person in base_results:
            pid = person["personId"]
            doc = mongo_docs.get(pid)
            if doc and doc.get("avatarURL"):
                avatar_urls.append(doc["avatarURL"])
        if avatar_urls:
            global _avatar_url_batch
            _avatar_url_batch.extend(avatar_urls)
            process_avatar_urls_batch()
        for person in base_results:
            pid = person["personId"]
            doc = mongo_docs.get(pid)
            if doc:
                work_exp = doc.get("workExperience", [])
                if work_exp:
                    first_exp = work_exp[0]
                    current_work = {
                        "companyName": first_exp.get("companyName", ""),
                        "duration": first_exp.get("duration", ""),
                        "description": first_exp.get("description", ""),
                        "location": first_exp.get("location", ""),
                        "title": first_exp.get("title", "")
                    }
                else:
                    current_work = {
                        "companyName": "",
                        "duration": "",
                        "description": "",
                        "location": "",
                        "title": ""
                    }
                enhanced_person = {
                    **person,
                    "type": "person",
                    "stage": doc.get("stage", ""),
                    "currentLocation": doc.get("currentLocation", ""),
                    "connectionLevel": doc.get("connectionLevel", ""),
                    "linkedinUsername": doc.get("linkedinUsername", ""),
                    "linkedinHeadline": doc.get("linkedinHeadline", ""),
                    "contacts": doc.get("contacts", {}),
                    "currentWork": current_work,
                    "avatarURL": _avatar_url_results.get(doc.get("avatarURL", ""), doc.get("avatarURL", "")),
                    "mutuals": process_mutuals(doc.get("mutual", []) or doc.get("contacts", {}).get("mutuals", []))
                }
                # Convert ObjectIds to strings for JSON serialization
                enhanced_person = convert_objectids_to_strings(enhanced_person)
                enhanced.append(enhanced_person)
        return enhanced
    # Ranking removed - now handled in Reasoning Lambda
    # Simply return enhanced candidates for ranking in the next stage
    logger.info(f"###LOGS: Returning {len(reasoning_transform_people)} candidates for ranking in Reasoning Lambda")
    return enhance_results(reasoning_transform_people)

# ------------------------------------------------------------------------------
# Helpers to handle alternative "union" logic (unchanged)
# ------------------------------------------------------------------------------
def union_people_dicts(list_of_dicts: list) -> dict:
    """
    Given multiple dicts of { personId -> {...} }, produce the union.
    For each person that appears in multiple dicts:
    - Merge their skill names and descriptions into unique lists
    - Keep the lowest distance score if present
    """
    combined = {}
    duplicate_count = 0
    total_entries = sum(len(d) for d in list_of_dicts)
    for d in list_of_dicts:
        for pid, val in d.items():
            if pid not in combined:
                combined[pid] = val.copy()
                if "skillName" in val and not isinstance(val["skillName"], list):
                    combined[pid]["skillName"] = [val["skillName"]]
                if "skillDescription" in val and not isinstance(val["skillDescription"], list):
                    combined[pid]["skillDescription"] = [val["skillDescription"]]
            else:
                duplicate_count += 1
                existing = combined[pid]
                existing_dist = existing.get("distance", float('inf'))
                new_dist = val.get("distance", float('inf'))
                if new_dist < existing_dist:
                    existing["distance"] = new_dist
                if "skillName" in val:
                    skill_names = existing.get("skillName", [])
                    new_skill = val["skillName"]
                    if isinstance(new_skill, list):
                        skill_names.extend(new_skill)
                    else:
                        skill_names.append(new_skill)
                    existing["skillName"] = list(set(skill_names))
                if "skillDescription" in val:
                    skill_descs = existing.get("skillDescription", [])
                    new_desc = val["skillDescription"]
                    if isinstance(new_desc, list):
                        skill_descs.extend(new_desc)
                    else:
                        skill_descs.append(new_desc)
                    existing["skillDescription"] = list(set(skill_descs))
    logger.info(f"Union stats: Total entries: {total_entries}, Unique entries: {len(combined)}, Duplicates merged: {duplicate_count}")
    return combined

# ------------------------------------------------------------------------------
# Regex-based location search helper
# ------------------------------------------------------------------------------

def getShortListedPeopleForLocationRegex(locationObj: dict,
                                         userid: str,
                                         shortListedPeople=None,
                                         max_results: int = search_limits.get_max_results_location()) -> dict:
    """
    Perform a plain-text / regex search on `currentLocation` (city, state or
    country) using the location name and its alt_names list supplied by Hyde.

    Returns the same schema used by the vector-search path so that downstream
    merge logic works unchanged. A constant similarity score of ``1.0`` is used
    for all regex matches.
    """
    try:
        loc_name = locationObj.get("name", "").strip()
        alt_names = locationObj.get("alt_names", [])
        # Combine primary and alternate names, deduplicated & stripped
        keywords = list({k.strip() for k in ([loc_name] + alt_names) if k})

        if not keywords:
            logger.info("Regex location search skipped – no keywords provided")
            return {}

        # Build a single "OR" regex for all keywords (case-insensitive)
        escaped_keywords = [re.escape(k) for k in keywords]
        regex_pattern = rf"(?:{'|'.join(escaped_keywords)})"

        # Build base match with user filter
        base_match = {
            "userId": ObjectId(userid),
            "currentLocation": {"$regex": regex_pattern, "$options": "i"}
        }
        
        # OPTIMIZATION: Pre-filter at database level if shortListedPeople is provided
        if shortListedPeople:
            shortlisted_ids = [ObjectId(pid) for pid in shortListedPeople.keys()]
            base_match["_id"] = {"$in": shortlisted_ids}
            logger.info(f"Pre-filtering location search to {len(shortlisted_ids)} shortlisted people")

        pipeline = [
            {"$match": base_match},
            {"$project": {
                "_id": 1,
                "userId": 1,
                "name": 1,
                "currentLocation": 1
            }},
            {"$limit": max_results}
        ]

        docs = list(mongoCollectionNodes.aggregate(pipeline))
        logger.info(
            f"Regex location search for '{loc_name}' (+{len(alt_names)} alt) matched {len(docs)} records")

        people = {}
        filtered_count = 0
        for doc in docs:
            pid = str(doc["_id"])
            # This check is now redundant if we pre-filtered in MongoDB, but keep for safety
            if shortListedPeople and pid not in shortListedPeople:
                filtered_count += 1
                logger.warning(f"Person {pid} found in DB but not in shortlist (should not happen with optimization)")
                continue

            res = {
                "personId": pid,
                "userId": str(doc["userId"]),
                "locationName": loc_name,
                "locationDescription": doc.get("currentLocation", ""),
                "similarity": 1.0,  # fixed score for regex hit
            }

            # Merge any already-known fields from shortListedPeople
            if shortListedPeople and pid in shortListedPeople:
                for k in [
                    "skillDescription",
                    "skillName",
                ]:
                    if k in shortListedPeople[pid]:
                        res[k] = shortListedPeople[pid][k]

            people[pid] = res

        return people

    except Exception as e:
        logger.error(f"Error in getShortListedPeopleForLocationRegex: {e}")
        traceback.print_exc()
        return {}

# ------------------------------------------------------------------------------
# Search for skill using Upstash (updated)
# ------------------------------------------------------------------------------


def getShortListedPeopleForSkill(skillObj: dict,
                                 userid: str,
                                 shortListedPeople=None,
                                 similarity_threshold=0.80,
                                 max_results=search_limits.get_max_results_skill()) -> dict:
    """
    Updated to combine vector search, regex search, and title keyword search.
    Returns union of all matching methods when multiple are available.
    """
    try:
        MAX_RESULTS = max_results
        skill_name = skillObj.get("name", "Unknown")
        
        # Initialize result collectors for each method
        title_results = {}
        vector_results = {}
        regex_results = {}
        
        # Method 1: Title keyword search
        title_keywords = skillObj.get("titleKeywords", [])
        if title_keywords:
            logger.info(f"Using title keyword search for skill '{skill_name}' with keywords: {title_keywords}")
            temporal = skillObj.get("temporal", "any")
            title_results = getShortListedPeopleForTitleKeywords(
                title_keywords,
                userid,
                temporal,
                shortListedPeople,
                max_results
            )
            logger.info(f"Title keyword search returned {len(title_results)} results")
        
        # Method 2: Vector search if embeddings are available
        skill_embeddings = skillObj.get("embeddings", [])
        if skill_embeddings:
            logger.info(f"Running vector search for skill '{skill_name}'")
            description_text = skillObj.get("description", "")
            
            try:
                upstash_result = upstash_client.query(
                    vector=skill_embeddings,
                    top_k=MAX_RESULTS,
                    filter=f"userId = '{userid}'",
                    include_metadata=True,
                    include_data=True,
                    include_vectors=False,
                    namespace="skills"
                )
                
                similarity_filtered = 0
                filtered_count = 0
                
                for res in upstash_result:
                    similarity = res.score
                    if similarity < similarity_threshold:
                        similarity_filtered += 1
                        continue
                        
                    metadata = res.metadata
                    data = res.data
                    person_id = metadata["personId"]
                    
                    if shortListedPeople and person_id not in shortListedPeople:
                        filtered_count += 1
                        continue
                        
                    if person_id in vector_results and vector_results[person_id].get("similarity", 0.0) >= similarity:
                        continue
                        
                    result_dict = {
                        "personId": person_id,
                        "userId": metadata["userId"],
                        "skillName": metadata["skillName"],
                        "skillDescription": data,
                        "similarity": similarity,
                        "vectorMatch": True
                    }
                    
                    if shortListedPeople and person_id in shortListedPeople:
                        for key in ["locationDescription"]:
                            if key in shortListedPeople[person_id]:
                                result_dict[key] = shortListedPeople[person_id][key]
                    
                    vector_results[person_id] = result_dict
                
                logger.info(
                    f"Vector search completed. Similarity filtered: {similarity_filtered}, "
                    f"Shortlist filtered: {filtered_count}, Matches: {len(vector_results)}"
                )
                
            except Exception as e:
                logger.error(f"Error in vector search for skill '{skill_name}': {str(e)}")
        
        # Method 3: Regex search if patterns are available
        regex_patterns = skillObj.get("regexPatterns", {})
        if regex_patterns and regex_patterns.get("keywords"):
            logger.info(f"Running regex search for skill '{skill_name}'")
            regex_results = getShortListedPeopleForSkillRegex(
                skillObj,
                userid,
                shortListedPeople,
                max_results
            )
            logger.info(f"Regex search returned {len(regex_results)} results")
        
        # Combine all results using union
        all_result_dicts = []
        if title_results:
            all_result_dicts.append(title_results)
        if vector_results:
            all_result_dicts.append(vector_results)
        if regex_results:
            all_result_dicts.append(regex_results)
        
        if not all_result_dicts:
            logger.info(f"No results found for skill '{skill_name}' using any method")
            return {}
        
        # Use union_people_dicts to combine results
        combined_results = union_people_dicts(all_result_dicts)
        
        # Sort by similarity (vector matches will have higher similarity than regex/title)
        sorted_results = dict(sorted(
            combined_results.items(), 
            key=lambda x: x[1].get('similarity', 0.5), 
            reverse=True
        ))
        
        logger.info(
            f"Skill search completed for '{skill_name}'. "
            f"Title matches: {len(title_results)}, Vector matches: {len(vector_results)}, "
            f"Regex matches: {len(regex_results)}, Total unique: {len(sorted_results)}"
        )
        
        return sorted_results
        
    except Exception as e:
        logger.error(f"Error in getShortListedPeopleForSkill: {str(e)}")
        traceback.print_exc()
        return {}
def getShortListedPeopleForSkillAlternative(skillObj: dict,
                                            userid: str,
                                            shortListedPeople=None,
                                            similarity_threshold=0.80,
                                            max_results=search_limits.get_max_results_skill()) -> dict:
    """
    For alternative_skills=True: 
    Uses hybrid matching combining all available methods (embeddings, titleKeywords, regexPatterns).
    The parent skill and each related role can use any combination of matching methods.
    We do sub-search for each item that has at least one matching method, then union the results.
    Updated to convert distance to similarity within sub-search functions.
    """
    try:
        # Collect all items from (parent + relatedRoles) 
        items = []
        parent_name = skillObj.get("name", "Unknown")
        
        # Create a synthetic skill object for the parent that includes all matching methods
        parent_item = {
            "name": parent_name,
            "description": skillObj.get("description", ""),
            "embeddings": skillObj.get("embeddings", []) if skillObj.get("embeddings") else [],
            "titleKeywords": skillObj.get("titleKeywords", []),
            "regexPatterns": skillObj.get("regexPatterns", {}),
            "temporal": skillObj.get("temporal", "any")
        }
        items.append(parent_item)
        
        # Add related roles (they typically don't have titleKeywords or regexPatterns)
        related_roles = skillObj.get("relatedRoles", [])
        for rr in related_roles:
            rr_name = rr.get("name", "UnknownRole")
            if rr.get("embeddings"):
                items.append({
                    "name": rr_name,
                    "description": rr.get("description", ""),
                    "embeddings": rr["embeddings"],
                    "titleKeywords": rr.get("titleKeywords", []),
                    "regexPatterns": rr.get("regexPatterns", {})
                })
        
        # Filter items that have at least one matching method (embeddings, titleKeywords, or regexPatterns)
        valid_items = []
        for item in items:
            has_embeddings = item.get("embeddings") and len(item["embeddings"]) > 0
            has_title_keywords = item.get("titleKeywords") and len(item["titleKeywords"]) > 0
            has_regex_patterns = item.get("regexPatterns") and item["regexPatterns"].get("keywords")
            
            if has_embeddings or has_title_keywords or has_regex_patterns:
                valid_items.append(item)
        
        if not valid_items:
            logger.info(f"No valid matching methods available for skill '{parent_name}' or related roles.")
            return {}
        
        items = valid_items
        results_list = []
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=len(items)) as executor:
            future_to_item = {executor.submit(
                    getShortListedPeopleForSkill,
                    itm,
                    userid,
                    shortListedPeople,
                    similarity_threshold,
                    max_results
                ): itm["name"] for itm in items}
            for future in as_completed(future_to_item):
                sub_name = future_to_item[future]
                try:
                    subres = future.result()
                    results_list.append(subres)
                    logger.info(f"Alt sub-search for '{sub_name}' returned {len(subres)} results.")
                except Exception as e:
                    logger.error(f"Error in alt sub-search for '{sub_name}': {e}")
        union_dict = union_people_dicts(results_list)
        logger.info(f"Skill search for '{parent_name}' => final union: {len(union_dict)} people")
        return union_dict
    except Exception as e:
        logger.error(f"Error in getShortListedPeopleForSkillAlternative: {str(e)}")
        traceback.print_exc()
        return {}

# ------------------------------------------------------------------------------
# Search for title keywords (NEW)
# ------------------------------------------------------------------------------
def getShortListedPeopleForTitleKeywords(titleKeywords: List[str],
                                        userid: str,
                                        temporal: str = "any",
                                        shortListedPeople=None,
                                        max_results: int = search_limits.get_max_results_title()) -> dict:
    """
    Search for people based on job title keywords using regex with temporal awareness.
    Searches in both linkedinHeadline and workExperience.title fields.
    
    Args:
        titleKeywords: List of title keywords to search for
        userid: User ID for filtering
        temporal: "current" | "past" | "any"
        shortListedPeople: Optional dict to filter results
        max_results: Maximum number of results to return
    
    Returns:
        Dictionary of person_id -> person data
    """
    try:
        if not titleKeywords:
            logger.info("No title keywords provided")
            return {}
            
        logger.info(f"Running title keyword search for: {titleKeywords}, temporal: {temporal}")
        start_time = time.time()
        
        # Build improved regex pattern for title keywords
        regex_pattern = build_improved_regex_pattern(titleKeywords)
        
        # If we have a shortlist, optimize by only searching within those IDs
        match_base = {"userId": ObjectId(userid)}
        
        if shortListedPeople:
            # Convert shortlisted people IDs to ObjectIds for MongoDB query
            shortlisted_ids = [ObjectId(pid) for pid in shortListedPeople.keys()]
            match_base["_id"] = {"$in": shortlisted_ids}
            logger.info(f"Optimizing search to only look within {len(shortlisted_ids)} shortlisted people")
        
        # Build pipeline based on temporal requirement
        if temporal == "current":
            # For current, search in linkedinHeadline and first work experience
            match_base["$or"] = [
                {
                    "linkedinHeadline": {
                        "$regex": regex_pattern,
                        "$options": "i"
                    }
                },
                {
                    "about": {
                        "$regex": regex_pattern,
                        "$options": "i"
                    }
                },
                {
                    "workExperience.0.title": {
                        "$regex": regex_pattern,
                        "$options": "i"
                    }
                }
            ]
            pipeline = [
                {"$match": match_base},
                {
                    "$project": {
                        "_id": 1,
                        "userId": 1,
                        "name": 1,
                        "linkedinHeadline": 1,
                        "workExperience": 1,
                        "currentLocation": 1
                    }
                },
                {"$limit": max_results}
            ]
            
        elif temporal == "past":
            # For past positions, need to unwind and check non-current positions
            pipeline = [
                {"$match": match_base},
                {"$unwind": {"path": "$workExperience", "includeArrayIndex": "workIndex"}},
                {"$match": {
                    "workIndex": {"$gt": 0},  # Skip first position
                    "workExperience.title": {"$regex": regex_pattern, "$options": "i"}
                }},
                {"$group": {
                    "_id": "$_id",
                    "userId": {"$first": "$userId"},
                    "name": {"$first": "$name"},
                    "linkedinHeadline": {"$first": "$linkedinHeadline"},
                    "currentLocation": {"$first": "$currentLocation"},
                    "matchedWorkExperience": {"$push": "$workExperience"},
                    "workExperience": {"$first": "$workExperience"}
                }},
                {"$limit": max_results}
            ]
            
        else:  # temporal == "any"
            # Search everywhere
            match_base["$or"] = [
                {
                    "linkedinHeadline": {
                        "$regex": regex_pattern,
                        "$options": "i"
                    }
                },
                {
                    "workExperience.title": {
                        "$regex": regex_pattern,
                        "$options": "i"
                    }
                }
            ]
            pipeline = [
                {"$match": match_base},
                {
                    "$project": {
                        "_id": 1,
                        "userId": 1,
                        "name": 1,
                        "linkedinHeadline": 1,
                        "workExperience": 1,
                        "currentLocation": 1
                    }
                },
                {"$limit": max_results}
            ]
        
        # Add debugging aggregation to understand what's being matched
        if logger.isEnabledFor(logging.DEBUG):
            debug_pipeline = pipeline[:-1] + [{"$count": "total"}]
            count_result = list(mongoCollectionNodes.aggregate(debug_pipeline))
            total_matches = count_result[0]["total"] if count_result else 0
            logger.debug(f"Total documents matching regex before limit: {total_matches}")
        
        docs = list(mongoCollectionNodes.aggregate(pipeline))
        logger.info(f"Title keyword search matched {len(docs)} records for temporal={temporal}")
        
        # Additional debugging: log a few examples of what was found
        if docs and logger.isEnabledFor(logging.DEBUG):
            for i, doc in enumerate(docs[:3]):
                logger.debug(f"Example match {i+1}: {doc.get('name')} - Title: {doc.get('workExperience', [{}])[0].get('title', 'N/A')}")
        
        people = {}
        filtered_count = 0
        
        for doc in docs:
            person_id = str(doc["_id"])
            
            # This check is now redundant if we pre-filtered in MongoDB
            if shortListedPeople and person_id not in shortListedPeople:
                filtered_count += 1
                logger.warning(f"Person {person_id} ({doc.get('name')}) found in DB but not in shortlist")
                continue
            
            # Extract current title from workExperience or headline
            current_title = ""
            if doc.get("workExperience") and len(doc["workExperience"]) > 0:
                current_title = doc["workExperience"][0].get("title", "")
            if not current_title:
                current_title = doc.get("linkedinHeadline", "")
            
            # Log matches for debugging
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Matched: {doc.get('name')} with title: {current_title}")
            
            result_dict = {
                "personId": person_id,
                "userId": str(doc["userId"]),
                "name": doc.get("name", ""),
                "currentTitle": current_title,
                "linkedinHeadline": doc.get("linkedinHeadline", ""),
                "titleMatch": True,
                "temporalMatch": temporal
            }
            
            # For past searches, add matched titles
            if temporal == "past" and "matchedWorkExperience" in doc:
                matched_titles = [exp.get("title", "") for exp in doc["matchedWorkExperience"]]
                result_dict["matchedTitles"] = list(set(matched_titles))
            
            # Include existing fields from shortListedPeople if available
            if shortListedPeople and person_id in shortListedPeople:
                for key in ["skillDescription", "skillName", "locationDescription"]:
                    if key in shortListedPeople[person_id]:
                        result_dict[key] = shortListedPeople[person_id][key]
            
            people[person_id] = result_dict
        
        total_time = time.time() - start_time
        logger.info(
            f"Title keyword search completed in {total_time:.2f}s. "
            f"Temporal: {temporal}, Shortlist filtered: {filtered_count}, Final matches: {len(people)}"
        )
        
        return people
        
    except Exception as e:
        logger.error(f"Error in getShortListedPeopleForTitleKeywords: {str(e)}")
        traceback.print_exc()
        return {}

# Add this new function after getShortListedPeopleForTitleKeywords
def getShortListedPeopleForSkillRegex(skillObj: dict,
                                     userid: str,
                                     shortListedPeople=None,
                                     max_results: int = search_limits.get_max_results_skill()) -> dict:
    """
    Search for people based on skill regex patterns.
    Searches in workExperience.description, education.description, bio, and linkedinHeadline.
    
    Args:
        skillObj: Skill object with regexPatterns field
        userid: User ID for filtering
        shortListedPeople: Optional dict to filter results
        max_results: Maximum number of results to return
    
    Returns:
        Dictionary of person_id -> person data
    """
    try:
        regex_patterns = skillObj.get("regexPatterns", {})
        keywords = regex_patterns.get("keywords", [])
        fields = regex_patterns.get("fields", [])
        
        if not keywords or not fields:
            logger.info(f"No regex patterns for skill '{skillObj.get('name')}'")
            return {}
            
        skill_name = skillObj.get("name", "Unknown")
        logger.info(f"Running regex search for skill '{skill_name}' with keywords: {keywords}")
        start_time = time.time()
        
        # Build improved regex pattern for skill keywords
        regex_pattern = build_improved_regex_pattern(keywords)
        
        # Build base match with user filter
        base_match = {"userId": ObjectId(userid)}
        
        # OPTIMIZATION: Pre-filter at database level if shortListedPeople is provided
        if shortListedPeople:
            shortlisted_ids = [ObjectId(pid) for pid in shortListedPeople.keys()]
            base_match["_id"] = {"$in": shortlisted_ids}
            logger.info(f"Pre-filtering skill regex search to {len(shortlisted_ids)} shortlisted people")
        
        # Build OR conditions for different fields
        or_conditions = []
        for field in fields:
            if field == "workExperience.description":
                or_conditions.append({
                    "workExperience.description": {
                        "$regex": regex_pattern,
                        "$options": "i"
                    }
                })
            elif field == "education.description":
                or_conditions.append({
                    "education.description": {
                        "$regex": regex_pattern,
                        "$options": "i"
                    }
                })
            elif field in ["bio", "linkedinHeadline"]:
                or_conditions.append({
                    field: {
                        "$regex": regex_pattern,
                        "$options": "i"
                    }
                })
        
        if not or_conditions:
            return {}
        
        # Combine base match with search criteria
        base_match["$or"] = or_conditions
        
        pipeline = [
            {"$match": base_match},
            {
                "$project": {
                    "_id": 1,
                    "userId": 1,
                    "name": 1,
                    "bio": 1,
                    "linkedinHeadline": 1,
                    "workExperience": 1,
                    "education": 1,
                    "currentLocation": 1
                }
            },
            {"$limit": max_results}
        ]
        
        docs = list(mongoCollectionNodes.aggregate(pipeline))
        logger.info(f"Skill regex search matched {len(docs)} records")
        
        people = {}
        filtered_count = 0
        
        for doc in docs:
            person_id = str(doc["_id"])
            
            # This check is now redundant if we pre-filtered in MongoDB, but keep for safety
            if shortListedPeople and person_id not in shortListedPeople:
                filtered_count += 1
                logger.warning(f"Person {person_id} found in DB but not in shortlist (should not happen with optimization)")
                continue
            
            # Build context for display without re-validating MongoDB's regex matches
            # MongoDB already found this person, so we trust its results and just build context
            matched_contexts = []
            
            # Build context from available work experience
            if doc.get("workExperience"):
                for i, exp in enumerate(doc["workExperience"][:3]):  # Only check first 3 for context
                    title = exp.get("title", "")
                    company = exp.get("companyName", "")
                    if title or company:
                        matched_contexts.append(f"Work: {title} at {company}")
            
            # Build context from education if no work context found
            if not matched_contexts and doc.get("education"):
                for i, edu in enumerate(doc["education"][:2]):  # Only check first 2 for context
                    degree = edu.get("degree", "")
                    school = edu.get("school", "")
                    if degree or school:
                        matched_contexts.append(f"Education: {degree} at {school}")
            
            # Add bio/headline context if no other context found
            if not matched_contexts:
                if doc.get("bio"):
                    matched_contexts.append("Bio match")
                elif doc.get("linkedinHeadline"):
                    matched_contexts.append("Headline match")
            
            result_dict = {
                "personId": person_id,
                "userId": str(doc["userId"]),
                "name": doc.get("name", ""),
                "skillName": skill_name,
                "skillDescription": f"Regex match: {', '.join(matched_contexts[:3])}",  # Include context
                "regexMatch": True,  # Flag to indicate this came from regex search
                "similarity": 0.75  # Fixed similarity score for regex matches
            }
            
            # Include existing fields from shortListedPeople if available
            if shortListedPeople and person_id in shortListedPeople:
                for key in ["locationDescription"]:
                    if key in shortListedPeople[person_id]:
                        result_dict[key] = shortListedPeople[person_id][key]
            
            people[person_id] = result_dict
        
        total_time = time.time() - start_time
        logger.info(
            f"Skill regex search completed in {total_time:.2f}s. "
            f"Shortlist filtered: {filtered_count}, Final matches: {len(people)}"
        )
        
        return people
        
    except Exception as e:
        logger.error(f"Error in getShortListedPeopleForSkillRegex: {str(e)}")
        traceback.print_exc()
        return {}


# ------------------------------------------------------------------------------
# Utility functions for temporal awareness
# ------------------------------------------------------------------------------
def is_current_position(duration: str) -> bool:
    """
    Determine if a duration string indicates a current position.
    Examples:
    - "Jan 2020 - Present" -> True
    - "Jan 2020 - Dec 2023" -> False
    """
    if not duration:
        return False
    
    current_indicators = ["present", "current", "now", "ongoing"]
    duration_lower = duration.lower()
    
    return any(indicator in duration_lower for indicator in current_indicators)



def build_improved_regex_pattern(keywords: List[str]) -> str:
    """
    Build an improved regex pattern from keywords that:
    1. Sorts keywords by length (longest first) to avoid partial matches
    2. Handles hyphenated and multi-word terms with flexible boundaries
    3. Uses proper word boundaries for single words
    4. Prevents false positives while maximizing true positives
    5. Handles case variations and common separators
    
    Args:
        keywords: List of keywords to match
        
    Returns:
        Compiled regex pattern string
    """
    if not keywords:
        return ""
    
    # Sort keywords by length (longest first) to avoid partial matches
    # This ensures "co-founder" is matched before "founder"
    sorted_keywords = sorted(keywords, key=len, reverse=True)
    
    # Build regex pattern with improved matching
    patterns = []
    for keyword in sorted_keywords:
        # Escape special regex characters
        escaped_keyword = re.escape(keyword)
        
        # Handle different types of keywords
        if '-' in keyword:
            # For hyphenated words like "co-founder"
            # Allow for variations: Co-founder, CO-FOUNDER, CoFounder, etc.
            # Also handle cases where hyphen might be replaced with space
            base_parts = keyword.split('-')
            
            # Create pattern for hyphenated version
            hyphen_pattern = r'\-'.join(re.escape(part) for part in base_parts)
            
            # Create pattern for space-separated version
            space_pattern = r'\s+'.join(re.escape(part) for part in base_parts)
            
            # Create pattern for concatenated version (no separator)
            concat_pattern = ''.join(re.escape(part) for part in base_parts)
            
            # Combine all variations with proper boundaries
            combined_pattern = f"(?:^|\\b|\\s)(?:{hyphen_pattern}|{space_pattern}|{concat_pattern})(?=\\b|\\s|$|[^a-zA-Z0-9])"
            patterns.append(combined_pattern)
            
        elif ' ' in keyword:
            # For multi-word terms like "chief executive"
            # Allow for variable whitespace between words
            parts = keyword.split()
            flexible_pattern = r'\s+'.join(re.escape(part) for part in parts)
            patterns.append(f"(?:^|\\b|\\s){flexible_pattern}(?=\\b|\\s|$|[^a-zA-Z0-9])")
            
        else:
            # For single words, use standard word boundaries
            # But also handle cases where the word might be part of a compound
            patterns.append(f"\\b{escaped_keyword}\\b")
    
    # Combine all patterns with OR
    return f"(?:{'|'.join(patterns)})"


def extract_position_dates(duration: str) -> tuple:
    """
    Extract start and end dates from duration string.
    Returns (start_date, end_date, is_current)
    
    Examples:
    - "Jan 2020 - Present" -> ("Jan 2020", "Present", True)
    - "Jan 2020 - Dec 2023" -> ("Jan 2020", "Dec 2023", False)
    """
    if not duration:
        return (None, None, False)
    
    # Common patterns for duration strings
    # Pattern 1: "Start - End" or "Start – End" (em dash)
    import re
    pattern = r'^(.*?)\s*[-–]\s*(.*)$'
    match = re.match(pattern, duration.strip())
    
    if match:
        start_date = match.group(1).strip()
        end_date = match.group(2).strip()
        is_current = is_current_position(end_date)
        return (start_date, end_date, is_current)
    
    # If no match, return the whole string as start date
    return (duration.strip(), None, False)

# ------------------------------------------------------------------------------
# Search for organisation (unchanged)
# ------------------------------------------------------------------------------
def getShortListedPeopleForOrganisation(organisationObj: dict,  # ← Changed parameter
                                        userid: str,
                                        shortListedPeople=None) -> dict:
    """
    Search for people based on organization.
    Now accepts organisation object instead of keywords list.
    """
    try:
        # Extract keywords and temporal from the object
        org_name = organisationObj.get("name", "Unknown")
        aliases = organisationObj.get("aliases", [])
        organisationKeywords = [org_name] + aliases
        temporal = organisationObj.get("temporal", "any")  # ← NEW
        
        MAX_RESULTS = search_limits.get_max_results_org()
        logger.info(f"Running organization search for '{org_name}', keywords: {organisationKeywords}, temporal: {temporal}")
        start_time = time.time()
        
        # Build improved regex pattern for organization keywords
        regex_pattern = build_improved_regex_pattern(organisationKeywords)
        
        # Build base match with user filter
        base_match = {"userId": ObjectId(userid)}
        
        # OPTIMIZATION: Pre-filter at database level if shortListedPeople is provided
        if shortListedPeople:
            shortlisted_ids = [ObjectId(pid) for pid in shortListedPeople.keys()]
            base_match["_id"] = {"$in": shortlisted_ids}
            logger.info(f"Pre-filtering organization search to {len(shortlisted_ids)} shortlisted people")

        # Build aggregation pipeline based on temporal requirement
        if temporal == "current":
            # Combine base match with current organization criteria
            current_match = base_match.copy()
            current_match["$or"] = [
                # Check first work experience
                {"workExperience.0.companyName": {"$regex": regex_pattern, "$options": "i"}},
                # Check any position with "Present" in duration
                {"workExperience": {
                    "$elemMatch": {
                        "companyName": {"$regex": regex_pattern, "$options": "i"},
                        "duration": {"$regex": "Present|present|Current|current", "$options": "i"}
                    }
                }}
            ]
            
            pipeline = [
                {"$match": current_match},
                {"$project": {
                    "_id": 1,
                    "userId": 1,
                    "name": 1,
                    "workExperience": 1
                }},
                {"$limit": MAX_RESULTS}
            ]
        elif temporal == "past":
            # For past positions, we need to use $unwind and filter
            pipeline = [
                {"$match": base_match},
                {"$unwind": {"path": "$workExperience", "includeArrayIndex": "workIndex"}},
                {"$match": {
                    "workExperience.companyName": {"$regex": regex_pattern, "$options": "i"},
                    "$and": [
                        {"workIndex": {"$gt": 0}},  # Not the first position
                        {"workExperience.duration": {"$not": {"$regex": "Present|present|Current|current", "$options": "i"}}}
                    ]
                }},
                {"$group": {
                    "_id": "$_id",
                    "userId": {"$first": "$userId"},
                    "name": {"$first": "$name"},
                    "matchedWorkExperience": {"$push": "$workExperience"}
                }},
                {"$limit": MAX_RESULTS}
            ]
        else:  # temporal == "any"
            # Combine base match with any organization criteria
            any_match = base_match.copy()
            any_match["$or"] = [
                {
                    "organizations.orgName": {
                        "$regex": regex_pattern,
                        "$options": "i"
                    }
                },
                {
                    "organizations.orgSynonyms": {
                        "$regex": regex_pattern,
                        "$options": "i"
                    }
                },
                {
                    "workExperience.companyName": {
                        "$regex": regex_pattern,
                        "$options": "i"
                    }
                }
            ]
            
            pipeline = [
                {"$match": any_match},
                {
                    "$project": {
                        "_id": 1,
                        "userId": 1,
                        "name": 1
                    }
                },
                {"$limit": MAX_RESULTS}
            ]
        
        result_org = list(mongoCollectionNodes.aggregate(pipeline))
        logger.info(f"Found {len(result_org)} matches for temporal={temporal}")
        
        people = {}
        filtered_count = 0
        
        for doc in result_org:
            person_id = str(doc["_id"])
            # This check is now redundant if we pre-filtered in MongoDB, but keep for safety
            if shortListedPeople and person_id not in shortListedPeople:
                filtered_count += 1
                logger.warning(f"Person {person_id} found in DB but not in shortlist (should not happen with optimization)")
                continue
                
            result_dict = {
                "personId": person_id,
                "userId": str(doc["userId"]),
                "name": doc["name"],
                "temporalMatch": temporal  # Add temporal match type
            }
            
            # For temporal searches, add matched organization info
            if temporal in ["current", "past"] and "matchedWorkExperience" in doc:
                matched_orgs = [exp.get("companyName", "") for exp in doc["matchedWorkExperience"]]
                result_dict["matchedOrganizations"] = list(set(matched_orgs))
            elif temporal in ["current", "past"] and "workExperience" in doc:
                # MongoDB already found this person with matching organizations
                # Build context from work experience without re-validating the regex
                matched_orgs = []
                work_experiences = doc.get("workExperience", [])
                
                # Apply only temporal filtering since MongoDB already did organization matching
                for idx, exp in enumerate(work_experiences):
                    company_name = exp.get("companyName", "")
                    if not company_name:
                        continue
                        
                    # Apply temporal filtering
                    include_experience = False
                    if temporal == "current":
                        if idx == 0 or is_current_position(exp.get("duration", "")):
                            include_experience = True
                    elif temporal == "past":
                        if idx > 0 or not is_current_position(exp.get("duration", "")):
                            include_experience = True
                    else:  # temporal == "any"
                        include_experience = True
                    
                    if include_experience:
                        matched_orgs.append(company_name)
                        if temporal != "any":  # For current/past, only take first match
                            break
                
                result_dict["matchedOrganizations"] = list(set(matched_orgs))
            
            for key, value in doc.items():
                if isinstance(value, datetime):
                    result_dict[key] = format_datetime(value)
                    
            if shortListedPeople and person_id in shortListedPeople:
                for k in ["locationDescription", "skillDescription", "skillName"]:
                    if k in shortListedPeople[person_id]:
                        result_dict[k] = shortListedPeople[person_id][k]
                        
            people[person_id] = result_dict
            
        sorted_people = dict(sorted(
            people.items(),
            key=lambda x: len(x[1].get("about") or ""),
            reverse=True
        ))
        
        total_time = time.time() - start_time
        logger.info(
            f"Organization search completed in {total_time:.2f}s. "
            f"Temporal: {temporal}, Shortlist filtered: {filtered_count}, Final matches: {len(sorted_people)}"
        )
        
        return sorted_people
        
    except Exception as e:
        logger.error(f"Error in getShortListedPeopleForOrganisation: {str(e)}")
        traceback.print_exc()
        return {}
    
    
def parse_company_size_range(size_str):
    """
    Parse company size string like "11-50 employees" into min/max integers.
    
    Returns:
        tuple: (min_size, max_size) or (None, None) if parsing fails
    
    Examples:
        "11-50 employees" -> (11, 50)
        "51-200 employees" -> (51, 200)
        "10,001+ employees" -> (10001, float('inf'))
        "1-10 employees" -> (1, 10)
        "5001-10,000 employees" -> (5001, 10000)
    """
    if not size_str or not isinstance(size_str, str):
        return None, None
    
    try:
        # Remove "employees" and extra spaces
        size_str = size_str.replace("employees", "").replace("employee", "").strip()
        
        # Handle "10,001+" format (unbounded upper range)
        if "+" in size_str:
            min_size = int(size_str.replace("+", "").replace(",", "").strip())
            return min_size, float('inf')
        
        # Handle "11-50" format (range)
        if "-" in size_str:
            parts = size_str.split("-")
            if len(parts) == 2:
                min_size = int(parts[0].replace(",", "").strip())
                max_size = int(parts[1].replace(",", "").strip())
                return min_size, max_size
        
        # Handle single number (treat as exact size)
        single_num = int(size_str.replace(",", "").strip())
        return single_num, single_num
        
    except (ValueError, AttributeError):
        return None, None


def check_size_overlap(company_range, search_range):
    """
    Check if company size range overlaps with search range.
    
    Args:
        company_range: tuple (min, max) from company data
        search_range: dict with 'min' and 'max' from search criteria
    
    Returns:
        bool: True if ranges overlap
    """
    comp_min, comp_max = company_range
    if comp_min is None or comp_max is None:
        return False
    
    search_min = search_range.get("min", 0)
    search_max = search_range.get("max", float('inf'))
    
    # Check for overlap: company_max >= search_min AND company_min <= search_max
    return comp_max >= search_min and comp_min <= search_max


def getShortListedPeopleForSector(sectorObj: dict,
                                  userid: str,
                                  shortListedPeople=None,
                                  max_results: int = search_limits.get_max_results_sector()) -> dict:
    """
    Search for people who worked in companies of specific sectors.
    Temporal context now comes from individual sectorObj.
    """
    try:
        keywords = sectorObj.get("keywords", [])
        sector_name = sectorObj.get("name", "Unknown")
        company_stage = sectorObj.get("companyStage", {})
        temporal = sectorObj.get("temporal", "any")  # ← NEW: Get from sectorObj
        
        # Check if we have any search criteria
        has_keywords = bool(keywords)
        has_size_constraint = company_stage.get("enabled", False)
        
        if not has_keywords and not has_size_constraint:
            logger.info(f"No search criteria for sector: {sector_name}")
            return {}
        
        logger.info(f"Sector search for '{sector_name}' - Keywords: {has_keywords}, Size constraint: {has_size_constraint}")
        
        if has_size_constraint:
            size_range = company_stage.get("sizeRange", {})
            logger.info(f"Size constraint: {size_range.get('min', 0)}-{size_range.get('max', 'unlimited')} employees")
        
        start_time = time.time()
        
        # Build base match with user filter
        base_match = {"userId": ObjectId(userid)}
        
        # OPTIMIZATION: Pre-filter if shortListedPeople provided
        if shortListedPeople:
            shortlisted_ids = [ObjectId(pid) for pid in shortListedPeople.keys()]
            base_match["_id"] = {"$in": shortlisted_ids}
            logger.info(f"Pre-filtering to {len(shortlisted_ids)} shortlisted people")
        
        # Build OR conditions list
        or_conditions = []
        
        # Add keyword conditions if keywords exist
        if has_keywords:
            regex_pattern = build_improved_regex_pattern(keywords)
            or_conditions.extend([
                {"workExperience.industry": {"$regex": regex_pattern, "$options": "i"}},
                {"workExperience.about": {"$regex": regex_pattern, "$options": "i"}},
                {"workExperience.specialties": {"$regex": regex_pattern, "$options": "i"}}
            ])
            logger.info(f"Added keyword matching for: {keywords[:3]}...")
        
        # Add size condition if enabled
        # For size, we just check if company_size field exists
        # Actual size filtering will be done in Python
        if has_size_constraint:
            or_conditions.append({
                "workExperience.company_size": {"$exists": True, "$ne": None, "$ne": ""}
            })
            logger.info(f"Added check for company_size field existence")
        
        # Apply conditions to base_match
        if or_conditions:
            base_match["$or"] = or_conditions
        else:
            logger.warning(f"No valid conditions for sector: {sector_name}")
            return {}
        
        # MongoDB aggregation pipeline
        pipeline = [
            {"$match": base_match},
            {
                "$project": {
                    "_id": 1,
                    "userId": 1,
                    "name": 1,
                    "workExperience": 1,
                    "currentLocation": 1
                }
            },
            {"$limit": max_results}
        ]
        
        # Execute query
        users = list(nodes_collection.aggregate(pipeline))
        logger.info(f"MongoDB returned {len(users)} matches for sector '{sector_name}'")
        
        # Process results with Python-based size filtering
        people = {}
        filtered_count = 0
        size_filtered_count = 0
        
        for user in users:
            person_id = str(user["_id"])
            
            # Safety check for shortlisted people
            if shortListedPeople and person_id not in shortListedPeople:
                filtered_count += 1
                continue
            
            # Extract matching context
            matching_companies = []
            match_types = set()
            matched_by_size = False
            matched_by_keyword = False
            work_experiences = user.get("workExperience", [])
            
            for idx, exp in enumerate(work_experiences):
                # Check temporal filter
                include_experience = False
                if temporal == "current":
                    if idx == 0 or is_current_position(exp.get("duration", "")):
                        include_experience = True
                elif temporal == "past":
                    if idx > 0 or not is_current_position(exp.get("duration", "")):
                        include_experience = True
                else:  # temporal == "any"
                    include_experience = True
                
                if not include_experience:
                    continue
                
                # Check if this experience matches our criteria
                exp_matches = False
                
                # Check keyword match
                if has_keywords:
                    # Check if any keyword field has content (already matched by MongoDB)
                    if any([exp.get("industry"), exp.get("about"), exp.get("specialties")]):
                        matched_by_keyword = True
                        exp_matches = True
                        match_types.add("keyword")
                
                # Check size match
                if has_size_constraint:
                    company_size_str = exp.get("company_size")
                    if company_size_str:
                        comp_min, comp_max = parse_company_size_range(company_size_str)
                        if comp_min is not None:
                            size_range = company_stage.get("sizeRange", {})
                            if check_size_overlap((comp_min, comp_max), size_range):
                                matched_by_size = True
                                exp_matches = True
                                match_types.add("size")
                
                # If this experience matches, add the company
                if exp_matches:
                    company_name = exp.get("companyName", "Unknown Company")
                    company_size_str = exp.get("company_size", "")
                    if company_size_str:
                        matching_companies.append(f"{company_name} ({company_size_str})")
                    else:
                        matching_companies.append(company_name)
                    
                    # For current/past, only take first match
                    if temporal != "any":
                        break
            
            # Filter out if size constraint exists but no size match found
            if has_size_constraint and not has_keywords:
                # Size-only search: must match size
                if not matched_by_size:
                    size_filtered_count += 1
                    continue
            elif has_size_constraint and has_keywords:
                # Both criteria: must match at least one (OR logic)
                if not matched_by_size and not matched_by_keyword:
                    size_filtered_count += 1
                    continue
            
            # Fallback if no matches but user was returned by MongoDB
            if not matching_companies and work_experiences:
                first_exp = work_experiences[0]
                company_name = first_exp.get("companyName", "Unknown Company")
                company_size_str = first_exp.get("company_size", "")
                if company_size_str:
                    matching_companies.append(f"{company_name} ({company_size_str})")
                else:
                    matching_companies.append(company_name)
            
            # Build result
            result_dict = {
                "personId": person_id,
                "userId": str(user["userId"]),
                "name": user.get("name", ""),
                "sectorName": sector_name,
                "sectorCompanies": list(set(matching_companies)),
                "matchTypes": list(match_types) if match_types else ["unknown"],
                "similarity": 0.85
            }
            
            # Preserve existing fields from shortListedPeople
            if shortListedPeople and person_id in shortListedPeople:
                for key in ["skillDescription", "skillName", "locationDescription"]:
                    if key in shortListedPeople[person_id]:
                        result_dict[key] = shortListedPeople[person_id][key]
            
            people[person_id] = result_dict
        
        # Log statistics
        if has_size_constraint:
            logger.info(f"Size filtering removed {size_filtered_count} people who didn't match size criteria")
        
        if has_size_constraint and people:
            size_matches = sum(1 for p in people.values() if "size" in p.get("matchTypes", []))
            keyword_matches = sum(1 for p in people.values() if "keyword" in p.get("matchTypes", []))
            both_matches = sum(1 for p in people.values() if len(p.get("matchTypes", [])) > 1)
            logger.info(f"Final matches - Size: {size_matches}, Keywords: {keyword_matches}, Both: {both_matches}")
        
        # Final logging
        total_time = time.time() - start_time
        logger.info(
            f"Sector search '{sector_name}' completed in {total_time:.2f}s. "
            f"MongoDB: {len(users)}, After filtering: {len(people)}"
        )
        
        return people
        
    except Exception as e:
        logger.error(f"Error in getShortListedPeopleForSector: {str(e)}")
        traceback.print_exc()
        return {}
    
# ------------------------------------------------------------------------------
# Search for database fields (NEW)
# ------------------------------------------------------------------------------
def getShortListedPeopleForDatabaseQuery(db_queries: list,
                                        userid: str,
                                        operator: str = "AND",
                                        shortListedPeople=None,
                                        max_results: int = search_limits.get_max_results_db_queries()) -> dict:
    """
    Perform regex-based search on various database fields like education.dates,
    accomplishments.Certifications, etc.
    
    Args:
        db_queries: List of query objects with 'field', 'regex', and 'description'
        userid: User ID for filtering
        operator: How to combine multiple queries ("AND" or "OR")
        shortListedPeople: Optional dict to filter results
        max_results: Maximum number of results to return
    
    Returns:
        Dictionary of person_id -> person data
    """
    try:
        if not db_queries:
            logger.info("No database queries provided")
            return {}
            
        logger.info(f"Running database query search with {len(db_queries)} queries, operator={operator}")
        
        # Build MongoDB match conditions
        match_conditions = []
        for query in db_queries:
            field = query.get("field", "")
            regex = query.get("regex", "")
            description = query.get("description", "")
            
            if not field or not regex:
                logger.warning(f"Skipping invalid query: field={field}, regex={regex}")
                continue
                
            logger.info(f"Adding query: {description} - field={field}, regex={regex}")
            
            # Handle array fields differently
            if field.startswith("education.") or field.startswith("workExperience."):
                # Check if it's a positional query (e.g., workExperience.0.companyName)
                field_parts = field.split(".")
                if len(field_parts) >= 3 and field_parts[1].isdigit():
                    # This is a positional query (e.g., workExperience.0.companyName)
                    match_conditions.append({
                        "field": field,
                        "regex": regex,
                        "type": "positional"
                    })
                else:
                    # For regular array fields, we'll handle them specially in the aggregation
                    match_conditions.append({
                        "field": field,
                        "regex": regex,
                        "type": "array"
                    })
            elif field.startswith("accomplishments.Certifications."):
                # Handle accomplishments.Certifications array
                # accomplishments.Certifications.certificateName -> accomplishments.Certifications with $elemMatch
                sub_field = ".".join(field.split(".")[2:])  # Get field after Certifications
                match_conditions.append({
                    "accomplishments.Certifications": {
                        "$elemMatch": {
                            sub_field: {"$regex": regex, "$options": "i"}
                        }
                    }
                })
            elif field.startswith("accomplishments.Courses."):
                # Handle accomplishments.Courses array
                sub_field = ".".join(field.split(".")[2:])  # Get field after Courses
                match_conditions.append({
                    "accomplishments.Courses": {
                        "$elemMatch": {
                            sub_field: {"$regex": regex, "$options": "i"}
                        }
                    }
                })
            elif field.startswith("accomplishments.Projects."):
                # Handle accomplishments.Projects array
                sub_field = ".".join(field.split(".")[2:])  # Get field after Projects
                match_conditions.append({
                    "accomplishments.Projects": {
                        "$elemMatch": {
                            sub_field: {"$regex": regex, "$options": "i"}
                        }
                    }
                })
            elif field == "accomplishments.Languages":
                # Languages is a simple string field
                match_conditions.append({
                    field: {"$regex": regex, "$options": "i"}
                })
            else:
                # Regular field matching
                match_conditions.append({
                    field: {"$regex": regex, "$options": "i"}
                })
        
        if not match_conditions:
            logger.warning("No valid match conditions generated")
            return {}
        
        # Group conditions by their base field (e.g., education, workExperience)
        grouped_conditions = {}
        regular_conditions = []
        positional_conditions = []
        
        for condition in match_conditions:
            if isinstance(condition, dict) and "type" in condition:
                if condition["type"] == "array":
                    field = condition["field"]
                    base_field = field.split(".")[0]
                    sub_field = ".".join(field.split(".")[1:])
                    
                    if base_field not in grouped_conditions:
                        grouped_conditions[base_field] = []
                    
                    grouped_conditions[base_field].append({
                        "sub_field": sub_field,
                        "regex": condition["regex"]
                    })
                elif condition["type"] == "positional":
                    # Handle positional queries directly (e.g., workExperience.0.companyName)
                    positional_conditions.append({
                        condition["field"]: {"$regex": condition["regex"], "$options": "i"}
                    })
            else:
                regular_conditions.append(condition)
        
        # Build MongoDB query
        base_match = {"userId": ObjectId(userid)}
        
        # OPTIMIZATION: Pre-filter at database level if shortListedPeople is provided
        if shortListedPeople:
            shortlisted_ids = [ObjectId(pid) for pid in shortListedPeople.keys()]
            base_match["_id"] = {"$in": shortlisted_ids}
            logger.info(f"Pre-filtering database query search to {len(shortlisted_ids)} shortlisted people")
        
        # Handle grouped array conditions (need all conditions to match in same array element)
        for base_field, conditions in grouped_conditions.items():
            if operator.upper() == "AND" and len(conditions) > 1:
                # All conditions must match in the same array element
                elem_match_condition = {}
                for cond in conditions:
                    elem_match_condition[cond["sub_field"]] = {"$regex": cond["regex"], "$options": "i"}
                base_match[base_field] = {"$elemMatch": elem_match_condition}
            else:
                # OR operator or single condition
                if operator.upper() == "OR":
                    or_conditions = []
                    for cond in conditions:
                        or_conditions.append({
                            base_field: {
                                "$elemMatch": {
                                    cond["sub_field"]: {"$regex": cond["regex"], "$options": "i"}
                                }
                            }
                        })
                    if base_field in base_match:
                        base_match["$or"] = base_match.get("$or", []) + or_conditions
                    else:
                        base_match["$or"] = or_conditions
                else:
                    # Single condition with AND
                    for cond in conditions:
                        base_match[base_field] = {
                            "$elemMatch": {
                                cond["sub_field"]: {"$regex": cond["regex"], "$options": "i"}
                            }
                        }
        
        # Add positional conditions (always use AND logic for these)
        if positional_conditions:
            for cond in positional_conditions:
                base_match.update(cond)
        
        # Add regular conditions
        if regular_conditions:
            if operator.upper() == "AND":
                for cond in regular_conditions:
                    base_match.update(cond)
            else:
                base_match["$or"] = base_match.get("$or", []) + regular_conditions
        
        pipeline = [
            {"$match": base_match},
            {"$project": {
                "_id": 1,
                "userId": 1,
                "name": 1,
                "currentLocation": 1,
                "education": 1,
                "workExperience": 1,
                "accomplishments": 1
            }},
            {"$limit": max_results}
        ]
        
        docs = list(mongoCollectionNodes.aggregate(pipeline))
        logger.info(f"Database query search matched {len(docs)} records")
        
        people = {}
        filtered_count = 0
        
        for doc in docs:
            person_id = str(doc["_id"])
            
            # This check is now redundant if we pre-filtered in MongoDB, but keep for safety
            if shortListedPeople and person_id not in shortListedPeople:
                filtered_count += 1
                logger.warning(f"Person {person_id} found in DB but not in shortlist (should not happen with optimization)")
                continue
            
            result_dict = {
                "personId": person_id,
                "userId": str(doc["userId"]),
                "name": doc.get("name", ""),
                "currentLocation": doc.get("currentLocation", ""),
                "dbQueryMatch": True  # Flag to indicate this came from DB query
            }
            
            # Include existing fields from shortListedPeople if available
            if shortListedPeople and person_id in shortListedPeople:
                for key in ["skillDescription", "skillName", "locationDescription"]:
                    if key in shortListedPeople[person_id]:
                        result_dict[key] = shortListedPeople[person_id][key]
            
            people[person_id] = result_dict
        
        logger.info(
            f"Database query search completed. "
            f"Shortlist filtered: {filtered_count}, Final matches: {len(people)}"
        )
        
        return people
        
    except Exception as e:
        logger.error(f"Error in getShortListedPeopleForDatabaseQuery: {str(e)}")
        traceback.print_exc()
        return {}

# ------------------------------------------------------------------------------
# CLASS: LOGICAL SEARCH PROCESSOR (updated to fire all queries concurrently)
# ------------------------------------------------------------------------------
# Complete Updated LogicalSearchProcessor Class with Progressive Search

class LogicalSearchProcessor:
    """
    Orchestrates the logical search pipeline (skills, locations, organizations).
    Now implements progressive search based on skill priorities.
    """

    def __init__(self,
                 user_id: str,
                 query: str,
                 alternative_skills: bool = False):
        self.user_id = user_id
        self.query = query
        self.max_results_location = search_limits.get_max_results_location()
        self.skill_similarity_threshold = 0.90 if alternative_skills else 0.80
        self.max_results_skill = search_limits.get_max_results_skill()
        self.alternative_skills = alternative_skills
        # New: threshold for progressive search
        self.progressive_search_threshold = search_limits.get_progressive_search_threshold()
        
        # Log the initialized limits
        logger.info(f"SearchLimits initialized - Location: {self.max_results_location}, Skill: {self.max_results_skill}, Progressive: {self.progressive_search_threshold}")

    async def process_skills(self, skill_details, current_results=None):
        """
        Process skills with progressive search based on priority levels.
        Search primary skills first, then secondary only if needed.
        """
        logger.info("Processing skills with progressive search...")
        if not skill_details or not skill_details.get("skills"):
            return current_results if current_results else {}
        
        skills = skill_details["skills"]
        operator = skill_details.get("operator", "AND")
        logger.info(f"Processing {len(skills)} skills with operator={operator}")
        
        # Separate skills by priority
        primary_skills = [s for s in skills if s.get("priority", "primary") == "primary"]
        secondary_skills = [s for s in skills if s.get("priority", "primary") == "secondary"]
        tertiary_skills = [s for s in skills if s.get("priority", "primary") == "tertiary"]
        
        logger.info(f"Skills by priority - Primary: {len(primary_skills)}, Secondary: {len(secondary_skills)}, Tertiary: {len(tertiary_skills)}")
        
        # Batch embed all skills first (for efficiency)
        batch_embed_skills(skills, self.alternative_skills)
        
        # Progressive search implementation
        all_skill_results = []
        total_unique_results = 0
        
        # Step 1: Search primary skills
        if primary_skills:
            logger.info(f"Searching {len(primary_skills)} primary skills...")
            primary_results = await self._search_skill_batch(primary_skills, current_results)
            all_skill_results.extend(primary_results)
            
            # Count unique results
            unique_ids = set()
            for result_dict in primary_results:
                unique_ids.update(result_dict.keys())
            total_unique_results = len(unique_ids)
            
            logger.info(f"Primary skills returned {total_unique_results} unique results")
            
            # Check if we have enough results
            if total_unique_results >= self.progressive_search_threshold:
                logger.info(f"Sufficient results from primary skills ({total_unique_results} >= {self.progressive_search_threshold}). Skipping secondary skills.")
                merged = self._merge_skill_results(all_skill_results, operator, current_results)
                return merged
        
        # Step 2: Search secondary skills if needed
        if secondary_skills and total_unique_results < self.progressive_search_threshold:
            logger.info(f"Searching {len(secondary_skills)} secondary skills (need more results)...")
            secondary_results = await self._search_skill_batch(secondary_skills, current_results)
            all_skill_results.extend(secondary_results)
            
            # Update unique count
            unique_ids = set()
            for result_dict in all_skill_results:
                unique_ids.update(result_dict.keys())
            total_unique_results = len(unique_ids)
            
            logger.info(f"After secondary skills: {total_unique_results} unique results")
            
            # Check again
            if total_unique_results >= self.progressive_search_threshold:
                logger.info(f"Sufficient results after secondary skills. Skipping tertiary skills.")
                merged = self._merge_skill_results(all_skill_results, operator, current_results)
                return merged
        
        # Step 3: Search tertiary skills only as last resort
        if tertiary_skills and total_unique_results < self.progressive_search_threshold:
            logger.info(f"Searching {len(tertiary_skills)} tertiary skills (still need more results)...")
            tertiary_results = await self._search_skill_batch(tertiary_skills, current_results)
            all_skill_results.extend(tertiary_results)
        
        # Final merge
        if not all_skill_results:
            logger.info("No results found for any skills")
            return {}
        
        merged = self._merge_skill_results(all_skill_results, operator, current_results)
        logger.info(f"Skill search complete. Final merged results: {len(merged)}")
        
        # Log which skills contributed to results
        self._log_skill_contribution(all_skill_results, skills)
        
        return merged
    
    async def _search_skill_batch(self, skill_batch, current_results):
        """Helper method to search a batch of skills concurrently."""
        skill_results = []
        
        with ThreadPoolExecutor(max_workers=len(skill_batch)) as executor:
            futures = {
                executor.submit(
                    getShortListedPeopleForSkillAlternative if self.alternative_skills else getShortListedPeopleForSkill,
                    skl,
                    self.user_id,
                    current_results,
                    self.skill_similarity_threshold,
                    self.max_results_skill
                ): skl.get("name", "UnknownSkill") 
                for skl in skill_batch
            }
            
            for future in as_completed(futures):
                skill_name = futures[future]
                try:
                    result = future.result()
                    if result:
                        skill_results.append(result)
                        logger.info(f"Skill search for '{skill_name}' found {len(result)} matches")
                except Exception as e:
                    logger.error(f"Skill search failed for '{skill_name}': {str(e)}")
                    traceback.print_exc()
        
        return skill_results
    
    def _log_skill_contribution(self, skill_results, all_skills):
        """Log metrics about which skills contributed to results."""
        skill_contribution = {}
        
        # Create skill name to priority mapping
        skill_priority_map = {s["name"]: s.get("priority", "primary") for s in all_skills}
        
        # Analyze contribution
        for i, result_dict in enumerate(skill_results):
            # Try to identify which skill this result came from
            # This is a simplified approach - in production you'd track this more precisely
            skill_name = f"Skill_{i}"
            for person_data in result_dict.values():
                if "skillName" in person_data:
                    if isinstance(person_data["skillName"], list) and person_data["skillName"]:
                        skill_name = person_data["skillName"][0]
                    elif isinstance(person_data["skillName"], str):
                        skill_name = person_data["skillName"]
                    break
            
            skill_contribution[skill_name] = len(result_dict)
        
        logger.info("=== Skill Contribution Metrics ===")
        for skill, count in sorted(skill_contribution.items(), key=lambda x: x[1], reverse=True):
            priority = skill_priority_map.get(skill, "unknown")
            logger.info(f"  {skill} ({priority}): {count} results")
        logger.info("=================================")

    async def process_locations(self, location_details, current_results=None):
        logger.info("Processing locations...")
        if not location_details or not location_details.get("locations"):
            return current_results if current_results else {}
        locations = location_details["locations"]
        operator = location_details.get("operator", "AND")
        logger.info(f"Processing {len(locations)} locations with operator={operator}")
        with ThreadPoolExecutor(max_workers=len(locations)) as executor:
            future_to_loc = {
                executor.submit(
                    getShortListedPeopleForLocationRegex,
                    loc,
                    str(self.user_id),
                    current_results,
                    self.max_results_location
                ): loc.get("name", "Unknown") for loc in locations
            }
            loc_results = []
            for future in as_completed(future_to_loc):
                loc_name = future_to_loc[future]
                try:
                    result = future.result()
                    if result:
                        loc_results.append(result)
                    logger.info(f"Location search for '{loc_name}' found {len(result)} matches")
                except Exception as e:
                    logger.error(f"Location search failed for '{loc_name}': {str(e)}")
                    traceback.print_exc()
        if not loc_results:
            logger.info("No results found for any locations")
            return {}
        merged = self._merge_location_results(loc_results, operator, current_results)
        logger.info(f"Location search complete. Merged results: {len(merged)}")
        return merged

    async def process_organizations(self, org_details, current_results=None):
        """Process organization-based search with individual temporal contexts."""
        logger.info("Processing organizations...")
        if not org_details or not org_details.get("organizations"):
            return current_results if current_results else {}
        
        organizations = org_details["organizations"]
        operator = org_details.get("operator", "AND")
        # ← REMOVED temporal extraction
        
        logger.info(f"Processing {len(organizations)} organizations with operator={operator}")
        
        # Log individual temporal contexts
        for org in organizations:
            temp = org.get("temporal", "any")
            logger.info(f"  - {org['name']}: temporal={temp}")
        
        with ThreadPoolExecutor(max_workers=min(len(organizations), 5)) as executor:
            future_to_org = {
                executor.submit(
                    getShortListedPeopleForOrganisation,
                    org,  # ← CHANGED: Pass full org object instead of keywords list
                    self.user_id,
                    # ← REMOVED temporal parameter
                    current_results
                ): org["name"] for org in organizations
            }
            org_results = []
            for future in as_completed(future_to_org):
                org_name = future_to_org[future]
                try:
                    result = future.result()
                    if result:
                        org_results.append(result)
                    logger.info(f"Organization search for '{org_name}' found {len(result)} matches")
                except Exception as e:
                    logger.error(f"Organization search failed for '{org_name}': {str(e)}")
                    traceback.print_exc()
        if not org_results:
            logger.info("No results found for any organizations")
            return {}
        merged = self._merge_organization_results(org_results, operator, current_results)
        logger.info(f"Organization search complete. Merged results: {len(merged)}")
        return merged

    async def process_sectors(self, sector_details, current_results=None):
        """Process sector-based search with individual temporal contexts."""
        logger.info("Processing sectors...")
        if not sector_details or not sector_details.get("sectors"):
            return current_results if current_results else {}
        
        sectors = sector_details["sectors"]
        operator = sector_details.get("operator", "OR")
        # ← REMOVED temporal extraction from here
        
        logger.info(f"Processing {len(sectors)} sectors with operator={operator}")
        
        # Log individual temporal contexts
        for sector in sectors:
            temp = sector.get("temporal", "any")
            logger.info(f"  - {sector['name']}: temporal={temp}")
        
        with ThreadPoolExecutor(max_workers=min(len(sectors), 5)) as executor:
            future_to_sector = {
                executor.submit(
                    getShortListedPeopleForSector,
                    sector,  # ← Already passing full sector object
                    self.user_id,
                    # ← REMOVED temporal parameter
                    current_results,
                    search_limits.get_max_results_sector()  # max_results
                ): sector["name"] for sector in sectors
            }
            
            sector_results = []
            for future in as_completed(future_to_sector):
                sector_name = future_to_sector[future]
                try:
                    result = future.result()
                    if result:
                        sector_results.append(result)
                    logger.info(f"Sector search for '{sector_name}' found {len(result)} matches")
                except Exception as e:
                    logger.error(f"Sector search failed for '{sector_name}': {str(e)}")
                    traceback.print_exc()
        
        if not sector_results:
            logger.info("No results found for any sectors")
            return {}
        
        # Merge sector results
        merged = self._merge_sector_results(sector_results, operator, current_results)
        logger.info(f"Sector search complete. Merged results: {len(merged)}")
        
        return merged
    
    
    def _merge_sector_results(self, results, operator, current_results=None):
        """Merge results from sector searches."""
        if not results:
            return {}
        
        id_sets = [set(r.keys()) for r in results]
        
        if operator.upper() == "AND":
            combined_ids = set.intersection(*id_sets)
        else:
            combined_ids = set.union(*id_sets)
        
        if current_results:
            combined_ids = combined_ids.intersection(set(current_results.keys()))
        
        merged = {}
        for pid in combined_ids:
            # Find the best result for this person across all sector results
            person_results = [r[pid] for r in results if pid in r]
            merged_result = person_results[0].copy()
            
            # Combine sector information from multiple results
            all_sectors = []
            all_companies = []
            for pr in person_results:
                if pr.get("sectorName"):
                    all_sectors.append(pr["sectorName"])
                if pr.get("sectorCompanies"):
                    all_companies.extend(pr["sectorCompanies"])
            
            merged_result["sectorName"] = list(set(all_sectors))
            merged_result["sectorCompanies"] = list(set(all_companies))
            
            # Preserve existing fields from current_results
            if current_results and pid in current_results:
                for k, v in current_results[pid].items():
                    if k not in merged_result:
                        merged_result[k] = v
            
            merged[pid] = merged_result
        
        return merged
    
    async def process_database(self, db_details, current_results=None):
        logger.info("Processing database queries...")
        if not db_details or not db_details.get("queries"):
            return current_results if current_results else {}
        
        queries = db_details["queries"]
        operator = db_details.get("operator", "AND")
        logger.info(f"Processing {len(queries)} database queries with operator={operator}")
        
        # If we have current_results and it's the last step, log a warning
        if current_results:
            logger.info(f"Database query will filter {len(current_results)} existing results")
        
        # Run the database query directly (no need for ThreadPoolExecutor since it's a single operation)
        try:
            result = getShortListedPeopleForDatabaseQuery(
                queries,
                self.user_id,
                operator,
                current_results,
                self.max_results_location  # Use same max as location queries
            )
            if result:
                logger.info(f"Database query search found {len(result)} matches")
            else:
                logger.info("No results found for database queries")
                # If we're filtering existing results and get 0, log which queries failed
                if current_results:
                    logger.warning(f"Database queries filtered out all {len(current_results)} candidates")
                return {}
                
            # For database queries, we just return the results directly since they're already merged
            # based on the operator inside getShortListedPeopleForDatabaseQuery
            return result
            
        except Exception as e:
            logger.error(f"Database query search failed: {str(e)}")
            traceback.print_exc()
            return {}

    # -----------------------------
    # Merge logic methods
    # -----------------------------
    def _merge_skill_results(self, results, operator, current_results=None):
        """Enhanced merge with priority tracking."""
        if not results:
            return {}
        
        id_sets = [set(r.keys()) for r in results]
        total_before_merge = sum(len(s) for s in id_sets)
        unique_before_merge = len(set.union(*id_sets))
        
        logger.info(f"Skill merge starting state:\n"
                    f"  - Total results: {total_before_merge}\n"
                    f"  - Unique before merge: {unique_before_merge}")
        
        if operator.upper() == "AND":
            combined_ids = set.intersection(*id_sets)
            logger.info(f"Using AND operator - Intersection resulted in {len(combined_ids)} people")
        else:
            combined_ids = set.union(*id_sets)
            logger.info(f"Using OR operator - Union resulted in {len(combined_ids)} people")
        
        if current_results:
            before_intersection = len(combined_ids)
            combined_ids = combined_ids.intersection(set(current_results.keys()))
            logger.info(f"After intersecting with current_results: {before_intersection} -> {len(combined_ids)} people")
        
        merged = {}
        unique_skill_counts = defaultdict(int)
        total_similarity = 0
        
        for pid in combined_ids:
            person_results = [r[pid] for r in results if pid in r]
            best_result = max(person_results, key=lambda x: x.get('similarity', 0.0))
            merged_result = best_result.copy()
            total_similarity += best_result.get('similarity', 0.0)
            
            skill_names = set()
            skill_descs = set()
            
            for pr in person_results:
                if isinstance(pr.get('skillName'), list):
                    skill_names.update(pr['skillName'])
                elif pr.get('skillName'):
                    skill_names.add(pr['skillName'])
                
                if isinstance(pr.get('skillDescription'), list):
                    skill_descs.update(pr['skillDescription'])
                elif pr.get('skillDescription'):
                    skill_descs.add(pr['skillDescription'])
                
                if pr.get('skillName'):
                    if isinstance(pr['skillName'], list):
                        for s in pr['skillName']:
                            unique_skill_counts[s] += 1
                    else:
                        unique_skill_counts[pr['skillName']] += 1
            
            merged_result['skillName'] = list(skill_names)
            merged_result['skillDescription'] = list(skill_descs)
            merged_result['similarity'] = best_result.get('similarity', 0.0)
            
            if 'distance' in merged_result:
                del merged_result['distance']
            
            if current_results and pid in current_results:
                for k, v in current_results[pid].items():
                    if k not in merged_result or k in ('locationDescription'):
                        merged_result[k] = v
            
            merged[pid] = merged_result
        
        logger.info(f"Skill merge completed: Final {len(merged)} people, Average similarity: {total_similarity/len(merged) if merged else 0:.3f}")
        logger.info("Skill distribution in final results:")
        for skill, count in sorted(unique_skill_counts.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  - {skill}: {count} people ({count/len(merged)*100:.1f}%)")
        
        return merged

    def _merge_location_results(self, results, operator, current_results=None):
        if not results:
            return {}
        id_sets = [set(r.keys()) for r in results]
        total_before_merge = sum(len(s) for s in id_sets)
        unique_before_merge = len(set.union(*id_sets))
        logger.info(f"Location merge starting state:\n"
                    f"  - Total results: {total_before_merge}\n"
                    f"  - Unique before merge: {unique_before_merge}")
        if operator.upper() == "AND":
            combined_ids = set.intersection(*id_sets)
            logger.info(f"Using AND operator - Intersection resulted in {len(combined_ids)} people")
        else:
            combined_ids = set.union(*id_sets)
            logger.info(f"Using OR operator - Union resulted in {len(combined_ids)} people")
        if current_results:
            before_intersection = len(combined_ids)
            combined_ids = combined_ids.intersection(set(current_results.keys()))
            logger.info(f"After intersecting with current_results: {before_intersection} -> {len(combined_ids)} people")
        merged = {}
        for pid in combined_ids:
            person_results = [r[pid] for r in results if pid in r]
            best_result = max(person_results, key=lambda x: x.get('similarity', 0.0))
            merged_result = best_result.copy()
            loc_desc = list({pr.get('locationDescription', '') for pr in person_results if pr.get('locationDescription')})
            merged_result['locationDescription'] = loc_desc
            merged_result['similarity'] = best_result.get('similarity', 0.0)
            if 'distance' in merged_result:
                del merged_result['distance']
            if current_results and pid in current_results:
                for k, v in current_results[pid].items():
                    if k not in merged_result or k in ('skillDescription'):
                        merged_result[k] = v
            merged[pid] = merged_result
        logger.info(f"Location merge completed: Final {len(merged)} people, "
                    f"Average similarity: {sum(p['similarity'] for p in merged.values())/len(merged) if merged else 0:.3f}")
        return merged

    def _merge_organization_results(self, results, operator, current_results=None):
        if not results:
            return {}
        id_sets = [set(r.keys()) for r in results]
        if operator.upper() == "AND":
            combined_ids = set.intersection(*id_sets)
        else:
            combined_ids = set.union(*id_sets)
        if current_results:
            combined_ids = combined_ids.intersection(set(current_results.keys()))
        merged = {}
        for pid in combined_ids:
            person_results = [r[pid] for r in results if pid in r]
            merged_result = person_results[0].copy()
            if current_results and pid in current_results:
                for k, v in current_results[pid].items():
                    if k not in merged_result or k in ('locationDescription', 'skillDescription'):
                        merged_result[k] = v
            merged[pid] = merged_result
        return merged
    
    def _clean_embeddings(self, hyde_res):
        if not hyde_res or not isinstance(hyde_res, dict):
            return hyde_res
        response = hyde_res.get("response", {})
        if response.get("skillBasedQuery"):
            skills = response.get("skillDetails", {}).get("skills", [])
            for skill in skills:
                if isinstance(skill, dict) and "embeddings" in skill:
                    del skill["embeddings"]
                for role in skill.get("relatedRoles", []):
                    if isinstance(role, dict) and "embeddings" in role:
                        del role["embeddings"]
        return hyde_res

    async def process_search(self, hyde_result, fallback=False):
        """
        Orchestrates the search pipeline with progressive search for skills.
        """
        logger.info("=== Starting Search Pipeline ===")
        
        async def run_pipeline(hyde_res):
            logger.info("--- Starting Pipeline Run ---")
            results = {}
            hyde_res = hyde_res["response"]
            
            # Log skill priorities if present
            if hyde_res.get("skillBasedQuery") and hyde_res.get("skillDetails", {}).get("skills"):
                skills = hyde_res["skillDetails"]["skills"]
                priority_counts = defaultdict(int)
                for skill in skills:
                    priority = skill.get("priority", "primary")
                    priority_counts[priority] += 1
                logger.info(f"Skill priorities: {dict(priority_counts)}")
            
            # Check if both sector and organization queries are present
            has_sectors = hyde_res.get("sectorBasedQuery", False)
            has_organizations = hyde_res.get("organisationBasedQuery", False)
            
            try:
                # Special handling when both sectors and organizations are present
                if has_sectors and has_organizations:
                    logger.info("Both sectors and organizations detected - determining merge logic...")
                    
                    # First apply any database filters if present
                    if hyde_res.get("dbBasedQuery", False):
                        logger.info("Processing database queries first")
                        db_details = hyde_res.get("dbQueryDetails", {})
                        results = await self.process_database(db_details, None)
                        logger.info(f"After database queries, we have {len(results)} results")
                        if not results:
                            logger.info("No results after database queries - stopping pipeline")
                            return [], {"pipeline_steps": "no_results_after_db"}
                    
                    # Process sectors and organizations separately
                    sector_results = await self.process_sectors(
                        hyde_res.get("sectorDetails", {}), 
                        results if results else None
                    )
                    org_results = await self.process_organizations(
                        hyde_res.get("organisationDetails", {}), 
                        results if results else None
                    )
                    
                    # Determine merge logic based on temporal contexts
                    sectors_list = hyde_res.get("sectorDetails", {}).get("sectors", [])
                    orgs_list = hyde_res.get("organisationDetails", {}).get("organizations", [])
                    
                    # Get all unique temporal values
                    sector_temporals = set(s.get("temporal", "any") for s in sectors_list)
                    org_temporals = set(o.get("temporal", "any") for o in orgs_list)
                    
                    logger.info(f"Temporal contexts - Sectors: {sector_temporals}, Organizations: {org_temporals}")
                    
                    # Use AND logic when temporals are clearly different and not "any"
                    use_and_logic = (
                        len(sector_temporals) == 1 and 
                        len(org_temporals) == 1 and 
                        sector_temporals != org_temporals and 
                        "any" not in sector_temporals and 
                        "any" not in org_temporals
                    )
                    
                    logger.info(f"Will use {'AND' if use_and_logic else 'OR'} logic for combining results")
                    
                    combined_results = self._merge_sector_and_org_results(
                        sector_results, org_results, use_and_logic=use_and_logic
                    )
                    logger.info(f"Combined sector+org results: {len(combined_results)} people")
                    
                    # Now process remaining steps with the combined results
                    remaining_steps = [
                        (hyde_res.get("regionBasedQuery", False), "locationDetails", self.process_locations),
                        (hyde_res.get("skillBasedQuery", False), "skillDetails", self.process_skills)
                    ]
                    
                    results = combined_results
                    for is_active, key, func in remaining_steps:
                        if is_active:
                            logger.info(f"Processing {key} after sector+org combination")
                            details = hyde_res.get(key, {})
                            if isinstance(details, dict):
                                for k, v in details.items():
                                    if isinstance(v, datetime):
                                        details[k] = format_datetime(v)
                            results = await func(details, results)
                            logger.info(f"After {key}, we have {len(results)} results")
                            if not results:
                                logger.info(f"No results after {key} - stopping pipeline")
                                break
                else:
                    # Normal pipeline processing when sectors and organizations are not both present
                    # Reorder steps to apply most restrictive filters first
                    steps = [
                        (hyde_res.get("dbBasedQuery", False), "dbQueryDetails", self.process_database),
                        (hyde_res.get("sectorBasedQuery", False), "sectorDetails", self.process_sectors),
                        (hyde_res.get("organisationBasedQuery", False), "organisationDetails", self.process_organizations),
                        (hyde_res.get("regionBasedQuery", False), "locationDetails", self.process_locations),
                        (hyde_res.get("skillBasedQuery", False), "skillDetails", self.process_skills)
                    ]
                    
                    for is_active, key, func in steps:
                        if is_active:
                            logger.info(f"Processing {key}")
                            details = hyde_res.get(key, {})
                            if isinstance(details, dict):
                                for k, v in details.items():
                                    if isinstance(v, datetime):
                                        details[k] = format_datetime(v)
                            results = await func(details, results)
                            logger.info(f"After {key}, we have {len(results)} results")
                            if not results:
                                logger.info(f"No results after {key} - stopping pipeline")
                                break
                
                if not results:
                    logger.info("Pipeline completed with no results")
                    return [], {"pipeline_steps": "no_results"}
                    
                final_list = list(results.values())
                logger.info(f"Pipeline completed successfully with {len(final_list)} results")
                return final_list, {"pipeline_steps": "completed", "result_count": len(final_list)}
                
            except Exception as e:
                logger.error(f"Error in search processing: {str(e)}")
                traceback.print_exc()
                return [], {"pipeline_steps": "error", "error": str(e)}
        
        # Store original for potential fallback
        original = json.loads(json.dumps(hyde_result))
        results_list, metrics = await run_pipeline(hyde_result)
        
        # Clean embeddings before returning
        hyde_result = self._clean_embeddings(hyde_result)
        
        if results_list:
            return results_list, metrics
        
        if fallback:
            logger.info("=== Starting Fallback Strategy ===")
            
            # Fallback 1: Relax similarity threshold while keeping operators intact
            old_skill_thresh = self.skill_similarity_threshold
            self.skill_similarity_threshold = max(0.70, self.skill_similarity_threshold - 0.15)
            logger.info(f"Fallback 1: Relaxing skill similarity threshold from {old_skill_thresh} to {self.skill_similarity_threshold}")
            results_list, metrics = await run_pipeline(original)
            if results_list:
                logger.info("Search successful after fallback 1 (relaxed similarity threshold)")
                self.skill_similarity_threshold = old_skill_thresh
                original = self._clean_embeddings(original)
                return results_list, metrics
            
            # Fallback 2: Further relax threshold
            self.skill_similarity_threshold = max(0.60, self.skill_similarity_threshold - 0.10)
            logger.info(f"Fallback 2: Further relaxing skill similarity threshold to {self.skill_similarity_threshold}")
            results_list, metrics = await run_pipeline(original)
            if results_list:
                logger.info("Search successful after fallback 2 (further relaxed similarity threshold)")
                self.skill_similarity_threshold = old_skill_thresh
                original = self._clean_embeddings(original)
                return results_list, metrics
            
            # Fallback 3: Change operators to OR for locations/orgs/sectors (NOT skills)
            fallback_original = original
            if fallback_original["response"].get("regionBasedQuery", False):
                fallback_original["response"]["locationDetails"]["operator"] = "OR"
                logger.info("Changed location operator to OR")
            if fallback_original["response"].get("organisationBasedQuery", False):
                fallback_original["response"]["organisationDetails"]["operator"] = "OR"
                logger.info("Changed organization operator to OR")
            if fallback_original["response"].get("sectorBasedQuery", False):
                fallback_original["response"]["sectorDetails"]["operator"] = "OR"
                logger.info("Changed sector operator to OR")
            # Explicitly keep skill operator as-is to preserve query semantics
            logger.info("Fallback 3: Changed location/org/sector operators to OR (kept skills operator unchanged)")
            
            results_list, metrics = await run_pipeline(fallback_original)
            if results_list:
                logger.info("Search successful after fallback 3")
                self.skill_similarity_threshold = old_skill_thresh
                fallback_original = self._clean_embeddings(fallback_original)
                return results_list, metrics
            
            self.skill_similarity_threshold = old_skill_thresh
            logger.info("No results found even after all fallback steps")
        
        logger.info("=== Search Pipeline Complete ===")
        return results_list, metrics
    
    def _merge_sector_and_org_results(self, sector_results, org_results, use_and_logic=False):
        """
        Special merge logic when both sectors and organizations are present.
        Can use either AND or OR logic based on the use_and_logic parameter.
        
        Args:
            sector_results: Results from sector search
            org_results: Results from organization search  
            use_and_logic: If True, use AND logic (intersection), else use OR logic (union)
        """
        if not sector_results and not org_results:
            return {}
        if not sector_results:
            return org_results
        if not org_results:
            return sector_results
        
        # Choose between intersection (AND) or union (OR) based on use_and_logic
        if use_and_logic:
            # Intersection the results (AND operation)
            all_ids = set(sector_results.keys()) & set(org_results.keys())
            logger.info(f"Using AND logic - intersection of sector and org results")
        else:
            # Union the results (OR operation)
            all_ids = set(sector_results.keys()) | set(org_results.keys())
            logger.info(f"Using OR logic - union of sector and org results")
        merged = {}
        
        for pid in all_ids:
            if use_and_logic:
                # For AND logic, person must be in both results
                if pid in sector_results and pid in org_results:
                    # Person appears in both - merge the data
                    result = sector_results[pid].copy()
                    
                    # Preserve the better similarity score
                    if "similarity" in org_results[pid]:
                        result["similarity"] = max(
                            result.get("similarity", 0),
                            org_results[pid].get("similarity", 0)
                        )
                    
                    # Combine any additional fields from org_results
                    for key in org_results[pid]:
                        if key not in result and key not in ["similarity", "distance"]:
                            result[key] = org_results[pid][key]
                        
                    # Mark that this person matched both sector and org
                    result["matchedBoth"] = True
                    merged[pid] = result
            else:
                # For OR logic, handle each case
                if pid in sector_results and pid in org_results:
                    # Person appears in both - merge the data
                    result = sector_results[pid].copy()
                    
                    # Preserve the better similarity score
                    if "similarity" in org_results[pid]:
                        result["similarity"] = max(
                            result.get("similarity", 0),
                            org_results[pid].get("similarity", 0)
                        )
                    
                    # Combine any additional fields from org_results
                    for key in org_results[pid]:
                        if key not in result and key not in ["similarity", "distance"]:
                            result[key] = org_results[pid][key]
                        
                    # Mark that this person matched both sector and org
                    result["matchedBoth"] = True
                    
                elif pid in sector_results:
                    result = sector_results[pid].copy()
                    result["matchedSectorOnly"] = True
                else:
                    result = org_results[pid].copy()
                    result["matchedOrgOnly"] = True
                
                merged[pid] = result
        
        logger.info(
            f"Merged sector and org results: {len(merged)} total people "
            f"(sectors: {len(sector_results)}, orgs: {len(org_results)})"
        )
        
        return merged
    
    
if __name__ == "__main__":
    async def main():
        start_time = time.time()
        with open('outputTest/4.1.json', 'r') as f:
            hyde_result = json.load(f)
        processor = LogicalSearchProcessor(
            user_id="6797bf304791caa516f6da9e",
            query="developers who have worked in ai in the last 2 years, and prior ml work in recommendation systems",
            # query="people in ml/ai from blr who have worked in the faang before",
            alternative_skills=True
        )
        results_list, search_metrics = await processor.process_search(hyde_result, fallback=True)
        end_time = time.time()
        logger.info(f"Total search execution time: {end_time - start_time:.2f} seconds")
        debug_filename = f'outputTest/output1_{int(time.time())}.json'
        
        # Ensure all data is JSON serializable
        serializable_results = []
        for result in results_list:
            # Convert any non-serializable types to their string representation
            serializable_result = {}
            for key, value in result.items():
                if isinstance(value, (datetime, ObjectId)):
                    serializable_result[key] = str(value)
                else:
                    serializable_result[key] = value
            serializable_results.append(serializable_result)
            
        with open(debug_filename, 'w') as f:
            json.dump({
                'results_list': serializable_results,
                'search_metrics': search_metrics,
                'execution_time': end_time - start_time
            }, f, indent=2)
        logger.info(f"Saved search results to {debug_filename}")
    asyncio.run(main())
