# FetchAndRank Lambda (Part 2 of Logical Search Migration)

This Lambda function implements the **Fetch & Rank** component of the logical search system, responsible for:
- Processing search queries using `LogicalSearchProcessor`
- Ranking results using sophisticated scoring algorithms
- Persisting results to MongoDB
- Returning ranked, enriched search results

## Input Format

```json
{
    "hyde_output": {
        "query_breakdown": {...},
        "response": {
            "regionBasedQuery": {...},
            "locationDetails": {...},
            "organisationDetails": {...},
            "sectorDetails": {...},
            "skillDetails": {...},
            "dbQueryDetails": {...}
        }
    },
    "user_id": "507f1f77bcf86cd799439012",
    "query": "original search query",
    "flags": {
        "fallback": false,
        "reasoning_model": "groq_llama",
        "alternative_skills": false,
        ...
    }
}
```

## Output Format

```json
{
    "success": true,
    "result": {
        "ranked_results": [...],
        "search_metrics": {...},
        "result_count": 15,
        "processing_time": 2.34,
        "hyde_analysis": {...}
    },
    "user_id": "...",
    "query": "...",
    "processed_at": "2024-01-01T12:00:00.000Z"
}
```

## Environment Variables

- `MONGODB_URI`: MongoDB connection string
- `UPSTASH_URL`: Upstash Vector DB URL
- `UPSTASH_TOKEN`: Upstash authentication token
- `REDIS_HOST`: Redis host for caching
- `REDIS_PORT`: Redis port (default: 6379)
- `REDIS_PASSWORD`: Redis authentication password
- `CLOUDFLARE_API_KEY`: For avatar image processing
- `JINA_API_KEY`: For embedding generation (optional)

## Architecture

- **lambda_handler.py**: Main Lambda entry point
- **search_processor.py**: Core FetchAndRank logic orchestration
- **logic/**: Contains all search, ranking and utility modules
  - `search.py`: LogicalSearchProcessor and search functions
  - `ranking.py`: Ranking algorithms and scoring
  - `search_config.py`: Configuration and limits
  - `jsonToXml.py`: Profile data conversion
  - `cloudflareFunctions.py`: Avatar processing
  - `utils.py`: Utility functions
- **llm_config/**: LLM management and configuration
- **prompts/**: Ranking prompts

## Deployment

1. Install dependencies: `pip install -r requirements.txt`
2. Set environment variables
3. Deploy to AWS Lambda with appropriate IAM permissions for MongoDB and Redis access