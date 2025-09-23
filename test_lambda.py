#!/usr/bin/env python3

import json
import asyncio
from lambda_handler import lambda_handler

def create_test_event():
    """Create a sample test event for FetchAndRank Lambda"""
    return {
        "hyde_output": {
            "query_breakdown": {
                "location": "San Francisco",
                "skills": ["python", "machine learning"],
                "organizations": ["tech startups"]
            },
            "response": {
                "regionBasedQuery": True,
                "locationDetails": {
                    "name": "San Francisco",
                    "alternatives": ["SF", "Bay Area"],
                    "description": "Tech hub city"
                },
                "skillDetails": {
                    "skills": [
                        {"name": "python", "description": "Programming language"},
                        {"name": "machine learning", "description": "AI/ML expertise"}
                    ]
                },
                "organisationDetails": {},
                "sectorDetails": {},
                "dbQueryDetails": {}
            }
        },
        "user_id": "507f1f77bcf86cd799439011",
        "query": "Find Python developers in San Francisco with machine learning experience",
        "flags": {
            "fallback": False,
            "reasoning_model": "groq_llama",
            "alternative_skills": False,
            "reasoning": True
        }
    }

def test_lambda_locally():
    """Test the Lambda function locally with sample data"""
    print("Testing FetchAndRank Lambda locally...")

    # Create test event
    event = create_test_event()
    context = {}  # Mock Lambda context

    print("Test Event:")
    print(json.dumps(event, indent=2))
    print("\n" + "="*50)

    try:
        # Execute Lambda
        result = lambda_handler(event, context)

        print("Lambda Response:")
        print(json.dumps(result, indent=2, default=str))

        if result.get('statusCode') == 200:
            print("\n✅ Lambda executed successfully!")
        else:
            print(f"\n❌ Lambda failed with status: {result.get('statusCode')}")

    except Exception as e:
        print(f"\n❌ Lambda execution failed: {str(e)}")

if __name__ == "__main__":
    test_lambda_locally()