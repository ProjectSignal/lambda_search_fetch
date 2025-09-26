#!/usr/bin/env python3
"""
Test script for Fetch Lambda function
"""

import json
import sys
import os
import uuid
import argparse
from datetime import datetime, timezone

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lambda_handler import lambda_handler, SearchStatus

def create_test_search_document_with_hyde():
    """Create a test search document in HYDE_COMPLETE status with HyDE analysis"""
    
    try:
        from api_client import create_search_document

        search_id = str(uuid.uuid4())
        user_id = "6797bf304791caa516f6da9e"  # Valid ObjectId for testing
        query = "Find machine learning experts based out of blr and graduated from iit"
        
        now = datetime.now(timezone.utc)
        test_doc = {
            "_id": search_id,
            "userId": user_id,
            "query": query,
            "flags": {
                "hyde_provider": "gemini",
                "description_provider": "gemini",
                "reasoning_model": "gemini",
                "alternative_skills": True,
                "reasoning": True,
                "fallback": False
            },
            "status": SearchStatus.HYDE_COMPLETE,
            "createdAt": now.isoformat(),
            "updatedAt": now.isoformat(),
            "events": [
                {
                    "stage": "INIT",
                    "message": "Search initiated (test document)",
                    "timestamp": now.isoformat()
                },
                {
                    "id": f"HYDE:{search_id}",
                    "stage": "HYDE",
                    "message": "HyDE analysis completed",
                    "timestamp": now.isoformat()
                }
            ],
            "metrics": {
                "hydeMs": 2500
            },
            "hydeAnalysis": {
            "query_breakdown": {
              "key_components": [
                "Machine Learning experts",
                "Python experience",
                "Location: San Francisco"
              ],
              "analysis": "Query seeks individuals with expertise in Machine Learning and experience with Python, located in San Francisco."
            },
            "response": {
              "regionBasedQuery": 1,
              "locationDetails": {
                "operator": "OR",
                "locations": [
                  {
                    "name": "San Francisco"
                  }
                ]
              },
              "sectorBasedQuery": 0,
              "sectorDetails": {
                "operator": "OR",
                "sectors": []
              },
              "organisationBasedQuery": 0,
              "organisationDetails": {
                "operator": "OR",
                "organizations": []
              },
              "skillBasedQuery": 1,
              "skillDetails": {
                "operator": "AND",
                "skills": [
                  {
                    "name": "Machine Learning",
                    "priority": "primary",
                    "relatedRoles": [
                      "ML Engineer",
                      "Data Scientist",
                      "AI Engineer"
                    ],
                    "titleKeywords": [
                      "machine learning engineer",
                      "ml engineer",
                      "data scientist",
                      "ml scientist"
                    ],
                    "regexPatterns": {
                      "keywords": [
                        "machine learning",
                        "\\bml\\b",
                        "deep learning",
                        "neural network",
                        "pytorch",
                        "tensorflow"
                      ],
                      "fields": [
                        "workExperience.title",
                        "linkedinHeadline",
                        "workExperience.description",
                        "bio"
                      ]
                    }
                  },
                  {
                    "name": "Python",
                    "priority": "secondary",
                    "regexPatterns": {
                      "keywords": [
                        "python"
                      ],
                      "fields": [
                        "workExperience.description",
                        "bio",
                        "education.description"
                      ]
                    }
                  }
                ]
              },
              "dbBasedQuery": 0,
              "dbQueryDetails": {
                "operator": "AND",
                "queries": []
              }
            }
          }
        }
        
        create_search_document(test_doc)
        print(f"✅ Created test search document with HyDE analysis: {search_id}")
        return search_id, user_id, query, test_doc["flags"]

    except Exception as e:
        print(f"❌ Error creating test search document: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

def find_existing_search_document(search_id):
    """Find an existing search document by searchId"""

    try:
        from api_client import get_search_document

        doc = get_search_document(search_id)
        if not doc:
            print(f"❌ No document found with searchId: {search_id}")
            return None, None, None, None

        # Validate that document has HyDE analysis
        if doc.get('status') != SearchStatus.HYDE_COMPLETE:
            print(f"❌ Document found but status is {doc.get('status')}, expected HYDE_COMPLETE")
            return None, None, None, None

        if not doc.get('hydeAnalysis'):
            print(f"❌ Document found but missing HyDE analysis")
            return None, None, None, None

        print(f"✅ Found existing document with HyDE analysis: {search_id}")

        user_id = doc.get('userId')
        query = doc.get('query')
        flags = doc.get('flags', {})

        return search_id, user_id, query, flags

    except Exception as e:
        print(f"❌ Error finding existing search document: {str(e)}")
        return None, None, None, None

def create_step_functions_event(search_id, user_id, query, flags):
    """Create Step Functions event format for Fetch Lambda"""
    return {
        "searchId": search_id,
        "userId": user_id,
        "query": query,
        "flags": flags
    }

def test_fetch_lambda(existing_search_id=None):
    """Test the Fetch Lambda with Step Functions event format"""

    print("Testing Fetch Lambda...")
    print("=" * 50)

    # Step 1: Get test search document (existing or create new)
    if existing_search_id:
        print(f"1. Using existing search document: {existing_search_id}")
        search_id, user_id, query, flags = find_existing_search_document(existing_search_id)
    else:
        print("1. Creating test search document with HyDE analysis...")
        search_id, user_id, query, flags = create_test_search_document_with_hyde()

    if not search_id:
        print("❌ Cannot proceed without test search document")
        return None

    # Step 2: Create Step Functions event
    test_event = create_step_functions_event(search_id, user_id, query, flags)
    
    print("\n2. Step Functions Test Event:")
    print(json.dumps(test_event, indent=2))

    # Step 3: Mock context
    context = {}

    try:
        print("\n3. Running Fetch Lambda...")
        
        # Execute Lambda (Note: lambda_handler is synchronous, not async)
        result = lambda_handler(test_event, context)

        print("\n=== Lambda Response ===")
        print(f"Status Code: {result['statusCode']}")

        # Parse and pretty print the body
        if isinstance(result['body'], str):
            body = json.loads(result['body'])
        else:
            body = result['body']
            
        print("\nResponse Body:")
        print(json.dumps(body, indent=2))

        # Step 4: Validate search document was updated
        if result['statusCode'] == 200:
            print("\n4. Validating search document update...")
            validate_search_document_update(search_id)

        return result

    except Exception as e:
        print(f"\n❌ Error running lambda: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def validate_search_document_update(search_id):
    """Validate that the search document was properly updated with search results"""
    
    try:
        from api_client import get_search_document

        doc = get_search_document(search_id)
        if not doc:
            print("❌ Search document not found")
            return False
            
        print(f"   Status: {doc.get('status')}")
        
        if doc.get('status') == SearchStatus.SEARCH_COMPLETE:
            print("   ✅ Status updated to SEARCH_COMPLETE")
            
            results = doc.get('results')
            if results:
                print("   ✅ Search results present")
                candidates = results.get('candidates', [])
                summary = results.get('summary', {})
                
                candidate_count = len(candidates)
                summary_count = summary.get('count', 0)
                
                print(f"   Candidates found: {candidate_count}")
                print(f"   Summary count: {summary_count}")
                
                if candidate_count > 0:
                    print("   ✅ Candidates retrieved successfully")
                    # Show sample candidate structure
                    sample = candidates[0]
                    print(f"   Sample candidate keys: {list(sample.keys()) if isinstance(sample, dict) else 'N/A'}")
                else:
                    print("   ⚠️  No candidates found (may be expected for test data)")
                    
                search_metrics = doc.get('searchMetrics', {})
                if search_metrics:
                    print(f"   Search metrics keys: {list(search_metrics.keys())}")
                    
                return True
            else:
                print("   ❌ Search results missing")
                return False
        else:
            print(f"   ❌ Unexpected status: {doc.get('status')}")
            return False
    except Exception as e:
        print(f"   ❌ Error validating document: {str(e)}")
        return False

def cleanup_test_document(search_id):
    """Clean up test document after test"""
    try:
        from api_client import delete_search_document

        delete_search_document(search_id)
        print(f"🧹 Cleaned up test document: {search_id}")
    except Exception as e:
        print(f"⚠️  Could not clean up test document: {str(e)}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test Fetch Lambda function')
    parser.add_argument('--search-id', type=str,
                       help='Use existing searchId from HyDE test (for pipeline testing)')
    args = parser.parse_args()

    print("🚀 Starting Fetch Lambda Test")
    if args.search_id:
        print(f"📌 Using existing searchId: {args.search_id}")

    # Run the test
    result = test_fetch_lambda(existing_search_id=args.search_id)

    print("\n" + "=" * 50)

    if result and result['statusCode'] == 200:
        print("🎉 Fetch Lambda test completed successfully!")

        # Extract searchId from result for cleanup (only if we created it)
        if not args.search_id:  # Only cleanup if we created the document
            try:
                if isinstance(result['body'], str):
                    body = json.loads(result['body'])
                else:
                    body = result['body']
                search_id = body.get('searchId')
                if search_id:
                    cleanup_test_document(search_id)
            except:
                pass
        else:
            print("💡 Using existing document - skipping cleanup")

        sys.exit(0)
    else:
        print("💥 Fetch Lambda test failed!")
        sys.exit(1)
