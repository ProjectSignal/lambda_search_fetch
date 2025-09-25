import os
import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from logic.search import LogicalSearchProcessor
from logging_config import setup_logger

logger = setup_logger(__name__)

class SearchProcessor:
    """
    Main processor for Search Lambda functionality.
    Handles search processing and candidate preparation.
    """

    def __init__(self):
        """Initialize the processor with required components"""
        # Note: LogicalSearchProcessor will be initialized per request with specific parameters
        logger.info("SearchProcessor initialized")

    async def process(self, hyde_output: Dict[str, Any], user_id: str, query: str, flags: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing method that orchestrates search and candidate preparation

        Args:
            hyde_output: HyDE analysis results from Lambda 1
            user_id: User ID for the request
            query: Original search query
            flags: Search configuration flags

        Returns:
            Dictionary containing candidate results and metadata
        """
        start_time = datetime.utcnow()

        try:
            logger.info(f"Starting Search processing for user {user_id}")

            # Extract search parameters from HyDE output
            search_params = self._extract_search_params(hyde_output)

            # Execute search processing
            logger.info("Executing search processing...")
            people_data, search_metrics = await self._execute_search(search_params, flags, user_id, query)

            # Prepare candidates for ranking in RankAndReasoning Lambda
            candidates = []
            if people_data.get('count', 0) > 0:
                logger.info(f"Found {people_data['count']} search results, preparing candidates for ranking...")
                # Convert people_data to list of candidates
                candidates = list(people_data.get('people', {}).values())

            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()

            # Prepare result
            result = {
                'candidates': candidates,
                'search_metrics': search_metrics,
                'result_count': len(candidates),
                'processing_time': processing_time,
                'hyde_analysis': hyde_output
            }

            # Results will be persisted by the lambda handler

            logger.info(f"Search processing completed: {len(candidates)} candidates")
            return result

        except Exception as e:
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            logger.error(f"Error in FetchAndRank processing: {str(e)}")

            # Store error result
            error_result = {
                'error': str(e),
                'processing_time': processing_time,
                'hyde_analysis': hyde_output
            }

            # Error will be persisted by the lambda handler

            raise

    def _extract_search_params(self, hyde_output: Dict[str, Any]) -> Dict[str, Any]:
        """Extract search parameters from HyDE analysis output"""
        response = hyde_output.get('response', {})

        return {
            'regionBasedQuery': response.get('regionBasedQuery'),
            'locationDetails': response.get('locationDetails', {}),
            'organisationBasedQuery': response.get('organisationBasedQuery'),
            'organisationDetails': response.get('organisationDetails', {}),
            'sectorBasedQuery': response.get('sectorBasedQuery'),
            'sectorDetails': response.get('sectorDetails', {}),
            'skillBasedQuery': response.get('skillBasedQuery'),
            'skillDetails': response.get('skillDetails', {}),
            'dbBasedQuery': response.get('dbBasedQuery'),
            'dbQueryDetails': response.get('dbQueryDetails', {})
        }

    async def _execute_search(self, search_params: Dict[str, Any], flags: Dict[str, Any], user_id: str, query: str) -> tuple:
        """Execute the search processing using LogicalSearchProcessor"""
        try:
            # Initialize LogicalSearchProcessor with required parameters
            search_processor = LogicalSearchProcessor(
                user_id=user_id,
                query=query,
                alternative_skills=flags.get('alternative_skills', False)
            )

            # Reconstruct hyde_result format expected by process_search
            hyde_result = {
                "response": search_params
            }

            # Call process_search with correct signature
            people_data, search_metrics = await search_processor.process_search(
                hyde_result=hyde_result,
                fallback=flags.get('fallback', False)
            )

            # Transform list to expected dictionary format (matching original implementation)
            if isinstance(people_data, list):
                people_dict = {}
                for person in people_data:
                    # Use nodeId as the key (each result has this field)
                    node_id = person.get("nodeId")
                    if node_id:
                        people_dict[node_id] = person

                # Create the expected format
                people_data = {
                    "count": len(people_data),
                    "people": people_dict
                }

            logger.info(f"Search completed: {people_data.get('count', 0)} results found")
            return people_data, search_metrics

        except Exception as e:
            logger.error(f"Search processing failed: {str(e)}")
            raise


