import os
import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from logic.search import LogicalSearchProcessor
from logic.search import enrich_candidates, analyze_hyde_data_requirements, process_mutuals, process_avatar_urls_batch
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

            # Execute ranking if we have results
            ranked_results = []
            if people_data.get('count', 0) > 0:
                logger.info(f"Ranking {people_data['count']} search results...")
                ranked_results = await self._execute_ranking(
                    people_data=people_data,
                    hyde_output=hyde_output,
                    flags=flags
                )

            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()

            # Prepare result
            result = {
                'ranked_results': ranked_results,
                'search_metrics': search_metrics,
                'result_count': len(ranked_results),
                'processing_time': processing_time,
                'hyde_analysis': hyde_output
            }

            # Results will be persisted by the lambda handler

            logger.info(f"Search processing completed: {len(ranked_results)} candidates")
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
            'organisationDetails': response.get('organisationDetails', {}),
            'sectorDetails': response.get('sectorDetails', {}),
            'skillDetails': response.get('skillDetails', {}),
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

            logger.info(f"Search completed: {people_data.get('count', 0)} results found")
            return people_data, search_metrics

        except Exception as e:
            logger.error(f"Search processing failed: {str(e)}")
            raise

    async def _execute_ranking(self, people_data: Dict[str, Any], hyde_output: Dict[str, Any], flags: Dict[str, Any]) -> list:
        """Execute ranking on search results"""
        try:
            # Analyze HyDE data requirements for ranking
            data_requirements = analyze_hyde_data_requirements(hyde_output)

            # Process mutual connections if needed
            if data_requirements.get('needs_mutuals'):
                people_data = process_mutuals(people_data)

            # Process avatar URLs if needed
            if data_requirements.get('needs_avatars'):
                people_data = await process_avatar_urls_batch(people_data)

            # Execute candidate enrichment (formerly ranking)
            enriched_results = await enrich_candidates(
                people_data=people_data,
                hyde_analysis=hyde_output,
                reasoning_model=flags.get('reasoning_model', 'groq_llama'),
                additional_flags=flags
            )

            logger.info(f"Enrichment completed: {len(enriched_results)} enriched results")
            return enriched_results

        except Exception as e:
            logger.error(f"Ranking failed: {str(e)}")
            raise

