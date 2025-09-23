import logging
from config import get_env_var

logger = logging.getLogger(__name__)


class SearchLimits:
    """
    Centralized configuration class for managing search limits.
    
    This class provides a single point of configuration for all search-related
    limits used throughout the application. It supports environment variable
    overrides while maintaining sensible defaults.
    """
    
    # Default values - doubled from original limits
    DEFAULT_MAX_RESULTS_LOCATION = 1000
    DEFAULT_MAX_RESULTS_SKILL = 700
    DEFAULT_PROGRESSIVE_SEARCH_THRESHOLD = 40
    DEFAULT_MAX_RESULTS_ORG = 1000
    DEFAULT_MAX_RESULTS_SECTOR = 4000
    
    @classmethod
    def get_max_results_location(cls) -> int:
        """
        Get the maximum number of results for location-based searches.
        
        Checks the SEARCH_MAX_RESULTS_LOCATION environment variable first,
        falling back to the default value of 500.
        
        Returns:
            int: Maximum results for location searches
        """
        env_value = get_env_var("SEARCH_MAX_RESULTS_LOCATION", required=False)
        if env_value:
            try:
                value = int(env_value)
                logger.info(f"Using environment override for location search limit: {value}")
                return value
            except ValueError:
                logger.warning(f"Invalid value for SEARCH_MAX_RESULTS_LOCATION: {env_value}. Using default: {cls.DEFAULT_MAX_RESULTS_LOCATION}")
        return cls.DEFAULT_MAX_RESULTS_LOCATION
    
    @classmethod
    def get_max_results_skill(cls) -> int:
        """
        Get the maximum number of results for skill-based searches.
        
        Checks the SEARCH_MAX_RESULTS_SKILL environment variable first,
        falling back to the default value of 350.
        
        Returns:
            int: Maximum results for skill searches
        """
        env_value = get_env_var("SEARCH_MAX_RESULTS_SKILL", required=False)
        if env_value:
            try:
                value = int(env_value)
                logger.info(f"Using environment override for skill search limit: {value}")
                return value
            except ValueError:
                logger.warning(f"Invalid value for SEARCH_MAX_RESULTS_SKILL: {env_value}. Using default: {cls.DEFAULT_MAX_RESULTS_SKILL}")
        return cls.DEFAULT_MAX_RESULTS_SKILL
    
    @classmethod
    def get_progressive_search_threshold(cls) -> int:
        """
        Get the threshold for progressive search functionality.
        
        Checks the SEARCH_PROGRESSIVE_THRESHOLD environment variable first,
        falling back to the default value of 40.
        
        Returns:
            int: Progressive search threshold
        """
        env_value = get_env_var("SEARCH_PROGRESSIVE_THRESHOLD", required=False)
        if env_value:
            try:
                value = int(env_value)
                logger.info(f"Using environment override for progressive search threshold: {value}")
                return value
            except ValueError:
                logger.warning(f"Invalid value for SEARCH_PROGRESSIVE_THRESHOLD: {env_value}. Using default: {cls.DEFAULT_PROGRESSIVE_SEARCH_THRESHOLD}")
        return cls.DEFAULT_PROGRESSIVE_SEARCH_THRESHOLD
    
    @classmethod
    def get_max_results_org(cls) -> int:
        """
        Get the maximum number of results for organization-based searches.
        
        Checks the SEARCH_MAX_RESULTS_ORG environment variable first,
        falling back to the default value of 500.
        
        Returns:
            int: Maximum results for organization searches
        """
        env_value = get_env_var("SEARCH_MAX_RESULTS_ORG", required=False)
        if env_value:
            try:
                value = int(env_value)
                logger.info(f"Using environment override for organization search limit: {value}")
                return value
            except ValueError:
                logger.warning(f"Invalid value for SEARCH_MAX_RESULTS_ORG: {env_value}. Using default: {cls.DEFAULT_MAX_RESULTS_ORG}")
        return cls.DEFAULT_MAX_RESULTS_ORG
    
    @classmethod
    def get_max_results_sector(cls) -> int:
        """
        Get the maximum number of results for sector-based searches.
        
        Checks the SEARCH_MAX_RESULTS_SECTOR environment variable first,
        falling back to the default value of 2000.
        
        Returns:
            int: Maximum results for sector searches
        """
        env_value = get_env_var("SEARCH_MAX_RESULTS_SECTOR", required=False)
        if env_value:
            try:
                value = int(env_value)
                logger.info(f"Using environment override for sector search limit: {value}")
                return value
            except ValueError:
                logger.warning(f"Invalid value for SEARCH_MAX_RESULTS_SECTOR: {env_value}. Using default: {cls.DEFAULT_MAX_RESULTS_SECTOR}")
        return cls.DEFAULT_MAX_RESULTS_SECTOR
    
    @classmethod
    def get_max_results_title(cls) -> int:
        """
        Get the maximum number of results for title keyword searches.
        
        Checks the SEARCH_MAX_RESULTS_TITLE environment variable first,
        falling back to the default value of 500.
        
        Returns:
            int: Maximum results for title keyword searches
        """
        env_value = get_env_var("SEARCH_MAX_RESULTS_TITLE", required=False)
        if env_value:
            try:
                value = int(env_value)
                logger.info(f"Using environment override for title search limit: {value}")
                return value
            except ValueError:
                logger.warning(f"Invalid value for SEARCH_MAX_RESULTS_TITLE: {env_value}. Using default: {cls.DEFAULT_MAX_RESULTS_LOCATION}")
        return cls.DEFAULT_MAX_RESULTS_LOCATION
    
    @classmethod
    def get_max_results_db_queries(cls) -> int:
        """
        Get the maximum number of results for database field searches.
        
        Checks the SEARCH_MAX_RESULTS_DB_QUERIES environment variable first,
        falling back to the default value of 500.
        
        Returns:
            int: Maximum results for database field searches
        """
        env_value = get_env_var("SEARCH_MAX_RESULTS_DB_QUERIES", required=False)
        if env_value:
            try:
                value = int(env_value)
                logger.info(f"Using environment override for database query search limit: {value}")
                return value
            except ValueError:
                logger.warning(f"Invalid value for SEARCH_MAX_RESULTS_DB_QUERIES: {env_value}. Using default: {cls.DEFAULT_MAX_RESULTS_LOCATION}")
        return cls.DEFAULT_MAX_RESULTS_LOCATION
    
    @classmethod
    def update_from_env(cls) -> dict:
        """
        Refresh all limits from environment variables and return current values.
        
        This method can be called at runtime to update all search limits
        from the current environment variable values.
        
        Returns:
            dict: Dictionary containing all current limit values
        """
        current_limits = {
            'max_results_location': cls.get_max_results_location(),
            'max_results_skill': cls.get_max_results_skill(),
            'progressive_search_threshold': cls.get_progressive_search_threshold(),
            'max_results_org': cls.get_max_results_org(),
            'max_results_sector': cls.get_max_results_sector(),
            'max_results_title': cls.get_max_results_title(),
            'max_results_db_queries': cls.get_max_results_db_queries()
        }
        
        logger.info(f"Search limits updated from environment: {current_limits}")
        return current_limits