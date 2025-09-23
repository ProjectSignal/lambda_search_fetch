import os
import logging
import sys
from datetime import datetime

# Flag to ensure root logger is configured only once
_root_logger_configured = False

def setup_logger(name: str) -> logging.Logger:
    """
    Sets up logging handlers on the root logger (if not already done)
    and returns a logger instance for the given name.
    """
    global _root_logger_configured
    
    if not _root_logger_configured:
        root_logger = logging.getLogger() # Get the root logger
        root_logger.setLevel(logging.INFO) # Set root level

        # Clear existing handlers from root logger (important in case libraries added some)
        # Make a copy of the list while iterating because we're modifying it
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Create formatters
        # Include logger name, level, and message for clarity
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        # Use a detailed formatter for console now too, including logger name
        console_formatter = logging.Formatter(
             '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Create console handler for INFO to stdout
        info_console_handler = logging.StreamHandler(sys.stdout)
        info_console_handler.setLevel(logging.INFO)
        info_console_handler.setFormatter(console_formatter)
        # Filter to ONLY handle INFO level logs (strictly INFO, not INFO and below)
        info_console_handler.addFilter(lambda record: record.levelno == logging.INFO)

        # Create console handler for WARNING and above to stderr
        error_console_handler = logging.StreamHandler(sys.stderr)
        error_console_handler.setLevel(logging.WARNING)
        error_console_handler.setFormatter(console_formatter)
        # No filter needed here, level setting is enough for WARNING+

        # Add handlers to the root logger
        root_logger.addHandler(info_console_handler)
        root_logger.addHandler(error_console_handler)

        _root_logger_configured = True
        # Use logging directly here as the logger instance isn't fully set up yet in this scope
        logging.info(f"Root logger configured successfully.") 

    # Return the specific logger for the module calling this function
    # It will inherit the handlers from the root logger.
    return logging.getLogger(name)