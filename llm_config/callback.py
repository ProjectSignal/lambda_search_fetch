from typing import Dict, Any
import json


class CustomCallback:
    def on_request_start(
        self,
        provider: str,
        model: str,
        messages: list,
        **kwargs
    ):
        print(f"Starting request to {provider} using model {model}")
        
    def on_request_end(
        self,
        provider: str,
        model: str,
        response: Dict[str, Any],
        **kwargs
    ):
        print(f"Request completed for {provider} using model {model}")
        
    def on_request_error(
        self,
        provider: str,
        model: str,
        error: Exception,
        **kwargs
    ):
        logger.error(f"Error in request to {provider} using model {model}: {str(error)}")