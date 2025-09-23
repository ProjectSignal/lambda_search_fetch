import os
from typing import List, Dict, Optional, Any
from litellm import ModelResponse
import litellm
from openai import OpenAIError
from .model_config import MODEL_CONFIGS
from .callback import CustomCallback
import time

class LLMManager:   
    def __init__(self):
        print("Initializing LLMManager")
        self.callbacks = []
        self.custom_callback = CustomCallback()
        self.callbacks.append(self.custom_callback)
        litellm.callbacks = self.callbacks
        
        try:
            self._set_credentials()
        except Exception as e:
            print(f"Failed to set credentials: {str(e)}")
            raise

    def _set_credentials(self):
        # Set standard API keys
        for provider, config in MODEL_CONFIGS.items():
            if config.get("api_key"):
                os.environ[f"{provider.upper()}_API_KEY"] = config["api_key"]
        
        # Set AWS credentials for Bedrock
        for provider, config in MODEL_CONFIGS.items():
            if provider == "anthropic_aws":
                if config.get("aws_access_key_id"):
                    os.environ["AWS_ACCESS_KEY_ID"] = config["aws_access_key_id"]
                if config.get("aws_secret_access_key"):
                    os.environ["AWS_SECRET_ACCESS_KEY"] = config["aws_secret_access_key"]
                if config.get("aws_region_name"):
                    os.environ["AWS_REGION_NAME"] = config["aws_region_name"]
    
    async def get_completion(
        self,
        provider: str,
        messages: List[Dict[str, str]],
        fallback: bool = True,
        response_format: Optional[Dict[str, Any]] = None,
        stop: Optional[List[str]] = None,
        temperature: Optional[float] = None,
    ) -> ModelResponse:
        """Get completion from LLM provider with improved error handling and logging"""
        print(f"Getting completion from provider: {provider}")
        
        try:
            config = MODEL_CONFIGS[provider]
        except KeyError:
            print(f"Invalid provider: {provider}")
            raise ValueError(f"Provider {provider} not found in MODEL_CONFIGS")

        model = config["model"]
        print(f"Using model: {model}")

        # Build model params with logging
        try:
            model_params = self._build_model_params(config, messages, stop, response_format, temperature)
        except Exception as e:
            print(f"Error building model parameters: {str(e)}")
            raise

        # Primary model attempt
        try:
            print("Sending request to primary model")
            response = await litellm.acompletion(**model_params)
            print("Primary model request successful")
            return response
            
        except OpenAIError as e:
            print(f"Error with primary model: {str(e)}")
            
            # Attempt fallback if enabled and available
            if fallback and "fallback_model" in config:
                return await self._try_fallback(config, model_params, e)
            raise

    async def _try_fallback(self, config: Dict, model_params: Dict, original_error: Exception) -> ModelResponse:
        """Helper method to handle fallback logic"""
        try:
            fallback_model = config["fallback_model"]
            print(f"Attempting fallback to {fallback_model}")
            model_params["model"] = fallback_model
            response = await litellm.acompletion(**model_params)
            print("Fallback request successful")
            return response
            
        except OpenAIError as e:
            print(f"Fallback also failed: {str(e)}")
            # Re-raise original error to maintain error context
            raise original_error

    def _build_model_params(
        self, 
        config: Dict, 
        messages: List, 
        stop: Optional[List[str]], 
        response_format: Optional[Dict],
        temperature: Optional[float] = None,
    ) -> Dict:
        """Helper method to build model parameters"""
        model_name = config["model"]

        # Always include required params
        model_params: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
        }

        # For any GPT-5 family model, do not pass temperature or max_tokens
        is_gpt5_family = "gpt-5" in model_name.lower()
        if is_gpt5_family:
            print(f"Building params for {model_name}: skipping temperature and max_tokens")
        else:
            # Only include max_tokens if explicitly set
            if config.get("max_tokens") is not None:
                model_params["max_tokens"] = config["max_tokens"]

            # Respect explicit temperature arg, otherwise use config if set
            effective_temperature = temperature if temperature is not None else config.get("temperature")
            if effective_temperature is not None:
                model_params["temperature"] = effective_temperature

        if stop:
            model_params["stop"] = stop
            print(f"Using stop sequences: {stop}")

        if response_format:
            model_params["response_format"] = response_format
            print(f"Using response format: {response_format}")

        # Add AWS-specific parameters for Bedrock
        if config.get("aws_access_key_id"):
            model_params.update({
                "aws_access_key_id": config["aws_access_key_id"],
                "aws_secret_access_key": config.get("aws_secret_access_key"),
                "aws_region_name": config.get("aws_region_name")
            })

        # Trace which keys are being sent to the provider for validation
        try:
            print(f"Model params keys for {model_name}: {sorted(list(model_params.keys()))}")
        except Exception:
            # Avoid logging failures from non-serializable types
            pass

        return model_params