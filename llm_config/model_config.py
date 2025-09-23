from typing import Dict, Any
from config import get_env_var

# Model configurations
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "openai4o": {
        "model": "gpt-4o",
        "fallback_model": "gpt-4o-mini",
        "api_key": get_env_var("OPENAI_API_KEY"),
        "max_tokens": 4096,
        "temperature": 0,
        "allowed_fails": 3,
        "cooldown_time": 60
    },
    "openainano": {
        "model": "gpt-4.1-nano-2025-04-14",
        "fallback_model": "gpt-4o-mini",
        "api_key": get_env_var("OPENAI_API_KEY"),
        "max_tokens": 8000,
        "temperature": 0,
        "allowed_fails": 3,
        "cooldown_time": 60
    },
    "openai4o_mini": {
        "model": "gpt-4o-mini",
        "fallback_model": "gpt-4o",
        "api_key": get_env_var("OPENAI_API_KEY"),
        "max_tokens": 4096,
        "temperature": 0,
        "allowed_fails": 3,
        "cooldown_time": 60
    },
    "anthropic_sonnet": {
        "model": "claude-3-5-sonnet-20240620",
        "fallback_model": "claude-3-5-haiku-20241022",
        "api_key": get_env_var("ANTHROPIC_API_KEY"),
        "max_tokens": 4096,
        "temperature": 0,
        "allowed_fails": 3,
        "cooldown_time": 60
    },
    "anthropic_haiku": {
        "model": "claude-3-5-haiku-20241022",
        "fallback_model": "claude-3-5-sonnet-20240620",
        "api_key": get_env_var("ANTHROPIC_API_KEY"),
        "max_tokens": 2000,
        "temperature": 0,
        "allowed_fails": 3,
        "cooldown_time": 60
    },
    "anthropic_aws": {
        "model": "anthropic.claude-3-5-sonnet-20240620-v1:0",
        "fallback_model": "anthropic.claude-3-haiku-20240307-v1:0",
        "aws_access_key_id": get_env_var("R2_ACCESS_KEY_ID"),
        "aws_secret_access_key": get_env_var("R2_SECRET_ACCESS_KEY"),
        "aws_region_name": get_env_var("R2_REGION"),
        "max_tokens": 4096,
        "temperature": 0,
        "allowed_fails": 3,
        "cooldown_time": 60
    },
    "gemini": {
        "model": "gemini/gemini-2.0-flash",
        # "model": "gemini/gemini-2.5-flash",
        "fallback_model": "claude-3-5-haiku-20241022",
        "api_key": get_env_var("GEMINI_API_KEY"),
        "max_tokens": 10000,
        "temperature": 0,
        "allowed_fails": 3,
        "cooldown_time": 60
    },
    "groq": {
        "model": "groq/gemma2-9b-it",
        "fallback_model": "gemini/gemini-2.0-flash",
        "api_key": get_env_var("GROQ_API_KEY"),
        "max_tokens": 4000,
        "temperature": 0,
        "allowed_fails": 3,
        "cooldown_time": 60
    },
    "groq_qwen": {
        "model": "groq/qwen/qwen3-32b",
        "fallback_model": "claude-3-5-haiku-20241022",
        "api_key": get_env_var("GROQ_API_KEY"),
        "max_tokens": 4000,
        "temperature": 0,
        "allowed_fails": 3,
        "cooldown_time": 60
    },
    "groq_gemma": {
        "model": "groq/gemma2-9b-it",
        "fallback_model": "claude-3-5-haiku-20241022",
        "api_key": get_env_var("GROQ_API_KEY"),
        "max_tokens": 4000,
        "temperature": 0,
        "allowed_fails": 3,
        "cooldown_time": 60
    },
    "groq_oss": {
        "model": "groq/openai/gpt-oss-120b",
        "fallback_model": "claude-3-5-haiku-20241022",
        "api_key": get_env_var("GROQ_API_KEY"),
        "max_tokens": 4000,
        "temperature": 0,
        "allowed_fails": 3,
        "cooldown_time": 60
    },
    "groq_deepseek_r": {
        "model": "groq/deepseek-r1-distill-llama-70b",
        "fallback_model": "groq/llama-3.3-70b-versatile",
        "api_key": get_env_var("GROQ_API_KEY"),
        "max_tokens": 4096,
        # "temperature": 0,
        "allowed_fails": 3,
        "cooldown_time": 60
    },
    "mistral_large": {
        "model": "mistral/mistral-large-latest",
        "fallback_model": "claude-3-5-haiku-20241022",
        "api_key": get_env_var("MISTRAL_API_KEY"),
        "max_tokens": 4096,
        "temperature": 0,
        "allowed_fails": 3,
        "cooldown_time": 60
    },
    "ministral": {
        "model": "mistral/ministral-8b-latest",
        "fallback_model": "claude-3-5-haiku-20241022",
        "api_key": get_env_var("MISTRAL_API_KEY"),
        "max_tokens": 4096,
        "temperature": 0,
        "allowed_fails": 3,
        "cooldown_time": 60
    },
    "deepseek": {
        "model": "deepseek/deepseek-chat",
        "fallback_model": "claude-3-5-haiku-20241022",
        "api_key": get_env_var("DEEPSEEK_API_KEY"),
        "max_tokens": 4096,
        # "temperature": 0,
        "allowed_fails": 3,
        "cooldown_time": 60
    },
    "deepseek-r": {
        "model": "deepseek/deepseek-reasoner",
        "fallback_model": "claude-3-5-sonnet-20240620",
        "api_key": get_env_var("DEEPSEEK_API_KEY"),
        "max_tokens": 6000,
        # "temperature": 1,
        "allowed_fails": 3,
        "cooldown_time": 60
    },
    "together-deepseek": {
        "model": "together_ai/deepseek-ai/DeepSeek-V3",
        "fallback_model": "claude-3-5-haiku-20241022",
        "api_key": get_env_var("TOGETHERAI_API_KEY"),
        "max_tokens": 8000,
        "temperature": 1,
        "allowed_fails": 3,
        "cooldown_time": 60
    },
     "azure-gpt-4.1-mini": {
        "model": "azure/gpt-4.1-mini",
        "fallback_model": "gemini/gemini-2.0-flash",
        "max_tokens": 8000,
        "temperature": 0.8,
        "allowed_fails": 3,
        "cooldown_time": 60
    },
     "azure-gpt-5-mini": {
        "model": "azure/gpt-5-mini",
        "fallback_model": "gemini/gemini-2.0-flash",
        "allowed_fails": 3,
        "cooldown_time": 60
    },
     "azure-gpt-5-nano": {
        "model": "azure/gpt-5-nano",
        "fallback_model": "gemini/gemini-2.0-flash",
        "allowed_fails": 3,
        "cooldown_time": 60
    }
}
# Callback configurations with empty default
ENABLED_CALLBACKS = get_env_var("ENABLED_CALLBACKS", required=False).split(",") if get_env_var("ENABLED_CALLBACKS", required=False) else []