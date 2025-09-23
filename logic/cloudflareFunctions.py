import os
import json
import time
import hmac
import hashlib
import requests
import subprocess
from typing import Optional, Dict, List, Tuple
from urllib.parse import urlparse
from logging_config import setup_logger
from config import (
    CLOUDFLARE_ACCOUNT_ID,
    CLOUDFLARE_API_TOKEN,
    CLOUDFLARE_SIGNATURE_KEY,
    CLOUDFLARE_ACCOUNT_HASH
)

logger = setup_logger(__name__)

# Simple in-memory cache for signed URLs
# Key: original URL, Value: (signed URL, expiry timestamp)
_signed_url_cache: Dict[str, tuple[str, int]] = {}

class CloudflareImageHandler:
    def __init__(self, debug: bool = False):
        self.debug = debug
        self._validate_config()
        self._cache: Dict[str, Tuple[str, int]] = {}  # image_id -> (signed_url, expiry)

    def _validate_config(self) -> None:
        """Validate that all required configuration is present."""
        if not all([CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN, 
                   CLOUDFLARE_SIGNATURE_KEY, CLOUDFLARE_ACCOUNT_HASH]):
            raise ValueError("Missing required Cloudflare configuration.")

    def _debug_print(self, message: str) -> None:
        """Print debug messages if debug mode is enabled."""
        if self.debug:
            logger.debug(message)

    def _check_cache(self, url: str) -> Optional[str]:
        """Check if URL is in cache and still valid."""
        if not url:
            return url
            
        if not _is_valid_cloudflare_url(url):
            return url
            
        if url in _signed_url_cache:
            signed_url, expiry = _signed_url_cache[url]
            if expiry > int(time.time()):
                return signed_url
        return None

    def _update_cache(self, original_url: str, signed_url: str, expiry: int) -> None:
        """Update the cache with a new signed URL."""
        _signed_url_cache[original_url] = (signed_url, expiry)

    def _get_image_id_from_url(self, url: str) -> Optional[str]:
        """Extract image ID from Cloudflare URL."""
        try:
            path_parts = urlparse(url).path.split('/')
            if len(path_parts) >= 2:
                return path_parts[-2]  # Second to last part is the image ID
        except Exception as e:
            logger.error(f"Error extracting image ID from URL {url}: {e}")
        return None

    def generate_signed_url(self, image_id: str, variant: str = "public", expiry_hours: int = 24) -> Optional[str]:
        """Generate a signed URL for a Cloudflare Image."""
        try:
            # Check cache first
            if image_id in self._cache:
                signed_url, expiry = self._cache[image_id]
                if expiry > int(time.time()) + 3600:  # If URL is valid for more than 1 hour
                    return signed_url

            # Calculate expiration timestamp
            expiry = int(time.time()) + (expiry_hours * 3600)
            
            # Construct the path
            path = f"/{CLOUDFLARE_ACCOUNT_HASH}/{image_id}/{variant}"
            
            # Build the string to sign
            string_to_sign = f"{path}?exp={expiry}"
            self._debug_print(f"String to sign: {string_to_sign}")
            
            # Generate HMAC signature
            signature = hmac.new(
                CLOUDFLARE_SIGNATURE_KEY.encode('utf-8'),
                string_to_sign.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            # Construct the final signed URL
            url_path = path.lstrip('/')
            signed_url = f"https://imagedelivery.net/{url_path}?exp={expiry}&sig={signature}"
            
            # Cache the result
            self._cache[image_id] = (signed_url, expiry)
            
            return signed_url

        except Exception as e:
            logger.error(f"Error generating signed URL: {str(e)}")
            return None

    def fetchImage(self, url: Optional[str], expiry_hours: int = 24) -> Optional[str]:
        """Enhanced version of fetchImage that handles caching internally."""
        if not url:
            return url

        # Check cache first
        cached_url = self._check_cache(url)
        if cached_url:
            return cached_url

        # If not in cache and not a Cloudflare URL, return as is
        if not _is_valid_cloudflare_url(url):
            return url

        try:
            image_id = self._get_image_id_from_url(url)
            if not image_id:
                return url

            signed_url = self.generate_signed_url(image_id, expiry_hours=expiry_hours)
            if signed_url:
                self._update_cache(url, signed_url, int(time.time()) + expiry_hours * 3600)
                return signed_url

        except Exception as e:
            logger.error(f"Error in fetchImage: {str(e)}")

        return url

    def fetchImageBatch(self, urls: List[Optional[str]], expiry_hours: int = 24) -> Dict[str, str]:
        """Enhanced version of fetchImageBatch that handles caching internally."""
        result = {}
        if not urls:
            return result

        urls_to_process = []
        
        # First pass: Check cache and collect URLs needing processing
        for url in urls:
            if not url:
                result[url] = None
                continue

            # Check cache first
            cached_url = self._check_cache(url)
            if cached_url:
                result[url] = cached_url
                continue

            # If not a Cloudflare URL, use as is
            if not _is_valid_cloudflare_url(url):
                result[url] = url
                continue

            urls_to_process.append(url)

        # Process uncached URLs in batches
        if urls_to_process:
            try:
                batch_size = 50
                for i in range(0, len(urls_to_process), batch_size):
                    batch = urls_to_process[i:i + batch_size]
                    
                    for url in batch:
                        image_id = self._get_image_id_from_url(url)
                        if not image_id:
                            result[url] = url
                            continue

                        signed_url = self.generate_signed_url(image_id, expiry_hours=expiry_hours)
                        if signed_url:
                            result[url] = signed_url
                            self._update_cache(url, signed_url, int(time.time()) + expiry_hours * 3600)
                        else:
                            result[url] = url

            except Exception as e:
                logger.error(f"Error in batch processing: {str(e)}")
                # On error, return original URLs for remaining items
                for url in urls_to_process:
                    if url not in result:
                        result[url] = url

        return result

    def upload_image(self, image_url: str, require_signed_urls: bool = True) -> Optional[Dict]:
        """Upload an image to Cloudflare Images via URL."""
        if not image_url:
            return None
            
        try:
            # First download the image
            image_response = requests.get(image_url)
            if image_response.status_code != 200:
                logger.error(f"Failed to download image from URL: {image_url}")
                return None

            # Prepare the upload request
            api_url = f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/images/v1"
            headers = {
                "Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}"
            }
            
            # Prepare form data
            files = {
                'file': ('image.jpg', image_response.content),
                'requireSignedURLs': (None, str(require_signed_urls).lower())
            }
            
            # Make the upload request
            response = requests.post(api_url, headers=headers, files=files)
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    # Return in the expected format
                    return {
                        "success": True,
                        "result": {
                            "id": result["result"]["id"],
                            "variants": [f"https://imagedelivery.net/{CLOUDFLARE_ACCOUNT_HASH}/{result['result']['id']}/public"],
                            "requireSignedURLs": require_signed_urls
                        },
                        "errors": [],
                        "messages": []
                    }
                else:
                    logger.error(f"Cloudflare API error: {result.get('errors')}")
            else:
                logger.error(f"Failed to upload image. Status: {response.status_code}, Response: {response.text}")
                
            return None
            
        except Exception as e:
            logger.error(f"Error uploading image to Cloudflare: {str(e)}")
            return None

# Initialize a global instance
_handler = CloudflareImageHandler()

def fetchImage(url: Optional[str], expiry_hours: int = 24) -> Optional[str]:
    """Public interface for fetching a single image URL."""
    return _handler.fetchImage(url, expiry_hours)

def fetchImageBatch(urls: List[Optional[str]], expiry_hours: int = 24) -> Dict[str, str]:
    """Public interface for fetching multiple image URLs."""
    return _handler.fetchImageBatch(urls, expiry_hours)

# Clean up function for cache maintenance
def _clean_cache(cutoff_time: int = None) -> None:
    """Remove expired entries from the URL cache."""
    if cutoff_time is None:
        cutoff_time = int(time.time())
    
    # Clean handler cache
    _handler._cache = {
        image_id: (signed_url, expiry)
        for image_id, (signed_url, expiry) in _handler._cache.items()
        if expiry > cutoff_time
    }
    
    # Clean global cache
    global _signed_url_cache
    _signed_url_cache = {
        url: (signed_url, expiry)
        for url, (signed_url, expiry) in _signed_url_cache.items()
        if expiry > cutoff_time
    }

# Backward compatibility functions
def runCurlCommandToSaveImage(imageURL: str) -> Optional[Dict]:
    """Legacy function for backward compatibility."""
    if not imageURL:
        return None
    
    try:
        image_id = _handler.upload_image(imageURL)
        if not image_id:
            return None
        
        # Match exact format of original response
        return {
            "success": True,
            "result": {
                "id": image_id,
                "variants": [f"https://imagedelivery.net/{CLOUDFLARE_ACCOUNT_HASH}/{image_id}/public"],
                "requireSignedURLs": True
            },
            "errors": [],
            "messages": []
        }
    except Exception as e:
        logger.error(f"Error in runCurlCommandToSaveImage: {str(e)}")
        return None

def _is_valid_cloudflare_url(url: Optional[str]) -> bool:
    """Validates if a URL is a valid Cloudflare image URL."""
    if not url:
        return False
    try:
        parsed_url = urlparse(url)
        return bool(parsed_url.netloc and 'imagedelivery.net' in parsed_url.netloc)
    except Exception:
        return False

def delete_cloudflare_image(image_url: str) -> None:
    """
    Delete image from Cloudflare to prevent orphaned images.
    """
    if not image_url:
        return
        
    try:
        image_id = image_url.split('/')[-2]
        api_url = f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/images/v1/{image_id}"
        headers = {"Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}"}

        response = requests.delete(api_url, headers=headers)
        if response.status_code == 200:
            logger.info(f"Successfully deleted image {image_id}")
        else:
            response_json = response.json()
            errors = response_json.get('errors', [])
            if any(error.get('code') == 5408 for error in errors):
                logger.warning("Cloudflare slow connection error detected, waiting 30 seconds...")
                time.sleep(30)
                retry_response = requests.delete(api_url, headers=headers)
                if retry_response.status_code == 200:
                    logger.info(f"Successfully deleted image {image_id} after retry")
                else:
                    logger.warning(f"Failed to delete image {image_id} after retry. Status: {retry_response.status_code}")
            else:
                logger.warning(f"Failed to delete image {image_id}. Status: {response.status_code}")
    except Exception as e:
        logger.error(f"Error deleting image: {e}")

def upload_image(image_url: str, require_signed_urls: bool = True) -> Optional[Dict]:
    """Public interface for uploading an image to Cloudflare."""
    return _handler.upload_image(image_url, require_signed_urls)

