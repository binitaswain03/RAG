import time
import logging
from functools import wraps
import openai

def retry_with_backoff(retries=3, backoff_in_seconds=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            x = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except (openai.RateLimitError, openai.APITimeoutError) as e:
                    if x == retries:
                        logging.error(f"Max retries reached. Last error: {e}")
                        raise e
                    sleep = (backoff_in_seconds * 2 ** x + 
                             (backoff_in_seconds / 2)) # Add some jitter? No, keep it simple first
                    logging.warning(f"Rate limit or timeout hit. Retrying in {sleep} seconds...")
                    time.sleep(sleep)
                    x += 1
                except Exception as e:
                    logging.error(f"Unexpected error in LLM call: {e}")
                    raise e
        return wrapper
    return decorator
