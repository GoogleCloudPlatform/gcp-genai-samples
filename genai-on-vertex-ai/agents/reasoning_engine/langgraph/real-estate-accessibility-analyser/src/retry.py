import time
import logging
from typing import Callable, Any

logger = logging.getLogger(__name__)

class RetryError(Exception):
    """Custom exception raised when retries are exhausted."""

def retry_with_exponential_backoff(func: Callable) -> Callable:
    """
    Decorator that applies exponential backoff retry logic.
    """
    def wrapper(*args, **kwargs) -> Any:
        self = args[0]
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                wait_time = self.retry_delay * (2 ** attempt)
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
        raise RetryError(f"Failed after {self.max_retries} attempts. Last error: {last_exception}")
    return wrapper
