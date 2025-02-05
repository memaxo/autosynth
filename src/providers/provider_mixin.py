import logging
import asyncio

class ProviderMixin:
    """
    Mixin class providing common functionality for search providers.

    This mixin includes:
    - A standard method to acquire a rate limiter token.
    - Standardized error handling that logs errors and re‑raises exceptions.
    - Stubs for a caching mechanism if needed.

    Providers inheriting from both BaseProvider and ProviderMixin must ensure that
    self.rate_limiter is defined.
    """
    logger = logging.getLogger(__name__)

    async def _acquire_rate_limit(self):
        """Acquire a token from the rate limiter if available."""
        if hasattr(self, 'rate_limiter') and self.rate_limiter:
            await self.rate_limiter.acquire()
        else:
            self.logger.warning("No rate limiter found; skipping rate limit check.")

    def _handle_error(self, error: Exception, query: str):
        """Log the error and re‑raise it."""
        self.logger.error(f"Error during API call for query '{query}': {str(error)}")
        raise error

    def _get_cached(self, key: str):
        """Stub method for retrieving a cached response; override if needed."""
        return None

    def _cache_response(self, key: str, response: any):
        """Stub method for caching a response; override if needed."""
        pass