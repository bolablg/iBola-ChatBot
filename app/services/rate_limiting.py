"""
Rate limiting service for API protection.
"""

import asyncio
import time
from typing import Dict, Tuple, Optional, Any
import sys
import os
from collections import defaultdict, deque

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from cachetools import TTLCache
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False


class RateLimiter:
    """Advanced rate limiting with multiple strategies."""

    def __init__(self):
        # Rate limit configurations
        self.global_limits = {
            "requests_per_minute": 60,
            "requests_per_hour": 1000,
            "burst_limit": 10
        }

        self.endpoint_limits = {
            "/chat": {"per_minute": 30, "per_hour": 500},
            "/welcome": {"per_minute": 10, "per_hour": 100},
            "/health": {"per_minute": 60, "per_hour": 1000},
            "/session": {"per_minute": 20, "per_hour": 200}
        }

        # Sliding window storage
        self.request_windows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Block list for abusive clients
        self.blocked_clients: Dict[str, float] = {}

        # Cache for faster lookups
        if CACHE_AVAILABLE:
            self.cache = TTLCache(maxsize=10000, ttl=3600)  # 1 hour TTL
        else:
            self.cache = None

    def _get_client_key(self, client_ip: str, endpoint: str) -> str:
        """Generate a unique key for rate limiting."""
        return f"{client_ip}:{endpoint}"

    def _get_global_key(self, client_ip: str) -> str:
        """Generate a global key for rate limiting."""
        return f"global:{client_ip}"

    def _cleanup_old_requests(self, client_key: str, window_seconds: int):
        """Clean up requests outside the time window."""
        current_time = time.time()
        window = self.request_windows[client_key]

        # Remove requests outside the time window
        while window and current_time - window[0] > window_seconds:
            window.popleft()

    def _is_blocked(self, client_ip: str) -> bool:
        """Check if client is currently blocked."""
        if client_ip in self.blocked_clients:
            block_expiry = self.blocked_clients[client_ip]
            if time.time() < block_expiry:
                return True
            else:
                # Block expired, remove from block list
                del self.blocked_clients[client_ip]
        return False

    def _block_client(self, client_ip: str, duration_seconds: int = 300):
        """Block a client for a specified duration."""
        self.blocked_clients[client_ip] = time.time() + duration_seconds

    async def check_rate_limit(self, client_ip: str, endpoint: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if request is within rate limits.

        Returns:
            Tuple[bool, Dict]: (allowed, metadata)
        """
        if self._is_blocked(client_ip):
            return False, {
                "blocked": True,
                "reason": "client_blocked",
                "retry_after": int(self.blocked_clients[client_ip] - time.time())
            }

        current_time = time.time()
        client_key = self._get_client_key(client_ip, endpoint)
        global_key = self._get_global_key(client_ip)

        # Get endpoint-specific limits
        endpoint_config = self.endpoint_limits.get(endpoint, self.endpoint_limits.get("/chat", {}))
        global_config = self.global_limits

        # Check endpoint limits
        endpoint_allowed, endpoint_info = self._check_endpoint_limit(
            client_key, endpoint_config, current_time
        )

        # Check global limits
        global_allowed, global_info = self._check_global_limit(
            global_key, global_config, current_time
        )

        # Record the request if allowed
        if endpoint_allowed and global_allowed:
            self.request_windows[client_key].append(current_time)
            self.request_windows[global_key].append(current_time)

            return True, {
                "endpoint_requests": endpoint_info["current_requests"],
                "global_requests": global_info["current_requests"],
                "endpoint_limit": endpoint_config.get("per_minute", 30),
                "global_limit": global_config["requests_per_minute"]
            }

        # Determine the most restrictive limit
        if not endpoint_allowed:
            return False, {
                "blocked": False,
                "reason": "endpoint_limit_exceeded",
                "retry_after": endpoint_info.get("retry_after", 60),
                "limit": endpoint_config.get("per_minute", 30)
            }

        if not global_allowed:
            return False, {
                "blocked": False,
                "reason": "global_limit_exceeded",
                "retry_after": global_info.get("retry_after", 60),
                "limit": global_config["requests_per_minute"]
            }

        return False, {"reason": "unknown_limit_exceeded"}

    def _check_endpoint_limit(self, client_key: str, config: Dict, current_time: float) -> Tuple[bool, Dict]:
        """Check endpoint-specific rate limits."""
        per_minute = config.get("per_minute", 30)
        per_hour = config.get("per_hour", 500)

        # Clean up old requests
        self._cleanup_old_requests(client_key, 60)  # 1 minute window
        self._cleanup_old_requests(f"{client_key}_hour", 3600)  # 1 hour window

        # Check minute limit
        minute_window = self.request_windows[client_key]
        if len(minute_window) >= per_minute:
            oldest_request = minute_window[0]
            retry_after = 60 - (current_time - oldest_request)
            return False, {
                "retry_after": max(1, int(retry_after)),
                "current_requests": len(minute_window)
            }

        # Check hour limit
        hour_key = f"{client_key}_hour"
        self._cleanup_old_requests(hour_key, 3600)
        hour_window = self.request_windows[hour_key]
        if len(hour_window) >= per_hour:
            oldest_request = hour_window[0]
            retry_after = 3600 - (current_time - oldest_request)
            return False, {
                "retry_after": max(1, int(retry_after)),
                "current_requests": len(hour_window)
            }

        return True, {"current_requests": len(minute_window)}

    def _check_global_limit(self, global_key: str, config: Dict, current_time: float) -> Tuple[bool, Dict]:
        """Check global rate limits."""
        per_minute = config.get("requests_per_minute", 60)
        per_hour = config.get("requests_per_hour", 1000)
        burst_limit = config.get("burst_limit", 10)

        # Clean up old requests
        self._cleanup_old_requests(global_key, 60)
        self._cleanup_old_requests(f"{global_key}_burst", 10)  # 10 second burst window

        # Check burst limit (requests in last 10 seconds)
        burst_key = f"{global_key}_burst"
        burst_window = self.request_windows[burst_key]
        if len(burst_window) >= burst_limit:
            return False, {
                "retry_after": 10,
                "current_requests": len(burst_window)
            }

        # Check minute limit
        minute_window = self.request_windows[global_key]
        if len(minute_window) >= per_minute:
            oldest_request = minute_window[0]
            retry_after = 60 - (current_time - oldest_request)
            return False, {
                "retry_after": max(1, int(retry_after)),
                "current_requests": len(minute_window)
            }

        return True, {"current_requests": len(minute_window)}

    def get_client_stats(self, client_ip: str) -> Dict[str, Any]:
        """Get rate limiting statistics for a client."""
        stats = {
            "client_ip": client_ip,
            "blocked": self._is_blocked(client_ip),
            "endpoints": {}
        }

        if stats["blocked"]:
            stats["block_expires"] = self.blocked_clients[client_ip]

        # Get stats for each endpoint
        for endpoint in self.endpoint_limits.keys():
            client_key = self._get_client_key(client_ip, endpoint)
            window = self.request_windows[client_key]
            stats["endpoints"][endpoint] = {
                "current_requests": len(window),
                "limit_per_minute": self.endpoint_limits[endpoint]["per_minute"]
            }

        return stats

    def reset_client(self, client_ip: str):
        """Reset all rate limits for a client."""
        # Remove from block list
        if client_ip in self.blocked_clients:
            del self.blocked_clients[client_ip]

        # Clear all request windows for this client
        keys_to_remove = []
        for key in self.request_windows:
            if client_ip in key:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.request_windows[key]

    def get_global_stats(self) -> Dict[str, Any]:
        """Get global rate limiting statistics."""
        total_requests = sum(len(window) for window in self.request_windows.values())
        blocked_clients = len(self.blocked_clients)

        return {
            "total_request_windows": len(self.request_windows),
            "total_requests_tracked": total_requests,
            "blocked_clients": blocked_clients,
            "endpoint_limits": self.endpoint_limits,
            "global_limits": self.global_limits
        }


# Global rate limiter instance
rate_limiter = RateLimiter()
