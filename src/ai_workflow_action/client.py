"""
Global Anthropic Client Manager.
Provides singleton access to Anthropic API client.
"""

from typing import Optional
from anthropic import Anthropic
import os


class AnthropicClientManager:
    """Global Anthropic Client Manager"""

    _instance: Optional[Anthropic] = None

    @classmethod
    def get_client(cls, api_key: Optional[str] = None) -> Anthropic:
        """
        Get global Anthropic Client instance (singleton pattern).

        Args:
            api_key: Optional API key. If not provided, uses env var.

        Returns:
            Anthropic client instance
        """
        if cls._instance is None:
            key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment")
            cls._instance = Anthropic(api_key=key)
        return cls._instance

    @classmethod
    def reset_client(cls) -> None:
        """Reset client (for testing)"""
        cls._instance = None
