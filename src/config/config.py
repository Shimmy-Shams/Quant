"""
Configuration management for the trading system.
Handles secure loading of credentials and settings from .env file.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
import logging


class Config:
    """
    Configuration manager for the trading system.
    Loads and validates environment variables securely.
    """

    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize configuration by loading environment variables.

        Args:
            env_file: Path to .env file. If None, searches in project root.
        """
        self.logger = logging.getLogger(__name__)

        if env_file is None:
            env_file = self._find_env_file()

        if env_file and os.path.exists(env_file):
            load_dotenv(env_file)
            self.logger.info(f"Loaded configuration from {env_file}")
        else:
            self.logger.warning("No .env file found. Using default/environment values.")

        self._load_settings()
        self._validate_config()

    def _find_env_file(self) -> str:
        """
        Find .env file in project root.

        Returns:
            Path to .env file
        """
        current_dir = Path(__file__).resolve().parent
        project_root = current_dir.parent.parent
        return str(project_root / ".env")

    def _load_settings(self):
        """Load all settings from environment variables."""

        # IB Connection Settings
        self.ib_host = os.getenv("IB_HOST", "127.0.0.1")
        self.ib_port = int(os.getenv("IB_PORT", "7497"))
        self.ib_client_id = int(os.getenv("IB_CLIENT_ID", "1"))

        # IB Credentials
        self.ib_username = os.getenv("IB_USERNAME", "")
        self.ib_password = os.getenv("IB_PASSWORD", "")

        # Account Information
        self.ib_account_id = os.getenv("IB_ACCOUNT_ID", "")

        # Trading Mode
        self.trading_mode = os.getenv("TRADING_MODE", "paper").lower()

        # Logging
        self.log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    def _validate_config(self):
        """Validate critical configuration values."""

        # Validate trading mode
        valid_modes = ["paper", "live"]
        if self.trading_mode not in valid_modes:
            raise ValueError(
                f"Invalid TRADING_MODE: {self.trading_mode}. "
                f"Must be one of {valid_modes}"
            )

        # Validate port
        if not (1024 <= self.ib_port <= 65535):
            raise ValueError(
                f"Invalid IB_PORT: {self.ib_port}. "
                "Must be between 1024 and 65535"
            )

        # Warn if credentials are missing
        if not self.ib_username:
            self.logger.warning("IB_USERNAME not set in .env file")

        if not self.ib_password:
            self.logger.warning("IB_PASSWORD not set in .env file")

    @property
    def is_paper_trading(self) -> bool:
        """Check if running in paper trading mode."""
        return self.trading_mode == "paper"

    @property
    def is_live_trading(self) -> bool:
        """Check if running in live trading mode."""
        return self.trading_mode == "live"

    def __repr__(self) -> str:
        """String representation (hides sensitive data)."""
        return (
            f"Config("
            f"host={self.ib_host}, "
            f"port={self.ib_port}, "
            f"client_id={self.ib_client_id}, "
            f"mode={self.trading_mode}, "
            f"username={'***' if self.ib_username else 'NOT_SET'}"
            ")"
        )
