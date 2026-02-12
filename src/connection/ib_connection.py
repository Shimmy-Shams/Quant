"""
Interactive Brokers connection manager.
Handles connection lifecycle, error handling, and basic account operations.
"""

import logging
from typing import Optional, List
from datetime import datetime

from ib_insync import IB, Contract, Stock, util
import asyncio

# Try absolute import first, fall back to relative import
try:
    from config.config import Config
except ImportError:
    from ..config.config import Config


class IBConnection:
    """
    Manages connection to Interactive Brokers Gateway/TWS.
    Provides methods for connecting, disconnecting, and basic account operations.
    """

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize IB connection manager.

        Args:
            config: Configuration object. If None, creates new Config instance.
        """
        self.config = config or Config()
        self.ib = IB()
        self.logger = logging.getLogger(__name__)

        # Connection state
        self._connected = False

        # Set up event handlers
        self._setup_event_handlers()

    def _setup_event_handlers(self):
        """Set up event handlers for IB connection events."""

        self.ib.connectedEvent += self._on_connected
        self.ib.disconnectedEvent += self._on_disconnected
        self.ib.errorEvent += self._on_error

    def _on_connected(self):
        """Called when connection is established."""
        self._connected = True
        self.logger.info("Successfully connected to IB Gateway/TWS")

    def _on_disconnected(self):
        """Called when connection is lost."""
        self._connected = False
        self.logger.warning("Disconnected from IB Gateway/TWS")

    def _on_error(self, reqId, errorCode, errorString, contract):
        """
        Called when an error occurs.

        Args:
            reqId: Request ID
            errorCode: Error code
            errorString: Error description
            contract: Related contract (if applicable)
        """
        if errorCode in [2104, 2106, 2158]:
            self.logger.info(f"IB Info [{errorCode}]: {errorString}")
        elif errorCode < 1000:
            self.logger.warning(f"IB Warning [{errorCode}]: {errorString}")
        else:
            self.logger.error(f"IB Error [{errorCode}]: {errorString}")

    def connect(self, timeout: int = 10) -> bool:
        """
        Connect to Interactive Brokers Gateway/TWS.

        Args:
            timeout: Connection timeout in seconds

        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.logger.info(
                f"Connecting to IB Gateway at {self.config.ib_host}:{self.config.ib_port} "
                f"(Client ID: {self.config.ib_client_id}, Mode: {self.config.trading_mode})"
            )

            self.ib.connect(
                host=self.config.ib_host,
                port=self.config.ib_port,
                clientId=self.config.ib_client_id,
                timeout=timeout,
            )

            if self.ib.isConnected():
                self._connected = True
                self.logger.info("Connection established successfully")
                return True
            else:
                self.logger.error("Failed to establish connection")
                return False

        except Exception as e:
            self.logger.error(f"Connection error: {str(e)}")
            self._connected = False
            return False

    def disconnect(self):
        """Disconnect from Interactive Brokers Gateway/TWS."""
        if self._connected and self.ib.isConnected():
            self.logger.info("Disconnecting from IB Gateway/TWS")
            self.ib.disconnect()
            self._connected = False
        else:
            self.logger.warning("Already disconnected")

    @property
    def is_connected(self) -> bool:
        """Check if currently connected to IB."""
        return self._connected and self.ib.isConnected()

    def get_account_summary(self) -> List:
        """
        Get account summary information.

        Returns:
            List of account summary items
        """
        if not self.is_connected:
            self.logger.error("Not connected to IB. Call connect() first.")
            return []

        try:
            account_values = self.ib.accountSummary()
            self.logger.info(f"Retrieved {len(account_values)} account summary items")
            return account_values
        except Exception as e:
            self.logger.error(f"Error getting account summary: {str(e)}")
            return []

    def get_positions(self) -> List:
        """
        Get current positions.

        Returns:
            List of positions
        """
        if not self.is_connected:
            self.logger.error("Not connected to IB. Call connect() first.")
            return []

        try:
            positions = self.ib.positions()
            self.logger.info(f"Retrieved {len(positions)} positions")
            return positions
        except Exception as e:
            self.logger.error(f"Error getting positions: {str(e)}")
            return []

    def get_portfolio_items(self) -> List:
        """
        Get portfolio items.

        Returns:
            List of portfolio items
        """
        if not self.is_connected:
            self.logger.error("Not connected to IB. Call connect() first.")
            return []

        try:
            portfolio = self.ib.portfolio()
            self.logger.info(f"Retrieved {len(portfolio)} portfolio items")
            return portfolio
        except Exception as e:
            self.logger.error(f"Error getting portfolio: {str(e)}")
            return []

    def get_account_values(self) -> dict:
        """
        Get key account values as a dictionary.

        Returns:
            Dictionary of account values
        """
        if not self.is_connected:
            self.logger.error("Not connected to IB. Call connect() first.")
            return {}

        try:
            account_values = self.ib.accountValues()
            result = {}

            for item in account_values:
                key = f"{item.tag}_{item.currency}" if item.currency else item.tag
                result[key] = item.value

            self.logger.info(f"Retrieved {len(result)} account values")
            return result

        except Exception as e:
            self.logger.error(f"Error getting account values: {str(e)}")
            return {}

    def test_connection(self) -> dict:
        """
        Test the connection and return status information.

        Returns:
            Dictionary with connection test results
        """
        result = {
            "timestamp": datetime.now().isoformat(),
            "connected": False,
            "server_version": None,
            "connection_time": None,
            "account_count": 0,
            "positions_count": 0,
        }

        if not self.is_connected:
            self.logger.error("Connection test failed: Not connected")
            return result

        try:
            result["connected"] = True
            result["server_version"] = self.ib.serverVersion()
            result["connection_time"] = self.ib.connectionTime()

            accounts = self.ib.managedAccounts()
            result["account_count"] = len(accounts)
            result["accounts"] = accounts

            positions = self.get_positions()
            result["positions_count"] = len(positions)

            self.logger.info("Connection test successful")
            return result

        except Exception as e:
            self.logger.error(f"Connection test error: {str(e)}")
            result["error"] = str(e)
            return result

    def __enter__(self):
        """Context manager entry."""
        if not self.is_connected:
            self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()

    def __repr__(self) -> str:
        """String representation."""
        status = "Connected" if self.is_connected else "Disconnected"
        return (
            f"IBConnection("
            f"status={status}, "
            f"host={self.config.ib_host}:{self.config.ib_port}, "
            f"mode={self.config.trading_mode}"
            ")"
        )
