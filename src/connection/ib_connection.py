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
        # Informational messages (not real errors)
        if errorCode in [2104, 2106, 2107, 2108, 2158]:
            self.logger.info(f"IB Info [{errorCode}]: {errorString}")
        elif errorCode < 1000:
            self.logger.warning(f"IB Warning [{errorCode}]: {errorString}")
        else:
            self.logger.error(f"IB Error [{errorCode}]: {errorString}")

    def _reset_ib_instance(self):
        """Create a fresh IB instance and rebind event handlers."""
        self.ib = IB()
        self._setup_event_handlers()

    def connect(self, timeout: int = 15) -> bool:
        """
        Connect to Interactive Brokers Gateway/TWS.
        Handles existing connections gracefully:
        - If already connected, reuses the existing connection
        - If stale/broken, disconnects and reconnects
        - Tries alternate client IDs if the primary is in use

        Args:
            timeout: Connection timeout in seconds

        Returns:
            True if connection successful, False otherwise
        """
        # Already connected -- reuse
        if self.is_connected:
            self.logger.info("Already connected -- reusing existing connection")
            return True

        # Internal state says connected but IB says no -- clean up stale state
        if self._connected and not self.ib.isConnected():
            self.logger.info("Stale connection detected -- cleaning up")
            try:
                self.ib.disconnect()
            except Exception:
                pass
            self._connected = False
            self._reset_ib_instance()

        # Attempt connection with client ID fallback
        client_ids = [self.config.ib_client_id]
        for offset in [1, 2]:
            alt = self.config.ib_client_id + offset
            if alt not in client_ids:
                client_ids.append(alt)

        last_error = None
        for i, cid in enumerate(client_ids):
            try:
                self.logger.info(
                    f"Connecting to {self.config.ib_host}:{self.config.ib_port} "
                    f"(Client ID: {cid}, Mode: {self.config.trading_mode})"
                )

                # Use util.patchAsyncio() to fix Jupyter event loop conflicts
                util.patchAsyncio()

                self.ib.connect(
                    host=self.config.ib_host,
                    port=self.config.ib_port,
                    clientId=cid,
                    timeout=timeout,
                )

                # Allow time for full synchronization
                self.ib.sleep(2)

                if self.ib.isConnected():
                    self._connected = True
                    if cid != self.config.ib_client_id:
                        self.logger.info(f"Connected using fallback Client ID {cid}")
                    self.logger.info("Connection established successfully")
                    return True

            except Exception as e:
                last_error = e
                # Check if connection actually succeeded despite the exception
                try:
                    self.ib.sleep(1)
                except Exception:
                    pass

                if self.ib.isConnected():
                    self._connected = True
                    self.logger.info("Connection established successfully (recovered)")
                    return True

                self.logger.warning(f"Client ID {cid} failed: {str(e)}")

                # Clean up and create fresh IB instance before next attempt
                try:
                    self.ib.disconnect()
                except Exception:
                    pass
                self._connected = False
                self._reset_ib_instance()
                continue

        self.logger.error(f"Connection failed after trying client IDs {client_ids}: {last_error}")
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
            # Check if cached values are available
            account_values = self.ib.accountValues()

            # If cache is empty, explicitly request account updates and wait
            if not account_values:
                accounts = self.ib.managedAccounts()
                if accounts:
                    self.ib.reqAccountUpdates(True, accounts[0])
                    self.ib.sleep(3)  # Wait for data to arrive
                    account_values = self.ib.accountValues()

            result = {}
            for item in account_values:
                key = f"{item.tag}_{item.currency}" if item.currency else item.tag
                result[key] = item.value
                # Also store without currency suffix for flexible lookup
                result[item.tag] = item.value

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
            
            # Try serverVersion as both method and property (varies by ib_insync version)
            try:
                result["server_version"] = self.ib.serverVersion()
            except (TypeError, AttributeError):
                try:
                    result["server_version"] = self.ib.client.serverVersion
                except Exception:
                    result["server_version"] = "unknown"
            
            try:
                result["connection_time"] = self.ib.connectionTime()
            except Exception:
                result["connection_time"] = datetime.now().isoformat()

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
