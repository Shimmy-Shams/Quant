"""
Test script for Interactive Brokers connection.
Run this to verify your IB Gateway/TWS connection is working.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from connection.ib_connection import IBConnection
from config.config import Config
import colorlog


def setup_logging():
    """Set up colored logging."""
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
            datefmt=None,
            reset=True,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )
    )

    logger = colorlog.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_account_summary(ib_conn: IBConnection):
    """Print account summary information."""
    print_section("Account Summary")

    account_values = ib_conn.get_account_values()

    if account_values:
        # Display key account metrics
        key_metrics = [
            "NetLiquidation",
            "TotalCashValue",
            "GrossPositionValue",
            "BuyingPower",
            "AvailableFunds",
        ]

        print(f"\n{'Metric':<30} {'Value':>20}")
        print("-" * 52)

        for metric in key_metrics:
            # Try to find the metric with USD suffix first
            value = account_values.get(f"{metric}_USD")
            if value is None:
                value = account_values.get(metric, "N/A")

            if value != "N/A":
                try:
                    value = f"${float(value):,.2f}"
                except:
                    pass

            print(f"{metric:<30} {value:>20}")
    else:
        print("No account data available")


def print_positions(ib_conn: IBConnection):
    """Print current positions."""
    print_section("Current Positions")

    positions = ib_conn.get_positions()

    if positions:
        print(f"\nTotal Positions: {len(positions)}\n")
        print(f"{'Symbol':<10} {'Quantity':>10} {'Avg Cost':>12} {'Market Price':>15} {'Market Value':>15}")
        print("-" * 65)

        for pos in positions:
            symbol = pos.contract.symbol
            quantity = pos.position
            avg_cost = pos.avgCost
            market_price = getattr(pos, "marketPrice", 0) or 0
            market_value = getattr(pos, "marketValue", 0) or 0

            print(
                f"{symbol:<10} {quantity:>10.2f} ${avg_cost:>10.2f} "
                f"${market_price:>13.2f} ${market_value:>13.2f}"
            )
    else:
        print("\nNo open positions")


def test_connection():
    """Main test function."""
    setup_logging()

    print_section("IB Gateway Connection Test")

    # Load configuration
    print("\n1. Loading configuration...")
    config = Config()
    print(f"   {config}")

    # Create connection
    print("\n2. Creating connection object...")
    ib_conn = IBConnection(config)
    print(f"   {ib_conn}")

    # Attempt connection
    print("\n3. Connecting to IB Gateway/TWS...")
    print(f"   Host: {config.ib_host}")
    print(f"   Port: {config.ib_port}")
    print(f"   Client ID: {config.ib_client_id}")
    print(f"   Trading Mode: {config.trading_mode.upper()}")

    success = ib_conn.connect()

    if not success:
        print("\n❌ CONNECTION FAILED")
        print("\nTroubleshooting steps:")
        print("1. Ensure IB Gateway or TWS is running")
        print("2. Check that API access is enabled in IB settings:")
        print("   - Go to Configure -> Settings -> API -> Settings")
        print("   - Enable 'Enable ActiveX and Socket Clients'")
        print("   - Check 'Read-Only API' for paper trading")
        print(f"3. Verify the port matches your .env file: {config.ib_port}")
        print("   - TWS Paper Trading: port 7497")
        print("   - TWS Live Trading: port 7496")
        print("   - IB Gateway Paper: port 4002")
        print("   - IB Gateway Live: port 4001")
        print("4. Check firewall settings")
        sys.exit(1)

    print("\n✅ CONNECTION SUCCESSFUL")

    # Test connection and get info
    print("\n4. Testing connection and retrieving information...")
    test_result = ib_conn.test_connection()

    print(f"\n   Server Version: {test_result.get('server_version')}")
    print(f"   Connection Time: {test_result.get('connection_time')}")
    print(f"   Managed Accounts: {test_result.get('accounts')}")

    # Display account summary
    try:
        print_account_summary(ib_conn)
    except Exception as e:
        print(f"\n⚠️  Could not retrieve account summary: {str(e)}")

    # Display positions
    try:
        print_positions(ib_conn)
    except Exception as e:
        print(f"\n⚠️  Could not retrieve positions: {str(e)}")

    # Disconnect
    print_section("Disconnecting")
    ib_conn.disconnect()
    print("\n✅ Test completed successfully!\n")


if __name__ == "__main__":
    try:
        test_connection()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
