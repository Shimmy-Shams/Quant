#!/usr/bin/env python3
"""Quick check of Alpaca order/position state."""
import sys; sys.path.insert(0, "src")
from connection.alpaca_connection import AlpacaConnection

from alpaca.trading.requests import GetOrdersRequest, QueryOrderStatus
from alpaca.trading.enums import OrderSide

conn = AlpacaConnection()
tc = conn.trading_client

# Check open orders
req = GetOrdersRequest(status=QueryOrderStatus.OPEN)
orders = tc.get_orders(req)
print("=== OPEN ORDERS ===")
for o in orders:
    print(f"  {o.side.value} {o.qty} {o.symbol} type={o.type.value} limit={o.limit_price} stop={o.stop_price} status={o.status.value} submitted={o.submitted_at}")
if not orders:
    print("  (none)")

# Check recent closed orders
req2 = GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=15)
orders2 = tc.get_orders(req2)
print("\n=== RECENT CLOSED ORDERS ===")
for o in orders2:
    print(f"  {o.side.value} {o.qty} {o.symbol} type={o.type.value} limit={o.limit_price} stop={o.stop_price} status={o.status.value} filled_avg={o.filled_avg_price} filled_at={o.filled_at}")

# Check positions
positions = tc.get_all_positions()
print("\n=== POSITIONS ===")
for p in positions:
    print(f"  {p.symbol}: {p.qty} shares @ avg={p.avg_entry_price} mkt={p.current_price} pnl={p.unrealized_pl} side={p.side.value}")

# Account
acct = tc.get_account()
print(f"\nAccount: ${float(acct.portfolio_value):,.2f} | Cash: ${float(acct.cash):,.2f}")
