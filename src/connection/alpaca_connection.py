"""
Alpaca Markets Connection Manager

Handles connection lifecycle, authentication, and account operations
for both paper trading and live trading via Alpaca API.

Uses alpaca-py SDK (official v2).
"""

import logging
from typing import Optional, Dict, List
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest, LimitOrderRequest,
    GetOrdersRequest, ClosePositionRequest,
    StopLossRequest, TakeProfitRequest,
)
from alpaca.trading.enums import (
    OrderSide, TimeInForce, OrderStatus, OrderType,
    QueryOrderStatus, OrderClass,
)
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestBarRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.common.exceptions import APIError

logger = logging.getLogger(__name__)


class TradingMode(Enum):
    """Trading execution modes"""
    LIVE = "live"               # Submit real orders to Alpaca paper/live account
    SHADOW = "shadow"           # Generate signals + log hypothetical trades, no orders  
    REPLAY = "replay"           # Replay historical data day-by-day through pipeline


@dataclass
class AlpacaConfig:
    """Alpaca connection configuration"""
    api_key: str = ""
    secret_key: str = ""
    paper: bool = True                      # True = paper trading, False = live
    trading_mode: TradingMode = TradingMode.SHADOW  # Default to shadow (safe)
    max_api_calls_per_min: int = 200        # Free tier limit
    data_feed: str = "iex"                  # 'iex' (free) or 'sip' ($99/mo)

    @classmethod
    def from_env(cls, env_path: Optional[Path] = None) -> 'AlpacaConfig':
        """Load config from .env file"""
        import os
        from dotenv import load_dotenv

        if env_path:
            load_dotenv(env_path)
        else:
            # Search up from src/ to find .env in project root
            project_root = Path(__file__).parent.parent.parent
            load_dotenv(project_root / '.env')

        return cls(
            api_key=os.getenv('ALPACA_API_KEY', ''),
            secret_key=os.getenv('ALPACA_SECRET_KEY', ''),
            paper=os.getenv('ALPACA_PAPER', 'true').lower() == 'true',
            trading_mode=TradingMode(os.getenv('ALPACA_TRADING_MODE', 'shadow')),
            data_feed=os.getenv('ALPACA_DATA_FEED', 'iex'),
        )


class AlpacaConnection:
    """
    Manages connection to Alpaca Markets API.
    
    Provides:
    - Trading client (orders, positions, account)
    - Data client (historical bars, latest quotes)
    - Account status and buying power queries
    - Market hours detection
    """

    def __init__(self, config: Optional[AlpacaConfig] = None):
        """
        Initialize Alpaca connection.

        Args:
            config: AlpacaConfig instance. If None, loads from .env
        """
        self.config = config or AlpacaConfig.from_env()
        self._trading_client: Optional[TradingClient] = None
        self._data_client: Optional[StockHistoricalDataClient] = None
        self.logger = logging.getLogger(__name__)

    @property
    def trading_client(self) -> TradingClient:
        """Lazy-initialized trading client"""
        if self._trading_client is None:
            self._trading_client = TradingClient(
                api_key=self.config.api_key,
                secret_key=self.config.secret_key,
                paper=self.config.paper
            )
            self.logger.info(
                f"Trading client initialized (paper={self.config.paper}, "
                f"mode={self.config.trading_mode.value})"
            )
        return self._trading_client

    @property
    def data_client(self) -> StockHistoricalDataClient:
        """Lazy-initialized market data client"""
        if self._data_client is None:
            self._data_client = StockHistoricalDataClient(
                api_key=self.config.api_key,
                secret_key=self.config.secret_key
            )
            self.logger.info("Data client initialized")
        return self._data_client

    @property
    def crypto_data_client(self) -> CryptoHistoricalDataClient:
        """Lazy-initialized crypto market data client"""
        if not hasattr(self, '_crypto_data_client') or self._crypto_data_client is None:
            self._crypto_data_client = CryptoHistoricalDataClient(
                api_key=self.config.api_key,
                secret_key=self.config.secret_key
            )
            self.logger.info("Crypto data client initialized")
        return self._crypto_data_client

    # ─── Account Operations ────────────────────────────────────────────

    def get_account(self) -> Dict:
        """Get account information"""
        account = self.trading_client.get_account()
        return {
            'account_id': account.id,
            'status': account.status,
            'cash': float(account.cash),
            'buying_power': float(account.buying_power),
            'portfolio_value': float(account.portfolio_value),
            'equity': float(account.equity),
            'long_market_value': float(account.long_market_value),
            'short_market_value': float(account.short_market_value),
            'initial_margin': float(account.initial_margin),
            'maintenance_margin': float(account.maintenance_margin),
            'daytrade_count': account.daytrade_count,
            'pattern_day_trader': account.pattern_day_trader,
            'trading_blocked': account.trading_blocked,
            'account_blocked': account.account_blocked,
        }

    def get_buying_power(self) -> float:
        """Get current buying power"""
        account = self.trading_client.get_account()
        return float(account.buying_power)

    # ─── Position Operations ───────────────────────────────────────────

    def get_positions(self) -> List[Dict]:
        """Get all open positions"""
        positions = self.trading_client.get_all_positions()
        return [{
            'symbol': p.symbol,
            'qty': float(p.qty),
            'side': p.side,
            'avg_entry_price': float(p.avg_entry_price),
            'current_price': float(p.current_price),
            'market_value': float(p.market_value),
            'unrealized_pl': float(p.unrealized_pl),
            'unrealized_plpc': float(p.unrealized_plpc),
            'change_today': float(p.change_today),
        } for p in positions]

    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get position for a specific symbol"""
        try:
            p = self.trading_client.get_open_position(symbol)
            return {
                'symbol': p.symbol,
                'qty': float(p.qty),
                'side': p.side,
                'avg_entry_price': float(p.avg_entry_price),
                'current_price': float(p.current_price),
                'market_value': float(p.market_value),
                'unrealized_pl': float(p.unrealized_pl),
                'unrealized_plpc': float(p.unrealized_plpc),
            }
        except APIError:
            return None

    # ─── Order Operations ──────────────────────────────────────────────

    def submit_market_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        time_in_force: str = 'day'
    ) -> Dict:
        """
        Submit a market order.

        Args:
            symbol: Stock ticker
            qty: Number of shares (positive integer)
            side: 'buy' or 'sell'
            time_in_force: 'day', 'gtc', 'ioc', 'fok'

        Returns:
            Order details dict
        """
        if self.config.trading_mode != TradingMode.LIVE:
            self.logger.info(
                f"[{self.config.trading_mode.value.upper()}] Would submit: "
                f"{side.upper()} {qty} {symbol} @ MARKET"
            )
            return {
                'id': f'shadow-{datetime.now().timestamp()}',
                'symbol': symbol,
                'qty': qty,
                'side': side,
                'type': 'market',
                'status': 'simulated',
                'submitted_at': datetime.now().isoformat(),
            }

        order_request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side == 'buy' else OrderSide.SELL,
            time_in_force=TimeInForce.DAY if time_in_force == 'day' else TimeInForce.GTC,
        )

        order = self.trading_client.submit_order(order_request)
        self.logger.info(f"Order submitted: {side.upper()} {qty} {symbol} → {order.status}")

        return {
            'id': str(order.id),
            'symbol': order.symbol,
            'qty': float(order.qty),
            'side': order.side.value,
            'type': order.type.value,
            'status': order.status.value,
            'submitted_at': str(order.submitted_at),
            'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else None,
        }

    def submit_bracket_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        stop_loss_price: float,
        take_profit_price: Optional[float] = None,
        time_in_force: str = 'gtc'
    ) -> Dict:
        """
        Submit a bracket (OTO/OCO) market order with attached stop-loss
        and optional take-profit legs.

        Alpaca handles the stop/TP server-side — no polling needed for
        these exit conditions.

        Args:
            symbol: Stock ticker
            qty: Number of shares
            side: 'buy' or 'sell'  (entry side)
            stop_loss_price: Stop-loss trigger price
            take_profit_price: Take-profit limit price (None = OTO stop only)
            time_in_force: 'day' or 'gtc' (default: 'gtc')

        Returns:
            Order details dict with leg order IDs
        """
        if self.config.trading_mode != TradingMode.LIVE:
            self.logger.info(
                f"[{self.config.trading_mode.value.upper()}] Would submit BRACKET: "
                f"{side.upper()} {qty} {symbol} @ MARKET | "
                f"SL=${stop_loss_price:.2f}"
                + (f" TP=${take_profit_price:.2f}" if take_profit_price else "")
            )
            return {
                'id': f'shadow-{datetime.now().timestamp()}',
                'symbol': symbol,
                'qty': qty,
                'side': side,
                'type': 'market',
                'order_class': 'bracket' if take_profit_price else 'oto',
                'stop_loss_price': stop_loss_price,
                'take_profit_price': take_profit_price,
                'status': 'simulated',
                'submitted_at': datetime.now().isoformat(),
            }

        # Build bracket order
        order_side = OrderSide.BUY if side == 'buy' else OrderSide.SELL
        tif = TimeInForce.DAY if time_in_force == 'day' else TimeInForce.GTC

        stop_loss = StopLossRequest(stop_price=stop_loss_price)

        order_kwargs = dict(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=tif,
            order_class=OrderClass.BRACKET if take_profit_price else OrderClass.OTO,
            stop_loss=stop_loss,
        )

        if take_profit_price is not None:
            order_kwargs['take_profit'] = TakeProfitRequest(
                limit_price=take_profit_price
            )

        order_request = MarketOrderRequest(**order_kwargs)
        order = self.trading_client.submit_order(order_request)

        self.logger.info(
            f"Bracket order submitted: {side.upper()} {qty} {symbol} | "
            f"SL=${stop_loss_price:.2f}"
            + (f" TP=${take_profit_price:.2f}" if take_profit_price else "")
            + f" → {order.status}"
        )

        result = {
            'id': str(order.id),
            'symbol': order.symbol,
            'qty': float(order.qty),
            'side': order.side.value,
            'type': order.type.value,
            'order_class': 'bracket' if take_profit_price else 'oto',
            'stop_loss_price': stop_loss_price,
            'take_profit_price': take_profit_price,
            'status': order.status.value,
            'submitted_at': str(order.submitted_at),
            'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else None,
        }

        # Capture leg order IDs if available
        if hasattr(order, 'legs') and order.legs:
            result['leg_ids'] = [str(leg.id) for leg in order.legs]

        return result

    def get_latest_trade(self, symbol: str) -> Optional[Dict]:
        """Get the latest trade (price) for a symbol via data API."""
        try:
            from alpaca.data.requests import StockLatestTradeRequest
            request = StockLatestTradeRequest(symbol_or_symbols=symbol)
            trade = self.data_client.get_stock_latest_trade(request)
            if symbol in trade:
                t = trade[symbol]
                return {'price': float(t.price), 'size': int(t.size)}
        except Exception as e:
            self.logger.debug(f"Latest trade lookup failed for {symbol}: {e}")
        return None

    def get_latest_trades(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get latest trade prices for multiple symbols in a single API call.

        Returns:
            Dict mapping symbol → latest price
        """
        prices = {}
        try:
            from alpaca.data.requests import StockLatestTradeRequest
            request = StockLatestTradeRequest(symbol_or_symbols=symbols)
            trades = self.data_client.get_stock_latest_trade(request)
            for sym, t in trades.items():
                prices[sym] = float(t.price)
        except Exception as e:
            self.logger.warning(f"Batch latest-trade lookup failed: {e}")
        return prices

    def submit_limit_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        limit_price: float,
        time_in_force: str = 'day'
    ) -> Dict:
        """Submit a limit order"""
        if self.config.trading_mode != TradingMode.LIVE:
            self.logger.info(
                f"[{self.config.trading_mode.value.upper()}] Would submit: "
                f"{side.upper()} {qty} {symbol} @ ${limit_price:.2f} LIMIT"
            )
            return {
                'id': f'shadow-{datetime.now().timestamp()}',
                'symbol': symbol,
                'qty': qty,
                'side': side,
                'type': 'limit',
                'limit_price': limit_price,
                'status': 'simulated',
                'submitted_at': datetime.now().isoformat(),
            }

        order_request = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY if side == 'buy' else OrderSide.SELL,
            time_in_force=TimeInForce.DAY if time_in_force == 'day' else TimeInForce.GTC,
            limit_price=limit_price,
        )

        order = self.trading_client.submit_order(order_request)
        self.logger.info(
            f"Order submitted: {side.upper()} {qty} {symbol} "
            f"@ ${limit_price:.2f} → {order.status}"
        )

        return {
            'id': str(order.id),
            'symbol': order.symbol,
            'qty': float(order.qty),
            'side': order.side.value,
            'type': order.type.value,
            'limit_price': float(order.limit_price),
            'status': order.status.value,
            'submitted_at': str(order.submitted_at),
            'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else None,
        }

    def get_orders(self, status: str = 'all', limit: int = 100) -> List[Dict]:
        """Get orders with filtering"""
        status_map = {
            'all': QueryOrderStatus.ALL,
            'open': QueryOrderStatus.OPEN,
            'closed': QueryOrderStatus.CLOSED,
        }
        request = GetOrdersRequest(
            status=status_map.get(status, QueryOrderStatus.ALL),
            limit=limit
        )
        orders = self.trading_client.get_orders(request)
        return [{
            'id': str(o.id),
            'symbol': o.symbol,
            'qty': float(o.qty) if o.qty else None,
            'side': o.side.value if o.side else None,
            'type': o.type.value if o.type else None,
            'status': o.status.value if o.status else None,
            'submitted_at': str(o.submitted_at),
            'filled_avg_price': float(o.filled_avg_price) if o.filled_avg_price else None,
        } for o in orders]

    def cancel_all_orders(self) -> int:
        """Cancel all open orders. Returns count of cancelled orders."""
        cancelled = self.trading_client.cancel_orders()
        self.logger.info(f"Cancelled {len(cancelled)} orders")
        return len(cancelled)

    def close_all_positions(self) -> List[Dict]:
        """Close all open positions"""
        if self.config.trading_mode != TradingMode.LIVE:
            positions = self.get_positions()
            self.logger.info(
                f"[{self.config.trading_mode.value.upper()}] Would close "
                f"{len(positions)} positions"
            )
            return positions

        closed = self.trading_client.close_all_positions(cancel_orders=True)
        self.logger.info(f"Closed {len(closed)} positions")
        return [{'symbol': str(c)} for c in closed]

    # ─── Crypto Orders ──────────────────────────────────────────────────

    def submit_crypto_order(
        self,
        symbol: str,
        notional: float = None,
        qty: float = None,
        side: str = 'buy',
    ) -> Dict:
        """
        Submit a crypto market order.

        Crypto uses GTC (good-til-cancelled) since markets are 24/7.
        Specify either notional (dollar amount) or qty (coin amount), not both.

        Args:
            symbol: Crypto pair e.g. 'BTC/USD', 'ETH/USD'
            notional: Dollar amount to buy/sell (e.g. 25.0 for $25)
            qty: Coin quantity (e.g. 0.001 for 0.001 BTC)
            side: 'buy' or 'sell'

        Returns:
            Order details dict
        """
        if notional is None and qty is None:
            raise ValueError("Must specify either notional or qty")

        if self.config.trading_mode != TradingMode.LIVE:
            self.logger.info(
                f"[{self.config.trading_mode.value.upper()}] Would submit: "
                f"{side.upper()} {symbol} "
                f"{'$' + str(notional) if notional else str(qty) + ' coins'} @ MARKET"
            )
            return {
                'id': f'shadow-{datetime.now().timestamp()}',
                'symbol': symbol,
                'qty': qty,
                'notional': notional,
                'side': side,
                'type': 'market',
                'status': 'simulated',
                'submitted_at': datetime.now().isoformat(),
            }

        order_kwargs = dict(
            symbol=symbol,
            side=OrderSide.BUY if side == 'buy' else OrderSide.SELL,
            time_in_force=TimeInForce.GTC,  # Crypto is 24/7, use GTC
        )
        if notional is not None:
            order_kwargs['notional'] = notional
        else:
            order_kwargs['qty'] = qty

        order_request = MarketOrderRequest(**order_kwargs)
        order = self.trading_client.submit_order(order_request)
        self.logger.info(f"Crypto order submitted: {side.upper()} {symbol} → {order.status}")

        return {
            'id': str(order.id),
            'symbol': order.symbol,
            'qty': float(order.qty) if order.qty else None,
            'notional': float(order.notional) if order.notional else None,
            'side': order.side.value,
            'type': order.type.value,
            'status': order.status.value,
            'submitted_at': str(order.submitted_at),
            'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else None,
            'filled_qty': float(order.filled_qty) if order.filled_qty else None,
        }

    # ─── Portfolio History ──────────────────────────────────────────────

    def get_portfolio_history(self, period: str = "3M", timeframe: str = "1D") -> List[Dict]:
        """
        Get portfolio equity history from Alpaca.

        Args:
            period: History period ('1D', '1W', '1M', '3M', '1A', 'all')
            timeframe: Data granularity ('1Min', '5Min', '15Min', '1H', '1D')

        Returns:
            List of {'timestamp': 'YYYY-MM-DD HH:MM', 'equity': float} dicts
        """
        try:
            from alpaca.trading.requests import GetPortfolioHistoryRequest

            request = GetPortfolioHistoryRequest(
                period=period,
                timeframe=timeframe,
            )
            history = self.trading_client.get_portfolio_history(history_filter=request)

            results = []
            if history.timestamp and history.equity:
                for ts, eq in zip(history.timestamp, history.equity):
                    if eq is not None and eq > 0:
                        dt = datetime.fromtimestamp(ts)
                        results.append({
                            "timestamp": dt.strftime("%Y-%m-%d %H:%M"),
                            "equity": float(eq),
                        })

            self.logger.info(
                f"Retrieved {len(results)} portfolio history points ({period}/{timeframe})"
            )
            return results

        except Exception as e:
            self.logger.warning(f"Could not get portfolio history: {e}")
            return []

    # ─── Market Clock ──────────────────────────────────────────────────

    def get_clock(self) -> Dict:
        """Get market clock (open/close times, is_open status)"""
        clock = self.trading_client.get_clock()
        return {
            'is_open': clock.is_open,
            'next_open': str(clock.next_open),
            'next_close': str(clock.next_close),
            'timestamp': str(clock.timestamp),
        }

    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        return self.trading_client.get_clock().is_open

    # ─── Connection Test ───────────────────────────────────────────────

    def test_connection(self) -> bool:
        """
        Test API connection and print account summary.
        
        Returns:
            True if connection successful
        """
        try:
            account = self.get_account()
            clock = self.get_clock()

            print("=" * 60)
            print("  ALPACA CONNECTION TEST")
            print("=" * 60)
            print(f"  Status:          {account['status']}")
            print(f"  Mode:            {'PAPER' if self.config.paper else 'LIVE'}")
            print(f"  Trading Mode:    {self.config.trading_mode.value.upper()}")
            print(f"  Cash:            ${account['cash']:,.2f}")
            print(f"  Buying Power:    ${account['buying_power']:,.2f}")
            print(f"  Portfolio Value: ${account['portfolio_value']:,.2f}")
            print(f"  Market Open:     {clock['is_open']}")
            print(f"  Day Trades:      {account['daytrade_count']}")
            print(f"  PDT Flag:        {account['pattern_day_trader']}")
            print("=" * 60)
            return True

        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
