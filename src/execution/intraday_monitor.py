"""
Intraday Exit Monitor — Hybrid Risk Management

Polls held positions between market open and the daily execution window
(9:45 AM – 3:50 PM ET) to enforce risk limits that bracket orders alone
cannot handle:

  • Trailing stop — requires tracking peak P&L per position
  • Time-decay exit — requires knowledge of entry date
  • Portfolio-level circuit breaker — kills all positions if portfolio DD
    exceeds a threshold intraday

Bracket orders (server-side stop-loss + take-profit) handle the simple
hard limits; this monitor handles the *dynamic* ones.

Architecture:
  IntradayMonitor.run()  →  polling loop every `poll_interval` seconds
    ├── fetch latest prices (single batch REST call)
    ├── check trailing stop per position
    ├── check time-decay exits
    ├── check portfolio circuit breaker
    └── submit exits for any triggered positions

Usage:
  monitor = IntradayMonitor(conn, config, shadow_sim=sim)
  monitor.run()   # blocks until execution window or shutdown

The main_trader script launches this *before* the daily 3:55 PM cycle.
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Set

import pandas as pd
import pytz

from connection.alpaca_connection import AlpacaConnection, TradingMode
from execution.simulation import SimulationEngine, SimulatedPosition

logger = logging.getLogger(__name__)

ET = pytz.timezone("US/Eastern")


class IntradayMonitorConfig:
    """Configuration for the intraday monitoring loop."""

    def __init__(
        self,
        poll_interval_sec: int = 300,           # 5 minutes between checks
        start_time_et: str = "09:45",            # Start monitoring (15 min after open)
        stop_time_et: str = "15:50",             # Stop before daily execution window
        trailing_stop_enabled: bool = True,
        trailing_stop_trail_pct: float = 0.05,   # 5% trail from peak
        trailing_stop_activation_pct: float = 0.02,  # Activate after 2% profit
        time_decay_enabled: bool = True,
        time_decay_days: int = 10,               # Check after N days
        time_decay_threshold: float = 0.01,      # Exit if |P&L| < 1%
        circuit_breaker_enabled: bool = True,
        circuit_breaker_drawdown_pct: float = 0.08,  # 8% portfolio drawdown
        circuit_breaker_cooldown_min: int = 60,       # Re-arm after 60 min
    ):
        self.poll_interval_sec = poll_interval_sec
        self.start_time_et = start_time_et
        self.stop_time_et = stop_time_et
        self.trailing_stop_enabled = trailing_stop_enabled
        self.trailing_stop_trail_pct = trailing_stop_trail_pct
        self.trailing_stop_activation_pct = trailing_stop_activation_pct
        self.time_decay_enabled = time_decay_enabled
        self.time_decay_days = time_decay_days
        self.time_decay_threshold = time_decay_threshold
        self.circuit_breaker_enabled = circuit_breaker_enabled
        self.circuit_breaker_drawdown_pct = circuit_breaker_drawdown_pct
        self.circuit_breaker_cooldown_min = circuit_breaker_cooldown_min

    @classmethod
    def from_dict(cls, d: dict) -> "IntradayMonitorConfig":
        """Create from config dict (e.g. from YAML)."""
        return cls(
            poll_interval_sec=d.get("poll_interval_sec", 300),
            start_time_et=d.get("start_time_et", "09:45"),
            stop_time_et=d.get("stop_time_et", "15:50"),
            trailing_stop_enabled=d.get("trailing_stop_enabled", True),
            trailing_stop_trail_pct=d.get("trailing_stop_trail_pct", 0.05),
            trailing_stop_activation_pct=d.get("trailing_stop_activation_pct", 0.02),
            time_decay_enabled=d.get("time_decay_enabled", True),
            time_decay_days=d.get("time_decay_days", 10),
            time_decay_threshold=d.get("time_decay_threshold", 0.01),
            circuit_breaker_enabled=d.get("circuit_breaker_enabled", True),
            circuit_breaker_drawdown_pct=d.get("circuit_breaker_drawdown_pct", 0.08),
            circuit_breaker_cooldown_min=d.get("circuit_breaker_cooldown_min", 60),
        )


class IntradayMonitor:
    """
    Monitors held positions intraday for dynamic exit conditions.

    Works in both LIVE and SHADOW mode:
      • LIVE  → submits real exit orders via Alpaca
      • SHADOW → updates SimulationEngine state and logs exits
    """

    def __init__(
        self,
        connection: AlpacaConnection,
        config: IntradayMonitorConfig,
        shadow_sim: Optional[SimulationEngine] = None,
        shutdown_flag=None,
    ):
        """
        Args:
            connection: Alpaca connection (for prices + order submission)
            config: IntradayMonitorConfig
            shadow_sim: SimulationEngine for shadow mode position tracking
            shutdown_flag: Callable returning True when shutdown requested
        """
        self.conn = connection
        self.config = config
        self.shadow_sim = shadow_sim
        self.mode = connection.config.trading_mode
        self._shutdown = shutdown_flag or (lambda: False)

        # Peak price tracking for trailing stop (symbol → peak_price)
        self._peak_prices: Dict[str, float] = {}

        # Circuit breaker state
        self._peak_equity: Optional[float] = None
        self._circuit_breaker_fired: bool = False
        self._circuit_breaker_time: Optional[datetime] = None

        # Track which symbols we already exited this session (avoid re-exit)
        self._exited_today: Set[str] = set()

    # ─── Main Loop ─────────────────────────────────────────────────────

    def run(self) -> Dict:
        """
        Run the intraday monitoring loop.

        Blocks until:
          • stop_time_et reached (returns control to daily execution)
          • shutdown requested
          • circuit breaker fires and all positions closed

        Returns:
            Summary dict with counts of exits triggered.
        """
        summary = {
            "trailing_stop_exits": 0,
            "time_decay_exits": 0,
            "circuit_breaker_exits": 0,
            "polls": 0,
            "errors": 0,
        }

        logger.info(
            f"🔍 Intraday monitor started | "
            f"mode={self.mode.value} | "
            f"poll={self.config.poll_interval_sec}s | "
            f"window={self.config.start_time_et}–{self.config.stop_time_et} ET"
        )

        while not self._shutdown():
            now_et = datetime.now(ET)

            # Check if we're in the monitoring window
            start_h, start_m = map(int, self.config.start_time_et.split(":"))
            stop_h, stop_m = map(int, self.config.stop_time_et.split(":"))
            start_mins = start_h * 60 + start_m
            stop_mins = stop_h * 60 + stop_m
            now_mins = now_et.hour * 60 + now_et.minute

            if now_mins < start_mins:
                # Before window — sleep until start
                wait = (start_mins - now_mins) * 60
                logger.debug(f"Before monitoring window — sleeping {wait/60:.0f}min")
                self._sleep(min(wait, 300))
                continue

            if now_mins >= stop_mins:
                # Past window — return control to daily execution
                logger.info(
                    f"📊 Intraday monitor complete | "
                    f"polls={summary['polls']} | "
                    f"trailing_exits={summary['trailing_stop_exits']} | "
                    f"decay_exits={summary['time_decay_exits']} | "
                    f"breaker_exits={summary['circuit_breaker_exits']}"
                )
                return summary

            # ── Poll cycle ─────────────────────────────────────────────
            try:
                positions = self._get_positions()
                if not positions:
                    summary["polls"] += 1
                    self._sleep(self.config.poll_interval_sec)
                    continue

                # Fetch latest prices (single batch call)
                symbols = list(positions.keys())
                prices = self.conn.get_latest_trades(symbols)

                if not prices:
                    logger.warning("No price data returned — skipping cycle")
                    summary["polls"] += 1
                    self._sleep(self.config.poll_interval_sec)
                    continue

                # Update peak prices for trailing stop tracking
                self._update_peaks(positions, prices)

                # ── Check exit conditions ──────────────────────────────
                exits_triggered: List[Dict] = []

                # 1. Trailing stop
                if self.config.trailing_stop_enabled:
                    ts_exits = self._check_trailing_stops(positions, prices)
                    exits_triggered.extend(ts_exits)
                    summary["trailing_stop_exits"] += len(ts_exits)

                # 2. Time-decay exit
                if self.config.time_decay_enabled:
                    td_exits = self._check_time_decay(positions, prices)
                    exits_triggered.extend(td_exits)
                    summary["time_decay_exits"] += len(td_exits)

                # 3. Circuit breaker
                if self.config.circuit_breaker_enabled:
                    cb_exits = self._check_circuit_breaker(positions, prices)
                    exits_triggered.extend(cb_exits)
                    summary["circuit_breaker_exits"] += len(cb_exits)

                # ── Execute exits ──────────────────────────────────────
                if exits_triggered:
                    self._execute_exits(exits_triggered, prices)

                summary["polls"] += 1

            except Exception as e:
                logger.error(f"Intraday monitor error: {e}", exc_info=True)
                summary["errors"] += 1

            self._sleep(self.config.poll_interval_sec)

        logger.info("Intraday monitor stopped (shutdown requested)")
        return summary

    # ─── Position Retrieval ────────────────────────────────────────────

    def _get_positions(self) -> Dict[str, Dict]:
        """
        Get current positions from live account or shadow sim.

        Returns:
            Dict[symbol] → {qty, side, entry_price, entry_date, current_price}
        """
        if self.mode == TradingMode.SHADOW and self.shadow_sim:
            return {
                sym: {
                    "qty": pos.qty,
                    "side": pos.side,
                    "entry_price": pos.entry_price,
                    "entry_date": pos.entry_date,
                    "current_price": pos.current_price,
                    "peak_price": getattr(pos, "peak_price", pos.entry_price),
                }
                for sym, pos in self.shadow_sim.positions.items()
                if sym not in self._exited_today
            }

        elif self.mode == TradingMode.LIVE:
            positions = {}
            try:
                for pos in self.conn.get_positions():
                    sym = pos["symbol"]
                    if sym in self._exited_today:
                        continue
                    qty = int(pos["qty"])
                    positions[sym] = {
                        "qty": abs(qty),
                        "side": "long" if qty > 0 else "short",
                        "entry_price": float(pos["avg_entry_price"]),
                        "entry_date": pd.Timestamp.now() - pd.Timedelta(days=1),
                        "current_price": float(pos["current_price"]),
                    }
            except Exception as e:
                logger.error(f"Failed to fetch live positions: {e}")
            return positions

        return {}

    # ─── Peak Price Tracking ───────────────────────────────────────────

    def _update_peaks(
        self, positions: Dict[str, Dict], prices: Dict[str, float]
    ) -> None:
        """Update peak prices for trailing stop logic."""
        for sym, pos in positions.items():
            current = prices.get(sym)
            if current is None:
                continue

            if sym not in self._peak_prices:
                # Initialize from position data or current price
                self._peak_prices[sym] = max(
                    pos.get("peak_price", pos["entry_price"]),
                    current if pos["side"] == "long" else pos["entry_price"],
                )

            if pos["side"] == "long":
                self._peak_prices[sym] = max(self._peak_prices[sym], current)
            else:
                # For shorts, "peak" means the lowest price seen
                if sym not in self._peak_prices or self._peak_prices[sym] > current:
                    self._peak_prices[sym] = min(
                        self._peak_prices.get(sym, current), current
                    )

            # Also update shadow sim position if available
            if (
                self.shadow_sim
                and sym in self.shadow_sim.positions
            ):
                self.shadow_sim.positions[sym].current_price = current

    # ─── Trailing Stop Check ──────────────────────────────────────────

    def _check_trailing_stops(
        self, positions: Dict[str, Dict], prices: Dict[str, float]
    ) -> List[Dict]:
        """Check trailing stop for each position."""
        exits = []
        for sym, pos in positions.items():
            current = prices.get(sym)
            if current is None:
                continue

            entry = pos["entry_price"]
            side = pos["side"]
            peak = self._peak_prices.get(sym, entry)

            # Calculate P&L from entry and from peak
            if side == "long":
                pnl_pct = (current - entry) / entry
                peak_pnl = (peak - entry) / entry
                drawback = peak_pnl - pnl_pct
            else:
                pnl_pct = (entry - current) / entry
                peak_pnl = (entry - peak) / entry
                drawback = peak_pnl - pnl_pct

            # Only activate after reaching activation threshold
            if peak_pnl < self.config.trailing_stop_activation_pct:
                continue

            if drawback >= self.config.trailing_stop_trail_pct:
                logger.info(
                    f"🔻 TRAILING STOP: {sym} {side} | "
                    f"peak_pnl={peak_pnl:.2%} → current={pnl_pct:.2%} | "
                    f"drawback={drawback:.2%} >= trail={self.config.trailing_stop_trail_pct:.2%}"
                )
                exits.append({
                    "symbol": sym,
                    "reason": f"Intraday trailing stop (peak {peak_pnl:.2%}, "
                              f"drawback {drawback:.2%})",
                    "side": side,
                    "qty": pos["qty"],
                })

        return exits

    # ─── Time Decay Check ─────────────────────────────────────────────

    def _check_time_decay(
        self, positions: Dict[str, Dict], prices: Dict[str, float]
    ) -> List[Dict]:
        """Exit positions that are flat after holding too long."""
        exits = []
        now = pd.Timestamp.now()

        for sym, pos in positions.items():
            current = prices.get(sym)
            if current is None:
                continue

            entry_date = pos.get("entry_date")
            if entry_date is None:
                continue

            days_held = (now - pd.Timestamp(entry_date)).days
            if days_held < self.config.time_decay_days:
                continue

            entry = pos["entry_price"]
            side = pos["side"]

            if side == "long":
                pnl_pct = (current - entry) / entry
            else:
                pnl_pct = (entry - current) / entry

            if abs(pnl_pct) < self.config.time_decay_threshold:
                logger.info(
                    f"⏳ TIME DECAY: {sym} {side} | "
                    f"held {days_held}d, P&L={pnl_pct:.2%} (flat < "
                    f"{self.config.time_decay_threshold:.2%})"
                )
                exits.append({
                    "symbol": sym,
                    "reason": f"Intraday time decay (held {days_held}d, "
                              f"P&L {pnl_pct:.2%})",
                    "side": side,
                    "qty": pos["qty"],
                })

        return exits

    # ─── Circuit Breaker ──────────────────────────────────────────────

    def _check_circuit_breaker(
        self, positions: Dict[str, Dict], prices: Dict[str, float]
    ) -> List[Dict]:
        """
        Portfolio-level kill switch: close everything if portfolio DD
        exceeds threshold intraday.
        """
        # Check cooldown
        if self._circuit_breaker_fired:
            if self._circuit_breaker_time:
                elapsed = (datetime.now(ET) - self._circuit_breaker_time).total_seconds()
                if elapsed < self.config.circuit_breaker_cooldown_min * 60:
                    return []
            self._circuit_breaker_fired = False
            self._circuit_breaker_time = None

        # Calculate current portfolio equity
        if self.mode == TradingMode.SHADOW and self.shadow_sim:
            current_equity = self.shadow_sim.equity
            if self._peak_equity is None:
                self._peak_equity = current_equity
        elif self.mode == TradingMode.LIVE:
            try:
                account = self.conn.get_account()
                current_equity = account["portfolio_value"]
                if self._peak_equity is None:
                    self._peak_equity = current_equity
            except Exception:
                return []
        else:
            return []

        self._peak_equity = max(self._peak_equity, current_equity)
        dd_pct = (self._peak_equity - current_equity) / self._peak_equity

        if dd_pct >= self.config.circuit_breaker_drawdown_pct:
            logger.warning(
                f"🚨 CIRCUIT BREAKER: Portfolio DD = {dd_pct:.2%} >= "
                f"{self.config.circuit_breaker_drawdown_pct:.2%} | "
                f"peak=${self._peak_equity:,.0f} → now=${current_equity:,.0f} | "
                f"CLOSING ALL POSITIONS"
            )
            self._circuit_breaker_fired = True
            self._circuit_breaker_time = datetime.now(ET)

            exits = []
            for sym, pos in positions.items():
                exits.append({
                    "symbol": sym,
                    "reason": f"Circuit breaker (portfolio DD {dd_pct:.2%})",
                    "side": pos["side"],
                    "qty": pos["qty"],
                })
            return exits

        return []

    # ─── Exit Execution ───────────────────────────────────────────────

    def _execute_exits(
        self, exits: List[Dict], prices: Dict[str, float]
    ) -> None:
        """Submit exit orders (live) or update simulation (shadow)."""
        for ex in exits:
            sym = ex["symbol"]
            side = ex["side"]
            qty = ex["qty"]
            reason = ex["reason"]

            if sym in self._exited_today:
                continue

            try:
                if self.mode == TradingMode.LIVE:
                    # Submit market exit order
                    exit_side = "sell" if side == "long" else "buy"
                    order = self.conn.submit_market_order(
                        symbol=sym,
                        qty=qty,
                        side=exit_side,
                        time_in_force="day",
                    )
                    logger.info(
                        f"📤 EXIT ORDER: {exit_side.upper()} {qty} {sym} | "
                        f"{reason} → {order['status']}"
                    )

                elif self.mode == TradingMode.SHADOW and self.shadow_sim:
                    # Close position in simulation
                    current_price = prices.get(sym)
                    if current_price and sym in self.shadow_sim.positions:
                        pos = self.shadow_sim.positions[sym]
                        # Record the completed trade
                        from execution.simulation import CompletedTrade
                        if side == "long":
                            pnl = (current_price - pos.entry_price) * abs(qty)
                            pnl_pct = (current_price - pos.entry_price) / pos.entry_price
                        else:
                            pnl = (pos.entry_price - current_price) * abs(qty)
                            pnl_pct = (pos.entry_price - current_price) / pos.entry_price

                        commission = current_price * abs(qty) * self.shadow_sim.commission_pct

                        trade = CompletedTrade(
                            symbol=sym,
                            side=side,
                            qty=abs(qty),
                            entry_price=pos.entry_price,
                            exit_price=current_price,
                            entry_date=pos.entry_date,
                            exit_date=pd.Timestamp.now(),
                            pnl=pnl - commission,
                            pnl_pct=pnl_pct,
                            commission=commission,
                            exit_reason=reason,
                            signal_strength=pos.signal_strength,
                        )
                        self.shadow_sim.completed_trades.append(trade)

                        # Update cash
                        if side == "long":
                            self.shadow_sim.cash += current_price * abs(qty) - commission
                        else:
                            self.shadow_sim.cash -= current_price * abs(qty) + commission

                        # Remove position
                        del self.shadow_sim.positions[sym]
                        logger.info(
                            f"📤 SHADOW EXIT: {sym} {side} | "
                            f"P&L={pnl_pct:.2%} (${pnl:,.2f}) | {reason}"
                        )

                self._exited_today.add(sym)

            except Exception as e:
                logger.error(f"Failed to exit {sym}: {e}", exc_info=True)

    # ─── Bracket Order Computations ───────────────────────────────────

    @staticmethod
    def compute_bracket_prices(
        entry_price: float,
        side: str,
        stop_loss_pct: float,
        take_profit_pct: Optional[float] = None,
    ) -> Dict[str, Optional[float]]:
        """
        Compute stop-loss and take-profit prices for a bracket order.

        Args:
            entry_price: Expected entry price
            side: 'long' or 'short'
            stop_loss_pct: Stop-loss as fraction (e.g. 0.08 = 8%)
            take_profit_pct: Take-profit as fraction (None = no TP leg)

        Returns:
            {'stop_loss_price': float, 'take_profit_price': float|None}
        """
        if side == "long":
            sl = round(entry_price * (1 - stop_loss_pct), 2)
            tp = round(entry_price * (1 + take_profit_pct), 2) if take_profit_pct else None
        else:
            sl = round(entry_price * (1 + stop_loss_pct), 2)
            tp = round(entry_price * (1 - take_profit_pct), 2) if take_profit_pct else None

        return {"stop_loss_price": sl, "take_profit_price": tp}

    # ─── Utilities ─────────────────────────────────────────────────────

    def _sleep(self, seconds: float) -> None:
        """Interruptible sleep."""
        end = time.time() + seconds
        while time.time() < end and not self._shutdown():
            time.sleep(min(1.0, end - time.time()))

    def reset_session(self) -> None:
        """Reset daily state for a new trading session."""
        self._exited_today.clear()
        self._peak_prices.clear()
        self._peak_equity = None
        self._circuit_breaker_fired = False
        self._circuit_breaker_time = None
        logger.info("Intraday monitor session reset")
