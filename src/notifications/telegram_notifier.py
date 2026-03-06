"""
Telegram Bot Notifier -- Trading Alerts via Telegram Bot API

Sends real-time notifications for:
  - Signal generation (Phase 1 post-close)
  - Trade execution (Phase 2 morning)
  - Intraday exits (stop-loss, trailing stop, time-decay, circuit breaker)
  - Daily account summary
  - System errors and alerts

Uses the Telegram Bot HTTP API (no extra dependencies beyond requests/urllib).
All sends are fire-and-forget with error logging -- never blocks trading logic.
"""

import logging
import os
import html
import urllib.request
import urllib.parse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any

logger = logging.getLogger("trader.telegram")

# ═══════════════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════════════

_instance: Optional["TelegramNotifier"] = None


def get_notifier() -> Optional["TelegramNotifier"]:
    """
    Return the global TelegramNotifier singleton.

    Lazily initializes from environment variables on first call.
    Returns None if TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is not set
    (notifications silently disabled).
    """
    global _instance
    if _instance is not None:
        return _instance

    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

    if not token or not chat_id:
        logger.info(
            "Telegram notifications disabled (TELEGRAM_BOT_TOKEN or "
            "TELEGRAM_CHAT_ID not set)"
        )
        return None

    _instance = TelegramNotifier(token=token, chat_id=chat_id)
    logger.info("Telegram notifier initialized")
    return _instance


# ═══════════════════════════════════════════════════════════════════════════
# NOTIFIER CLASS
# ═══════════════════════════════════════════════════════════════════════════


class TelegramNotifier:
    """
    Send trading notifications via Telegram Bot API.

    Uses urllib (stdlib) so there are zero extra dependencies.
    All public methods are safe to call in production -- they catch
    and log any network/API errors without raising.
    """

    API_URL = "https://api.telegram.org/bot{token}/sendMessage"

    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self._url = self.API_URL.format(token=token)

    # ─── Low-Level Send ────────────────────────────────────────────────

    def _send(self, text: str, parse_mode: str = "HTML") -> bool:
        """
        Send a message via Telegram Bot API.

        Returns True on success, False on failure (logged, never raised).
        """
        try:
            payload = json.dumps({
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True,
            }).encode("utf-8")

            req = urllib.request.Request(
                self._url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read().decode())
                if not result.get("ok"):
                    logger.warning(f"Telegram API error: {result}")
                    return False
                msg_id = result.get("result", {}).get("message_id", "?")
                logger.info(f"Telegram message sent (id={msg_id})")
                return True

        except Exception as e:
            logger.warning(f"Telegram send failed: {e}")
            return False

    # ─── High-Level Notification Methods ───────────────────────────────

    def notify_signals_generated(
        self,
        date,
        mode: str,
        total_valid: int,
        actionable_count: int,
        entry_threshold: float,
        actionable_signals: Optional[Dict[str, Dict]] = None,
    ) -> bool:
        """
        Notification after Phase 1 signal generation completes.

        Args:
            date: Signal date
            mode: 'live' or 'shadow'
            total_valid: Number of valid (non-NaN) signals
            actionable_count: Number above entry threshold
            entry_threshold: Current threshold value
            actionable_signals: {symbol: {signal, zscore, price}} for actionable
        """
        header = (
            f"<b>SIGNALS GENERATED</b>  [{mode.upper()}]\n"
            f"Date: {date}\n"
            f"Threshold: {entry_threshold}\n"
        )

        body = (
            f"Valid signals: {total_valid}\n"
            f"Actionable: {actionable_count}\n"
        )

        if actionable_signals and actionable_count > 0:
            body += "\n"
            # Split into longs and shorts
            longs = {s: d for s, d in actionable_signals.items() if d["signal"] < 0}
            shorts = {s: d for s, d in actionable_signals.items() if d["signal"] > 0}

            if longs:
                body += f"<b>LONG ({len(longs)}):</b>\n"
                for sym, d in sorted(longs.items(), key=lambda x: x[1]["signal"]):
                    body += (
                        f"  BUY {sym}  sig={d['signal']:+.3f}  "
                        f"z={d['zscore']:+.2f}  ${d['price']:,.2f}\n"
                    )
            if shorts:
                body += f"<b>SHORT ({len(shorts)}):</b>\n"
                for sym, d in sorted(shorts.items(), key=lambda x: -x[1]["signal"]):
                    body += (
                        f"  SELL {sym}  sig={d['signal']:+.3f}  "
                        f"z={d['zscore']:+.2f}  ${d['price']:,.2f}\n"
                    )
        elif actionable_count == 0:
            body += "\nNo actionable signals today (gated mode -- waiting for high-conviction setups)."

        return self._send(header + body)

    def notify_trade_executed(
        self,
        date,
        mode: str,
        decisions_data: List[Dict],
        portfolio_value: float,
        cash: float,
    ) -> bool:
        """
        Notification after Phase 2 trade execution completes.

        Args:
            date: Execution date
            mode: 'live' or 'shadow'
            decisions_data: List of {symbol, action, qty, status, price}
            portfolio_value: Current portfolio value
            cash: Current cash
        """
        _OK = ("filled", "submitted", "pending_new", "accepted")
        filled = sum(1 for d in decisions_data if d.get("status") in _OK)
        total = len(decisions_data)

        header = (
            f"<b>TRADES EXECUTED</b>  [{mode.upper()}]\n"
            f"Date: {date}\n"
        )

        if total == 0:
            body = "No trades today.\n"
        else:
            body = f"Decisions: {total} | Filled: {filled}\n\n"
            for d in decisions_data:
                status_icon = "[OK]" if d.get("status") in _OK else "[FAIL]"
                body += (
                    f"{status_icon} {d['action'].upper()} {d.get('qty', '?')} "
                    f"{d['symbol']}  @${d.get('price', 0):,.2f}\n"
                )

        body += (
            f"\n<b>Account:</b>\n"
            f"Portfolio: ${portfolio_value:,.2f}\n"
            f"Cash: ${cash:,.2f}\n"
        )

        return self._send(header + body)

    def notify_intraday_exit(
        self,
        symbol: str,
        side: str,
        qty: int,
        reason: str,
        entry_price: Optional[float] = None,
        exit_price: Optional[float] = None,
        pnl_pct: Optional[float] = None,
        mode: str = "live",
    ) -> bool:
        """
        Notification when the intraday monitor triggers an exit.

        Args:
            symbol: Ticker symbol
            side: 'long' or 'short'
            qty: Position quantity
            reason: Exit reason (stop_loss, trailing_stop, time_decay, circuit_breaker)
            entry_price: Entry price (if available)
            exit_price: Current/exit price (if available)
            pnl_pct: P&L percentage (if available)
            mode: 'live' or 'shadow'
        """
        # Determine exit type label
        if "stop_loss" in reason.lower():
            exit_type = "STOP-LOSS"
        elif "trailing" in reason.lower():
            exit_type = "TRAILING STOP"
        elif "time" in reason.lower() or "decay" in reason.lower():
            exit_type = "TIME DECAY"
        elif "circuit" in reason.lower() or "breaker" in reason.lower():
            exit_type = "CIRCUIT BREAKER"
        else:
            exit_type = "EXIT"

        action = "SELL" if side == "long" else "BUY TO COVER"

        header = f"<b>INTRADAY {exit_type}</b>  [{mode.upper()}]\n\n"

        body = (
            f"{action} {qty} {symbol}\n"
            f"Side: {side}\n"
            f"Reason: {reason}\n"
        )

        if entry_price is not None:
            body += f"Entry: ${entry_price:,.2f}\n"
        if exit_price is not None:
            body += f"Exit: ${exit_price:,.2f}\n"
        if pnl_pct is not None:
            pnl_sign = "+" if pnl_pct >= 0 else ""
            body += f"P&L: {pnl_sign}{pnl_pct:.2%}\n"

        return self._send(header + body)

    def notify_daily_summary(
        self,
        date,
        mode: str,
        portfolio_value: float,
        cash: float,
        positions: List[Dict],
        day_pnl: Optional[float] = None,
        day_pnl_pct: Optional[float] = None,
        signals_generated: int = 0,
        trades_executed: int = 0,
    ) -> bool:
        """
        End-of-day summary notification.

        Args:
            date: Date
            mode: 'live' or 'shadow'
            portfolio_value: Total portfolio value
            cash: Cash balance
            positions: List of {symbol, side, qty, entry_price, current_price, pnl_pct}
            day_pnl: Day's P&L in dollars
            day_pnl_pct: Day's P&L percentage
            signals_generated: Number of actionable signals
            trades_executed: Number of trades filled
        """
        header = (
            f"<b>DAILY SUMMARY</b>  [{mode.upper()}]\n"
            f"Date: {date}\n"
            f"{'='*30}\n"
        )

        body = (
            f"Portfolio: ${portfolio_value:,.2f}\n"
            f"Cash: ${cash:,.2f}\n"
        )

        if day_pnl is not None:
            sign = "+" if day_pnl >= 0 else ""
            body += f"Day P&L: {sign}${day_pnl:,.2f}"
            if day_pnl_pct is not None:
                body += f" ({sign}{day_pnl_pct:.2%})"
            body += "\n"

        body += (
            f"\nSignals: {signals_generated} actionable\n"
            f"Trades: {trades_executed} executed\n"
        )

        if positions:
            body += f"\n<b>Open Positions ({len(positions)}):</b>\n"
            for p in positions:
                pnl_str = ""
                if p.get("pnl_pct") is not None:
                    pnl_sign = "+" if p["pnl_pct"] >= 0 else ""
                    pnl_str = f"  P&L: {pnl_sign}{p['pnl_pct']:.2%}"
                body += (
                    f"  {p.get('side', '?').upper()} {p.get('qty', '?')} {p['symbol']}"
                    f"  @${p.get('entry_price', 0):,.2f}{pnl_str}\n"
                )
        else:
            body += "\nNo open positions.\n"

        return self._send(header + body)

    def notify_error(
        self,
        error_msg: str,
        context: str = "",
        mode: str = "live",
    ) -> bool:
        """
        Notification for errors or system alerts.

        Args:
            error_msg: Error description
            context: Where the error occurred
            mode: 'live' or 'shadow'
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text = (
            f"<b>ALERT</b>  [{mode.upper()}]\n"
            f"Time: {now}\n"
        )
        if context:
            text += f"Context: {context}\n"
        text += f"\n{error_msg}\n"

        return self._send(text)

    def notify_execution_preview(
        self,
        date,
        planned_exits: List[Dict],
        planned_entries: List[Dict],
        holding: List[str],
    ) -> bool:
        """
        Notification showing planned T+1 execution actions.

        Sent after signal generation to show what will happen at market open.
        """
        header = (
            f"<b>EXECUTION PREVIEW</b>  [T+1]\n"
            f"Signals: {date}\n"
            f"{'='*30}\n"
        )

        body = ""

        if planned_exits:
            body += f"\n<b>PLANNED EXITS ({len(planned_exits)}):</b>\n"
            for ex in planned_exits:
                pnl_sign = "+" if ex["pnl_pct"] >= 0 else ""
                reason = html.escape(ex['reason'])
                body += (
                    f"  {ex['action']} {ex['qty']} {ex['symbol']}  "
                    f"P&L: {pnl_sign}{ex['pnl_pct']:.2%}\n"
                    f"    Reason: {reason}\n"
                )
        else:
            body += "\nNo planned exits.\n"

        if planned_entries:
            body += f"\n<b>PLANNED ENTRIES ({len(planned_entries)}):</b>\n"
            for en in planned_entries:
                body += (
                    f"  {en['action']} {en['symbol']}  "
                    f"sig={en['signal']:+.3f}  ${en['price']:,.2f}\n"
                )
        else:
            body += "\nNo planned entries.\n"

        if holding:
            body += f"\n<b>HOLDING ({len(holding)}):</b> {', '.join(holding)}\n"

        return self._send(header + body)

    def notify_startup(self, mode: str, universe_size: int, pid: int) -> bool:
        """Notification when the trading service starts."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text = (
            f"<b>SERVICE STARTED</b>  [{mode.upper()}]\n"
            f"Time: {now}\n"
            f"PID: {pid}\n"
            f"Universe: {universe_size} symbols\n"
        )
        return self._send(text)

    def notify_shutdown(self, mode: str, cycles: int, reason: str = "normal") -> bool:
        """Notification when the trading service stops."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text = (
            f"<b>SERVICE STOPPED</b>  [{mode.upper()}]\n"
            f"Time: {now}\n"
            f"Cycles: {cycles}\n"
            f"Reason: {reason}\n"
        )
        return self._send(text)
