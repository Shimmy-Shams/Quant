"""
Tests for the Intraday Exit Monitor.

Validates all dynamic exit conditions:
  - Hard stop-loss (long & short)
  - Trailing stop (activation + trail)
  - Time-decay (flat positions held too long)
  - Circuit breaker (portfolio-level drawdown)
  - Edge cases (missing data, boundaries, session reset)

Run:
  python -m pytest src/tests/test_intraday_monitor.py -v
  python -m pytest src/tests/test_intraday_monitor.py -v --tb=short
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest
import pandas as pd
import pytz

# Bootstrap module path
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from execution.intraday_monitor import IntradayMonitor, IntradayMonitorConfig
from connection.alpaca_connection import TradingMode

ET = pytz.timezone("US/Eastern")


# ═══════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def config():
    """Standard monitor config with known thresholds."""
    return IntradayMonitorConfig(
        stop_loss_pct=0.10,              # 10% hard stop
        short_stop_loss_pct=0.10,
        trailing_stop_trail_pct=0.05,    # 5% trail from peak
        trailing_stop_activation_pct=0.02,  # Activate after 2% profit
        time_decay_days=10,              # Check after 10 days
        time_decay_threshold=0.01,       # Exit if |P&L| < 1%
        circuit_breaker_drawdown_pct=0.08,  # 8% portfolio DD
        circuit_breaker_cooldown_min=60,
    )


@pytest.fixture
def monitor(config):
    """Create an IntradayMonitor with mocked connection."""
    conn = MagicMock()
    conn.config = MagicMock()
    conn.config.trading_mode = TradingMode.LIVE

    m = IntradayMonitor(
        connection=conn,
        config=config,
        shutdown_flag=lambda: False,
    )
    return m


# ═══════════════════════════════════════════════════════════════════════════
# STOP-LOSS TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestStopLoss:
    """Tests for hard stop-loss exit logic."""

    def test_long_stop_loss_triggered(self, monitor):
        """Long position down 12% should trigger stop-loss (threshold 10%)."""
        positions = {
            "AAPL": {
                "qty": 100, "side": "long",
                "entry_price": 200.0,
                "entry_date": pd.Timestamp("2026-02-20"),
            }
        }
        prices = {"AAPL": 176.0}  # -12%

        exits = monitor._check_stop_loss(positions, prices)
        assert len(exits) == 1
        assert exits[0]["symbol"] == "AAPL"
        assert exits[0]["reason"] == "stop_loss"
        assert exits[0]["pnl_pct"] == pytest.approx(-0.12, abs=0.001)

    def test_long_no_stop_loss(self, monitor):
        """Long position down 5% should NOT trigger stop-loss (threshold 10%)."""
        positions = {
            "AAPL": {
                "qty": 100, "side": "long",
                "entry_price": 200.0,
                "entry_date": pd.Timestamp("2026-02-20"),
            }
        }
        prices = {"AAPL": 190.0}  # -5%

        exits = monitor._check_stop_loss(positions, prices)
        assert len(exits) == 0

    def test_short_stop_loss_triggered(self, monitor):
        """Short position with stock up 12% should trigger stop-loss."""
        positions = {
            "TSLA": {
                "qty": 50, "side": "short",
                "entry_price": 300.0,
                "entry_date": pd.Timestamp("2026-02-20"),
            }
        }
        prices = {"TSLA": 336.0}  # +12% price rise = -12% short P&L

        exits = monitor._check_stop_loss(positions, prices)
        assert len(exits) == 1
        assert exits[0]["symbol"] == "TSLA"
        assert exits[0]["reason"] == "stop_loss"

    def test_short_no_stop_loss(self, monitor):
        """Short position with stock up 5% should NOT trigger stop-loss."""
        positions = {
            "TSLA": {
                "qty": 50, "side": "short",
                "entry_price": 300.0,
                "entry_date": pd.Timestamp("2026-02-20"),
            }
        }
        prices = {"TSLA": 315.0}  # +5%

        exits = monitor._check_stop_loss(positions, prices)
        assert len(exits) == 0

    def test_multiple_positions_mixed(self, monitor):
        """Multiple positions: only the breached ones trigger."""
        positions = {
            "AAPL": {
                "qty": 100, "side": "long",
                "entry_price": 200.0,
                "entry_date": pd.Timestamp("2026-02-20"),
            },
            "MSFT": {
                "qty": 50, "side": "long",
                "entry_price": 400.0,
                "entry_date": pd.Timestamp("2026-02-20"),
            },
            "TSLA": {
                "qty": 30, "side": "short",
                "entry_price": 300.0,
                "entry_date": pd.Timestamp("2026-02-20"),
            },
        }
        prices = {
            "AAPL": 170.0,  # -15% → triggered
            "MSFT": 395.0,  # -1.25% → safe
            "TSLA": 340.0,  # +13.3% → triggered (loss for short)
        }

        exits = monitor._check_stop_loss(positions, prices)
        assert len(exits) == 2
        triggered_symbols = {e["symbol"] for e in exits}
        assert triggered_symbols == {"AAPL", "TSLA"}

    def test_stop_loss_at_exact_threshold(self, monitor):
        """Stop-loss at exactly -10% (boundary)."""
        positions = {
            "AAPL": {
                "qty": 100, "side": "long",
                "entry_price": 200.0,
                "entry_date": pd.Timestamp("2026-02-20"),
            },
        }
        prices = {"AAPL": 180.0}  # Exactly -10%

        exits = monitor._check_stop_loss(positions, prices)
        # pnl_pct = -0.10 <= threshold -0.10 → should trigger
        assert len(exits) == 1


# ═══════════════════════════════════════════════════════════════════════════
# TRAILING STOP TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestTrailingStop:
    """Tests for trailing stop exit logic."""

    def test_trailing_stop_triggered_long(self, monitor):
        """Long: peaked at +8%, dropped to +2% → 6% drawback > 5% trail."""
        positions = {
            "AAPL": {
                "qty": 100, "side": "long",
                "entry_price": 100.0,
                "entry_date": pd.Timestamp("2026-02-20"),
            }
        }
        # Simulate peak at $108 (+8%)
        monitor._peak_prices["AAPL"] = 108.0
        prices = {"AAPL": 102.0}  # Current: +2%, drawback from peak = 6%

        exits = monitor._check_trailing_stops(positions, prices)
        assert len(exits) == 1
        assert "trailing" in exits[0]["reason"].lower()

    def test_trailing_stop_not_activated(self, monitor):
        """Position hasn't reached 2% activation threshold — no trail."""
        positions = {
            "AAPL": {
                "qty": 100, "side": "long",
                "entry_price": 100.0,
                "entry_date": pd.Timestamp("2026-02-20"),
            }
        }
        monitor._peak_prices["AAPL"] = 101.5  # Peak: +1.5% < 2%
        prices = {"AAPL": 100.5}

        exits = monitor._check_trailing_stops(positions, prices)
        assert len(exits) == 0

    def test_trailing_stop_short_triggered(self, monitor):
        """Short: peaked (price dropped to 270), then reversed to 288."""
        positions = {
            "TSLA": {
                "qty": 50, "side": "short",
                "entry_price": 300.0,
                "entry_date": pd.Timestamp("2026-02-20"),
            }
        }
        # For shorts, peak profit = lowest price seen
        monitor._peak_prices["TSLA"] = 270.0  # Peak P&L: +10%
        prices = {"TSLA": 288.0}  # Current P&L: +4%, drawback: 6%

        exits = monitor._check_trailing_stops(positions, prices)
        assert len(exits) == 1

    def test_trailing_stop_within_trail(self, monitor):
        """Drawback (3%) is within trail threshold (5%) — no exit."""
        positions = {
            "AAPL": {
                "qty": 100, "side": "long",
                "entry_price": 100.0,
                "entry_date": pd.Timestamp("2026-02-20"),
            }
        }
        monitor._peak_prices["AAPL"] = 110.0  # Peak: +10%
        prices = {"AAPL": 107.0}  # Current: +7%, drawback from peak = 3%

        exits = monitor._check_trailing_stops(positions, prices)
        assert len(exits) == 0

    def test_trailing_stop_just_at_threshold(self, monitor):
        """Drawback exactly equals trail threshold (5%)."""
        positions = {
            "AAPL": {
                "qty": 100, "side": "long",
                "entry_price": 100.0,
                "entry_date": pd.Timestamp("2026-02-20"),
            }
        }
        monitor._peak_prices["AAPL"] = 110.0  # Peak: +10%
        prices = {"AAPL": 105.0}  # Current: +5%, drawback = 5%

        exits = monitor._check_trailing_stops(positions, prices)
        # drawback (5%) >= trail (5%) → should trigger
        assert len(exits) == 1


# ═══════════════════════════════════════════════════════════════════════════
# TIME DECAY TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestTimeDecay:
    """Tests for time-decay exit logic (exit flat positions held too long)."""

    def test_time_decay_flat_long(self, monitor):
        """Long held 15 days with only 0.5% P&L → should exit."""
        positions = {
            "IBM": {
                "qty": 200, "side": "long",
                "entry_price": 150.0,
                "entry_date": pd.Timestamp.now() - pd.Timedelta(days=15),
            }
        }
        prices = {"IBM": 150.75}  # +0.5% P&L

        exits = monitor._check_time_decay(positions, prices)
        assert len(exits) == 1
        assert "time decay" in exits[0]["reason"].lower()

    def test_time_decay_too_early(self, monitor):
        """Position held only 5 days — too early for time decay (threshold 10)."""
        positions = {
            "IBM": {
                "qty": 200, "side": "long",
                "entry_price": 150.0,
                "entry_date": pd.Timestamp.now() - pd.Timedelta(days=5),
            }
        }
        prices = {"IBM": 150.50}  # +0.33%

        exits = monitor._check_time_decay(positions, prices)
        assert len(exits) == 0

    def test_time_decay_profitable_position(self, monitor):
        """Held 15 days but 5% profit — NOT flat, should NOT exit."""
        positions = {
            "IBM": {
                "qty": 200, "side": "long",
                "entry_price": 150.0,
                "entry_date": pd.Timestamp.now() - pd.Timedelta(days=15),
            }
        }
        prices = {"IBM": 157.50}  # +5% P&L >> 1% threshold

        exits = monitor._check_time_decay(positions, prices)
        assert len(exits) == 0

    def test_time_decay_short_flat(self, monitor):
        """Short held 12 days with 0.3% profit → should exit."""
        positions = {
            "NFLX": {
                "qty": 30, "side": "short",
                "entry_price": 500.0,
                "entry_date": pd.Timestamp.now() - pd.Timedelta(days=12),
            }
        }
        prices = {"NFLX": 498.50}  # Short P&L: +0.3%

        exits = monitor._check_time_decay(positions, prices)
        assert len(exits) == 1

    def test_time_decay_no_entry_date(self, monitor):
        """Position with missing entry_date should be skipped."""
        positions = {
            "GOOG": {
                "qty": 10, "side": "long",
                "entry_price": 2800.0,
                "entry_date": None,
            }
        }
        prices = {"GOOG": 2800.50}

        exits = monitor._check_time_decay(positions, prices)
        assert len(exits) == 0


# ═══════════════════════════════════════════════════════════════════════════
# CIRCUIT BREAKER TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestCircuitBreaker:
    """Tests for portfolio-level circuit breaker."""

    def test_circuit_breaker_triggered(self, monitor):
        """Portfolio drawdown exceeds 8% threshold → close all."""
        # Mock Alpaca account
        monitor.conn.get_account.return_value = {"portfolio_value": 90000.0}
        monitor._peak_equity = 100000.0  # Peak was $100k, now $90k = -10%

        positions = {
            "AAPL": {
                "qty": 100, "side": "long",
                "entry_price": 150.0,
                "entry_date": pd.Timestamp("2026-02-20"),
            },
            "MSFT": {
                "qty": 50, "side": "long",
                "entry_price": 400.0,
                "entry_date": pd.Timestamp("2026-02-20"),
            },
        }
        prices = {"AAPL": 130.0, "MSFT": 350.0}

        exits = monitor._check_circuit_breaker(positions, prices)
        assert len(exits) == 2  # Should close ALL positions
        assert monitor._circuit_breaker_fired is True

    def test_circuit_breaker_not_triggered(self, monitor):
        """Portfolio drawdown is 3% — below 8% threshold."""
        monitor.conn.get_account.return_value = {"portfolio_value": 97000.0}
        monitor._peak_equity = 100000.0

        positions = {
            "AAPL": {
                "qty": 100, "side": "long",
                "entry_price": 150.0,
                "entry_date": pd.Timestamp("2026-02-20"),
            },
        }
        prices = {"AAPL": 148.0}

        exits = monitor._check_circuit_breaker(positions, prices)
        assert len(exits) == 0
        assert monitor._circuit_breaker_fired is False

    def test_circuit_breaker_cooldown(self, monitor):
        """Circuit breaker recently fired — should be in cooldown."""
        monitor._circuit_breaker_fired = True
        monitor._circuit_breaker_time = datetime.now(ET)  # Just fired

        monitor.conn.get_account.return_value = {"portfolio_value": 85000.0}
        monitor._peak_equity = 100000.0

        positions = {
            "AAPL": {
                "qty": 100, "side": "long",
                "entry_price": 150.0,
                "entry_date": pd.Timestamp("2026-02-20"),
            },
        }
        prices = {"AAPL": 120.0}

        exits = monitor._check_circuit_breaker(positions, prices)
        assert len(exits) == 0  # In cooldown, should not fire again

    def test_circuit_breaker_equity_tracking(self, monitor):
        """Peak equity should be updated when portfolio grows."""
        monitor.conn.get_account.return_value = {"portfolio_value": 105000.0}
        monitor._peak_equity = 100000.0

        positions = {
            "AAPL": {
                "qty": 100, "side": "long",
                "entry_price": 150.0,
                "entry_date": pd.Timestamp("2026-02-20"),
            },
        }
        prices = {"AAPL": 160.0}

        exits = monitor._check_circuit_breaker(positions, prices)
        assert len(exits) == 0
        # Peak should have been updated to the higher value
        assert monitor._peak_equity == 105000.0


# ═══════════════════════════════════════════════════════════════════════════
# PEAK PRICE TRACKING TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestPeakTracking:
    """Tests for the peak price update logic used by trailing stops."""

    def test_long_peak_updated_on_higher_price(self, monitor):
        """Long position: peak should track the highest price."""
        positions = {
            "AAPL": {
                "qty": 100, "side": "long",
                "entry_price": 100.0,
                "entry_date": pd.Timestamp("2026-02-20"),
            }
        }
        monitor._peak_prices["AAPL"] = 105.0

        # Update with a higher price
        monitor._update_peaks(positions, {"AAPL": 110.0})
        assert monitor._peak_prices["AAPL"] == 110.0

        # Update with a lower price — peak should NOT decrease
        monitor._update_peaks(positions, {"AAPL": 107.0})
        assert monitor._peak_prices["AAPL"] == 110.0

    def test_short_peak_tracks_lowest_price(self, monitor):
        """Short position: 'peak P&L' means lowest price seen."""
        positions = {
            "TSLA": {
                "qty": 50, "side": "short",
                "entry_price": 300.0,
                "entry_date": pd.Timestamp("2026-02-20"),
            }
        }
        monitor._peak_prices["TSLA"] = 285.0

        # Price drops further — peak should update
        monitor._update_peaks(positions, {"TSLA": 275.0})
        assert monitor._peak_prices["TSLA"] == 275.0

        # Price rises — peak should NOT change
        monitor._update_peaks(positions, {"TSLA": 290.0})
        assert monitor._peak_prices["TSLA"] == 275.0


# ═══════════════════════════════════════════════════════════════════════════
# EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_missing_price_data(self, monitor):
        """Position with no price data available should be skipped."""
        positions = {
            "AAPL": {
                "qty": 100, "side": "long",
                "entry_price": 200.0,
                "entry_date": pd.Timestamp("2026-02-20"),
            },
        }
        prices = {}  # No price data

        assert len(monitor._check_stop_loss(positions, prices)) == 0
        assert len(monitor._check_trailing_stops(positions, prices)) == 0
        assert len(monitor._check_time_decay(positions, prices)) == 0

    def test_no_positions(self, monitor):
        """No positions — all checks should return empty."""
        positions = {}
        prices = {}

        assert len(monitor._check_stop_loss(positions, prices)) == 0
        assert len(monitor._check_trailing_stops(positions, prices)) == 0
        assert len(monitor._check_time_decay(positions, prices)) == 0

    def test_session_reset(self, monitor):
        """reset_session should clear all daily tracking state."""
        monitor._exited_today = {"AAPL", "MSFT"}
        monitor._peak_prices = {"AAPL": 200.0}
        monitor._peak_equity = 100000.0
        monitor._circuit_breaker_fired = True

        monitor.reset_session()

        assert len(monitor._exited_today) == 0
        assert len(monitor._peak_prices) == 0
        assert monitor._peak_equity is None
        assert monitor._circuit_breaker_fired is False

    def test_profitable_long_no_stop_loss(self, monitor):
        """Profitable long position should never trigger stop-loss."""
        positions = {
            "AAPL": {
                "qty": 100, "side": "long",
                "entry_price": 100.0,
                "entry_date": pd.Timestamp("2026-02-20"),
            },
        }
        prices = {"AAPL": 120.0}  # +20%

        exits = monitor._check_stop_loss(positions, prices)
        assert len(exits) == 0

    def test_profitable_short_no_stop_loss(self, monitor):
        """Profitable short position should never trigger stop-loss."""
        positions = {
            "TSLA": {
                "qty": 50, "side": "short",
                "entry_price": 300.0,
                "entry_date": pd.Timestamp("2026-02-20"),
            },
        }
        prices = {"TSLA": 260.0}  # Price dropped → short profits

        exits = monitor._check_stop_loss(positions, prices)
        assert len(exits) == 0


# ═══════════════════════════════════════════════════════════════════════════
# BRACKET PRICE COMPUTATION TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestBracketPrices:
    """Tests for the static bracket price calculator."""

    def test_long_bracket_prices(self):
        """Long bracket: SL below, TP above entry."""
        result = IntradayMonitor.compute_bracket_prices(
            entry_price=100.0, side="long",
            stop_loss_pct=0.08, take_profit_pct=0.15,
        )
        assert result["stop_loss_price"] == 92.0   # 100 * (1 - 0.08)
        assert result["take_profit_price"] == 115.0  # 100 * (1 + 0.15)

    def test_short_bracket_prices(self):
        """Short bracket: SL above, TP below entry."""
        result = IntradayMonitor.compute_bracket_prices(
            entry_price=200.0, side="short",
            stop_loss_pct=0.10, take_profit_pct=0.20,
        )
        assert result["stop_loss_price"] == 220.0   # 200 * (1 + 0.10)
        assert result["take_profit_price"] == 160.0  # 200 * (1 - 0.20)

    def test_bracket_no_take_profit(self):
        """Bracket without take-profit leg."""
        result = IntradayMonitor.compute_bracket_prices(
            entry_price=150.0, side="long",
            stop_loss_pct=0.05, take_profit_pct=None,
        )
        assert result["stop_loss_price"] == 142.5  # 150 * (1 - 0.05)
        assert result["take_profit_price"] is None
