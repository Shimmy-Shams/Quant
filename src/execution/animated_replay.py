"""
Animated Historical Replay Engine

Replays the full trading pipeline day-by-day with live-updating
Plotly FigureWidget charts and scrolling execution logs — replicating
the VM trading experience visually inside a Jupyter notebook.

Usage (inside a notebook cell):
    replay = AnimatedReplay(config, bt_config, price_df, signal_df,
                            volume_df, zscore_df, analysis_df)
    replay.run(start_date='2024-02-01', end_date='2026-02-01')
"""

import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets

from execution.alpaca_executor import AlpacaExecutor, TradeDecision
from execution.simulation import SimulationEngine, SimulatedPosition, DailySnapshot

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ReplayEvent:
    """A single event during replay for the log feed"""
    timestamp: pd.Timestamp
    event_type: str          # 'entry', 'exit', 'stop_placed', 'signal', 'skip', 'info'
    symbol: str
    message: str
    icon: str
    details: Optional[str] = None
    pnl: Optional[float] = None


# ═══════════════════════════════════════════════════════════════════════════
# ANIMATED REPLAY ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class AnimatedReplay:
    """
    Full-pipeline animated replay with live charts and execution logs.

    Replicates the VM trading loop:
    1. Load day's data
    2. Generate signals (pre-computed)
    3. Check exits (stop-loss, trailing, time-decay, signal)
    4. Check entries (threshold, sentiment filter, position sizing)
    5. Simulate order execution (market → fill → GTC stop)
    6. Update portfolio state
    7. Animate charts and log events
    """

    def __init__(
        self,
        config,
        bt_config,
        price_df: pd.DataFrame,
        signal_df: pd.DataFrame,
        volume_df: pd.DataFrame,
        zscore_df: pd.DataFrame,
        analysis_df: pd.DataFrame,
        initial_capital: float = 1_000_000,
    ):
        """
        Args:
            config: ConfigLoader instance
            bt_config: BacktestConfig instance
            price_df: Historical prices (DatetimeIndex × symbols)
            signal_df: Pre-computed composite signals
            volume_df: Volume data
            zscore_df: Z-score exit signals
            analysis_df: Universe analysis (is_mean_reverting filter)
            initial_capital: Starting capital
        """
        self.config = config
        self.bt_config = bt_config
        self.price_df = price_df
        self.signal_df = signal_df
        self.volume_df = volume_df
        self.zscore_df = zscore_df
        self.analysis_df = analysis_df
        self.initial_capital = initial_capital

        # Replay state
        self.events: List[ReplayEvent] = []
        self.equity_history: List[Tuple[pd.Timestamp, float]] = []
        self.position_history: List[Tuple[pd.Timestamp, int, int]] = []  # date, longs, shorts
        self.trade_markers: List[dict] = []  # For chart annotations

    def run(
        self,
        start_date: str = '2024-02-01',
        end_date: str = '2026-02-01',
        speed_multiplier: float = 1.0,
    ):
        """
        Run the animated replay.

        Args:
            start_date: Replay start date (YYYY-MM-DD)
            end_date: Replay end date (YYYY-MM-DD)
            speed_multiplier: 1.0 = default (~2 min), <1 = faster, >1 = slower
        """
        # ── Setup simulation engine ──
        from connection.alpaca_connection import AlpacaConfig, AlpacaConnection, TradingMode

        alpaca_cfg = AlpacaConfig(
            api_key='REPLAY',
            secret_key='REPLAY',
            paper=True,
            trading_mode=TradingMode.REPLAY,
        )
        conn = AlpacaConnection(alpaca_cfg)

        executor = AlpacaExecutor(
            connection=conn,
            commission_pct=self.bt_config.commission_pct,
            max_position_pct=self.bt_config.max_position_size,
            max_total_exposure=self.bt_config.max_total_exposure,
            stop_loss_pct=getattr(self.bt_config, 'stop_loss_pct', None),
            take_profit_pct=getattr(self.bt_config, 'take_profit_pct', None),
        )

        sim = SimulationEngine(
            executor=executor,
            initial_capital=self.initial_capital,
            commission_pct=self.bt_config.commission_pct,
            slippage_pct=getattr(self.bt_config, 'slippage_pct', 0.0005),
        )

        # ── Date range ──
        dates = self.signal_df.index.sort_values()
        mask = (dates >= pd.Timestamp(start_date)) & (dates <= pd.Timestamp(end_date))
        dates = dates[mask]

        if len(dates) == 0:
            print("No trading days in the specified range.")
            return

        # ── Build widgets ──
        fig, log_output, progress_bar, stats_html, controls_box = self._build_ui(dates)
        display(controls_box)

        # ── Timing: adaptive frame rate ──
        quiet_batch = 5                # batch N quiet days per frame
        quiet_delay = 0.10 * speed_multiplier
        signal_delay = 0.20 * speed_multiplier
        trade_delay = 0.70 * speed_multiplier
        stop_event_delay = 0.80 * speed_multiplier

        # Pre-compute which days have entries above threshold
        entry_threshold = self.bt_config.entry_threshold

        # ── Run replay loop ──
        self.events = []
        self.equity_history = []
        self.position_history = []
        self.trade_markers = []

        prev_equity = self.initial_capital
        total_days = len(dates)
        batch_start = 0

        i = 0
        while i < total_days:
            date = dates[i]

            if date not in self.price_df.index:
                i += 1
                continue

            today_prices = self.price_df.loc[date].dropna()

            # Update position prices and peak prices
            for symbol, pos in sim.positions.items():
                if symbol in today_prices.index:
                    pos.current_price = today_prices[symbol]
                    if pos.peak_price == 0:
                        pos.peak_price = pos.entry_price
                    if pos.side == 'long':
                        pos.peak_price = max(pos.peak_price, pos.current_price)
                    else:
                        pos.peak_price = min(pos.peak_price, pos.current_price)

            # Generate decisions
            decisions = executor.generate_decisions_from_signals(
                signal_df=self.signal_df,
                price_df=self.price_df,
                volume_df=self.volume_df,
                exit_signal_df=self.zscore_df,
                date=date,
                current_positions=sim.positions_as_dict,
                config=self.bt_config,
                current_equity=sim.equity,
            )

            # Classify day
            has_trades = len(decisions) > 0
            today_signals = self.signal_df.loc[date].dropna()
            has_strong_signals = (today_signals.abs() > entry_threshold * 0.8).any()

            # ── Process exits ──
            exit_decisions = [d for d in decisions if d.action in ('sell', 'cover')]
            entry_decisions = [d for d in decisions if d.action in ('buy', 'short')]

            for decision in exit_decisions:
                pos = sim.positions.get(decision.symbol)
                if pos:
                    exit_price = today_prices.get(decision.symbol, pos.current_price)
                    if pos.side == 'long':
                        pnl = (exit_price - pos.entry_price) * pos.qty
                        pnl_pct = (exit_price - pos.entry_price) / pos.entry_price * 100
                    else:
                        pnl = (pos.entry_price - exit_price) * abs(pos.qty)
                        pnl_pct = (pos.entry_price - exit_price) / pos.entry_price * 100

                    icon = '🗑️✅' if pnl >= 0 else '🗑️❌'
                    self.events.append(ReplayEvent(
                        timestamp=date,
                        event_type='exit',
                        symbol=decision.symbol,
                        message=(
                            f"EXIT {decision.action.upper()} {abs(decision.target_qty)} "
                            f"{decision.symbol} @ ${exit_price:.2f}"
                        ),
                        icon=icon,
                        details=(
                            f"  Reason: {decision.reason}\n"
                            f"  Entry: ${pos.entry_price:.2f} → Exit: ${exit_price:.2f}\n"
                            f"  P&L: ${pnl:+,.0f} ({pnl_pct:+.2f}%) | "
                            f"Held: {(date - pos.entry_date).days}d"
                        ),
                        pnl=pnl,
                    ))

                    self.trade_markers.append({
                        'date': date, 'symbol': decision.symbol,
                        'price': exit_price, 'type': 'exit',
                        'pnl': pnl, 'side': pos.side,
                    })

                sim._process_exit(decision, date, today_prices)

            # ── Process entries ──
            for decision in entry_decisions:
                price = today_prices.get(decision.symbol, 0)
                if price <= 0:
                    continue

                side = 'long' if decision.action == 'buy' else 'short'

                # Log the entry
                self.events.append(ReplayEvent(
                    timestamp=date,
                    event_type='entry',
                    symbol=decision.symbol,
                    message=(
                        f"ENTRY {decision.action.upper()} {decision.target_qty} "
                        f"{decision.symbol} @ ${price:.2f}"
                    ),
                    icon='🟢' if side == 'long' else '🔴',
                    details=(
                        f"  Signal: {decision.signal_strength:.3f} | "
                        f"Size: ${decision.target_qty * price:,.0f}\n"
                        f"  Reason: {decision.reason}"
                    ),
                ))

                # Log stop-loss placement (Option D behavior)
                sl_pct = self.bt_config.stop_loss_pct
                if sl_pct:
                    if side == 'long':
                        sl_price = round(price * (1 - sl_pct), 2)
                        sl_side = 'SELL'
                    else:
                        sl_price = round(price * (1 + sl_pct), 2)
                        sl_side = 'BUY'

                    self.events.append(ReplayEvent(
                        timestamp=date,
                        event_type='stop_placed',
                        symbol=decision.symbol,
                        message=(
                            f"🛡️ GTC Stop placed: {sl_side} {decision.target_qty} "
                            f"{decision.symbol} @ ${sl_price:.2f} "
                            f"(entry=${price:.2f}, {sl_pct:.0%})"
                        ),
                        icon='🛡️',
                    ))

                self.trade_markers.append({
                    'date': date, 'symbol': decision.symbol,
                    'price': price, 'type': 'entry', 'side': side,
                })

                sim._process_entry(decision, date, today_prices)

            # ── Snapshot ──
            current_equity = sim.equity
            daily_pnl = current_equity - prev_equity
            daily_return = daily_pnl / prev_equity if prev_equity > 0 else 0

            self.equity_history.append((date, current_equity))
            n_longs = sum(1 for p in sim.positions.values() if p.side == 'long')
            n_shorts = sum(1 for p in sim.positions.values() if p.side == 'short')
            self.position_history.append((date, n_longs, n_shorts))

            # Record DailySnapshot for _compile_results()
            positions_value = sum(p.market_value for p in sim.positions.values())
            sim.daily_snapshots.append(DailySnapshot(
                date=date,
                equity=current_equity,
                cash=sim.cash,
                positions_value=positions_value,
                n_positions=len(sim.positions),
                n_longs=n_longs,
                n_shorts=n_shorts,
                daily_pnl=daily_pnl,
                daily_return_pct=daily_return * 100,
                trades_entered=len(entry_decisions),
                trades_exited=len(exit_decisions),
                signals_generated=len(today_signals.dropna()) if has_strong_signals else 0,
                entries_above_threshold=int((today_signals.abs() > entry_threshold).sum()) if has_strong_signals else 0,
            ))

            prev_equity = current_equity

            # ── Determine frame speed and whether to update UI ──
            if has_trades:
                # Trade day: always update UI with full detail
                self._update_chart(fig, date)
                self._update_log(log_output, max_events=30)
                self._update_stats(stats_html, sim, date, i, total_days, daily_pnl)
                progress_bar.value = i + 1

                delay = trade_delay
                if any(e.event_type == 'stop_placed' for e in self.events[-5:]):
                    delay = stop_event_delay
                time.sleep(delay)
            elif has_strong_signals:
                # Near-threshold day
                self._update_chart(fig, date)
                self._update_stats(stats_html, sim, date, i, total_days, daily_pnl)
                progress_bar.value = i + 1
                time.sleep(signal_delay)
            else:
                # Quiet day: batch updates
                if (i - batch_start) >= quiet_batch or i == total_days - 1:
                    self._update_chart(fig, date)
                    self._update_stats(stats_html, sim, date, i, total_days, daily_pnl)
                    progress_bar.value = i + 1
                    time.sleep(quiet_delay)
                    batch_start = i

            i += 1

        # ── Final summary ──
        results = sim._compile_results()
        self._show_final_summary(log_output, results, sim)
        self._update_chart(fig, dates[-1])

        return results, sim

    # ═══════════════════════════════════════════════════════════════════
    # UI CONSTRUCTION
    # ═══════════════════════════════════════════════════════════════════

    def _build_ui(self, dates):
        """Build the interactive dashboard widgets."""

        # ── Equity curve chart (FigureWidget for in-place updates) ──
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.7, 0.3],
            vertical_spacing=0.05,
            subplot_titles=['Portfolio Equity', 'Positions (Long / Short)'],
        )

        # Equity trace
        fig.add_trace(go.Scatter(
            x=[], y=[], mode='lines',
            name='Equity',
            line=dict(color='#667eea', width=2),
        ), row=1, col=1)

        # Entry markers
        fig.add_trace(go.Scatter(
            x=[], y=[], mode='markers',
            name='Entry',
            marker=dict(color='#3fb950', size=8, symbol='triangle-up'),
        ), row=1, col=1)

        # Exit markers (win)
        fig.add_trace(go.Scatter(
            x=[], y=[], mode='markers',
            name='Exit (win)',
            marker=dict(color='#58a6ff', size=8, symbol='triangle-down'),
        ), row=1, col=1)

        # Exit markers (loss)
        fig.add_trace(go.Scatter(
            x=[], y=[], mode='markers',
            name='Exit (loss)',
            marker=dict(color='#da3633', size=8, symbol='triangle-down'),
        ), row=1, col=1)

        # Positions bar chart (longs/shorts)
        fig.add_trace(go.Bar(
            x=[], y=[], name='Longs',
            marker_color='#238636', opacity=0.7,
        ), row=2, col=1)

        fig.add_trace(go.Bar(
            x=[], y=[], name='Shorts',
            marker_color='#da3633', opacity=0.7,
        ), row=2, col=1)

        fig.update_layout(
            height=500,
            template='plotly_dark',
            paper_bgcolor='#0d1117',
            plot_bgcolor='#161b22',
            margin=dict(l=60, r=20, t=40, b=20),
            legend=dict(orientation='h', x=0, y=1.12),
            barmode='relative',
            showlegend=True,
        )
        fig.update_yaxes(title_text='Equity ($)', row=1, col=1, gridcolor='#21262d')
        fig.update_yaxes(title_text='Count', row=2, col=1, gridcolor='#21262d')
        fig.update_xaxes(gridcolor='#21262d')

        fig_widget = go.FigureWidget(fig)

        # ── Execution log ──
        log_output = widgets.Output(layout=widgets.Layout(
            height='300px',
            overflow_y='auto',
            border='1px solid #30363d',
            padding='8px',
        ))

        # ── Progress bar ──
        progress_bar = widgets.IntProgress(
            value=0, min=0, max=len(dates),
            description='Replay:',
            bar_style='info',
            style={'bar_color': '#667eea'},
            layout=widgets.Layout(width='100%'),
        )

        # ── Stats panel ──
        stats_html = widgets.HTML(value=self._stats_html_template(
            equity=self.initial_capital,
            daily_pnl=0, date=dates[0],
            n_positions=0, n_longs=0, n_shorts=0,
            total_return=0, day_num=0, total_days=len(dates),
            n_trades=0,
        ))

        # ── Layout ──
        header = widgets.HTML(value="""
        <div style="background:#161b22; border:1px solid #30363d; border-radius:8px;
                    padding:12px 16px; margin-bottom:8px; color:#c9d1d9;">
            <span style="font-size:1.3em; font-weight:700; color:#667eea;">
                📼 Historical Replay</span>
            <span style="color:#8b949e; margin-left:12px;">
                Simulating VM trading pipeline day-by-day</span>
        </div>
        """)

        # Top: chart + stats side by side
        chart_panel = widgets.VBox(
            [fig_widget],
            layout=widgets.Layout(flex='3'),
        )
        stats_panel = widgets.VBox(
            [stats_html],
            layout=widgets.Layout(
                flex='1', min_width='220px',
            ),
        )
        top_row = widgets.HBox(
            [chart_panel, stats_panel],
            layout=widgets.Layout(width='100%'),
        )

        # Log section
        log_label = widgets.HTML(value="""
        <div style="color:#8b949e; font-size:0.9em; font-weight:600;
                    margin:8px 0 4px 0;">
            📋 Execution Log (VM-style)
        </div>
        """)

        controls_box = widgets.VBox([
            header,
            progress_bar,
            top_row,
            log_label,
            log_output,
        ])

        return fig_widget, log_output, progress_bar, stats_html, controls_box

    def _stats_html_template(
        self, equity, daily_pnl, date, n_positions, n_longs, n_shorts,
        total_return, day_num, total_days, n_trades,
    ):
        """Generate the stats panel HTML."""
        pnl_color = '#3fb950' if daily_pnl >= 0 else '#da3633'
        ret_color = '#3fb950' if total_return >= 0 else '#da3633'
        return f"""
        <div style="background:#161b22; border:1px solid #30363d; border-radius:8px;
                    padding:12px; color:#c9d1d9; font-size:0.85em; line-height:1.8;">
            <div style="font-weight:700; color:#667eea; margin-bottom:8px;">
                📊 Live Stats</div>
            <div>📅 <b>{date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else date}</b></div>
            <div>Day <b>{day_num + 1}</b> / {total_days}</div>
            <hr style="border-color:#30363d; margin:6px 0;">
            <div>💰 Equity: <b>${equity:,.0f}</b></div>
            <div>📈 Return: <span style="color:{ret_color}; font-weight:600;">{total_return:+.2f}%</span></div>
            <div>📊 Daily P&L: <span style="color:{pnl_color};">${daily_pnl:+,.0f}</span></div>
            <hr style="border-color:#30363d; margin:6px 0;">
            <div>📦 Positions: <b>{n_positions}</b></div>
            <div style="margin-left:12px;">
                <span style="color:#238636;">▲ {n_longs}L</span> /
                <span style="color:#da3633;">▼ {n_shorts}S</span>
            </div>
            <div>🔄 Total Trades: <b>{n_trades}</b></div>
        </div>
        """

    # ═══════════════════════════════════════════════════════════════════
    # UI UPDATE METHODS
    # ═══════════════════════════════════════════════════════════════════

    def _update_chart(self, fig_widget, current_date):
        """Update equity chart and position bars in-place."""
        if not self.equity_history:
            return

        dates = [e[0] for e in self.equity_history]
        equities = [e[1] for e in self.equity_history]

        # Entry markers on equity curve
        entry_dates = []
        entry_equities = []
        exit_win_dates = []
        exit_win_equities = []
        exit_loss_dates = []
        exit_loss_equities = []

        equity_lookup = dict(self.equity_history)

        for m in self.trade_markers:
            eq_val = equity_lookup.get(m['date'], None)
            if eq_val is None:
                continue
            if m['type'] == 'entry':
                entry_dates.append(m['date'])
                entry_equities.append(eq_val)
            elif m['type'] == 'exit':
                if m.get('pnl', 0) >= 0:
                    exit_win_dates.append(m['date'])
                    exit_win_equities.append(eq_val)
                else:
                    exit_loss_dates.append(m['date'])
                    exit_loss_equities.append(eq_val)

        # Position bars
        pos_dates = [p[0] for p in self.position_history]
        longs = [p[1] for p in self.position_history]
        shorts = [-p[2] for p in self.position_history]  # Negative for visual

        with fig_widget.batch_update():
            fig_widget.data[0].x = dates
            fig_widget.data[0].y = equities
            fig_widget.data[1].x = entry_dates
            fig_widget.data[1].y = entry_equities
            fig_widget.data[2].x = exit_win_dates
            fig_widget.data[2].y = exit_win_equities
            fig_widget.data[3].x = exit_loss_dates
            fig_widget.data[3].y = exit_loss_equities
            fig_widget.data[4].x = pos_dates
            fig_widget.data[4].y = longs
            fig_widget.data[5].x = pos_dates
            fig_widget.data[5].y = shorts

    def _update_log(self, log_output, max_events=30):
        """Update the scrolling execution log."""
        recent = self.events[-max_events:]
        log_output.clear_output(wait=True)
        with log_output:
            for event in recent:
                # Color based on type
                if event.event_type == 'entry':
                    color = '#3fb950' if event.icon == '🟢' else '#da3633'
                    bg = 'rgba(35,134,54,0.1)' if event.icon == '🟢' else 'rgba(218,54,51,0.1)'
                elif event.event_type == 'exit':
                    color = '#58a6ff' if (event.pnl and event.pnl >= 0) else '#f85149'
                    bg = 'rgba(88,166,255,0.05)' if (event.pnl and event.pnl >= 0) else 'rgba(248,81,73,0.05)'
                elif event.event_type == 'stop_placed':
                    color = '#d29922'
                    bg = 'rgba(210,153,34,0.05)'
                else:
                    color = '#8b949e'
                    bg = 'transparent'

                date_str = event.timestamp.strftime('%Y-%m-%d')
                html = f"""
                <div style="border-left:3px solid {color}; padding:4px 8px;
                            margin:2px 0; background:{bg}; font-family:monospace;
                            font-size:0.82em; color:#c9d1d9;">
                    <span style="color:#8b949e;">{date_str}</span>
                    {event.icon} <b>{event.message}</b>
                """
                if event.details:
                    html += f'<div style="color:#8b949e; margin-left:20px; white-space:pre;">{event.details}</div>'
                html += '</div>'
                display(HTML(html))

    def _update_stats(self, stats_html, sim, date, day_num, total_days, daily_pnl):
        """Update the stats panel."""
        equity = sim.equity
        total_return = (equity / self.initial_capital - 1) * 100
        n_longs = sum(1 for p in sim.positions.values() if p.side == 'long')
        n_shorts = sum(1 for p in sim.positions.values() if p.side == 'short')

        stats_html.value = self._stats_html_template(
            equity=equity,
            daily_pnl=daily_pnl,
            date=date,
            n_positions=len(sim.positions),
            n_longs=n_longs,
            n_shorts=n_shorts,
            total_return=total_return,
            day_num=day_num,
            total_days=total_days,
            n_trades=len(sim.completed_trades),
        )

    def _show_final_summary(self, log_output, results, sim):
        """Show final summary in the log output."""
        with log_output:
            # Separator
            display(HTML("""
            <div style="border-top:2px solid #667eea; margin:12px 0; padding-top:12px;">
                <span style="font-size:1.1em; font-weight:700; color:#667eea;">
                    🏁 REPLAY COMPLETE
                </span>
            </div>
            """))

            summary_html = f"""
            <div style="font-family:monospace; font-size:0.85em; color:#c9d1d9;
                        background:#161b22; border:1px solid #30363d;
                        border-radius:8px; padding:12px; margin:4px 0;">
                <div style="font-weight:700; margin-bottom:8px;">Performance Summary</div>
                <div>💰 Final Equity:  <b>${results['final_equity']:,.2f}</b></div>
                <div>📈 Total Return:  <b>{results['total_return_pct']:+.2f}%</b></div>
                <div>📊 Sharpe Ratio:  <b>{results['sharpe_ratio']:.2f}</b></div>
                <div>📉 Max Drawdown:  <b>{results['max_drawdown_pct']:.2f}%</b></div>
                <div>🔄 Total Trades:  <b>{results['total_trades']}</b></div>
                <div>✅ Win Rate:      <b>{results['win_rate']:.1f}%</b></div>
                <div>📈 Avg Winner:    <b>{results['avg_win_pct']:+.2f}%</b></div>
                <div>📉 Avg Loser:     <b>{results['avg_loss_pct']:+.2f}%</b></div>
                <div>📦 Open Positions: <b>{results['open_positions']}</b></div>
            </div>
            """
            display(HTML(summary_html))

            # Show open positions if any
            if sim.positions:
                pos_html = """
                <div style="font-family:monospace; font-size:0.82em; color:#c9d1d9;
                            margin-top:8px;">
                    <div style="font-weight:600; margin-bottom:4px;">Open Positions:</div>
                """
                for sym, pos in sim.positions.items():
                    if pos.side == 'long':
                        pnl_pct = (pos.current_price - pos.entry_price) / pos.entry_price * 100
                    else:
                        pnl_pct = (pos.entry_price - pos.current_price) / pos.entry_price * 100
                    pnl_color = '#3fb950' if pnl_pct >= 0 else '#da3633'
                    side_color = '#238636' if pos.side == 'long' else '#da3633'
                    pos_html += f"""
                    <div style="margin-left:8px;">
                        <span style="color:{side_color}; font-weight:600;">{pos.side.upper()}</span>
                        <b>{sym}</b> x{abs(pos.qty)}
                        | Entry: ${pos.entry_price:.2f} → ${pos.current_price:.2f}
                        | <span style="color:{pnl_color};">{pnl_pct:+.2f}%</span>
                    </div>
                    """
                pos_html += '</div>'
                display(HTML(pos_html))

            # Trade breakdown by exit reason
            if results.get('trades_df') is not None and len(results['trades_df']) > 0:
                trades_df = results['trades_df']
                display(HTML("""
                <div style="font-family:monospace; font-size:0.82em; color:#c9d1d9;
                            margin-top:8px;">
                    <div style="font-weight:600; margin-bottom:4px;">Exit Reasons:</div>
                """))
                for reason, group in trades_df.groupby('exit_reason'):
                    avg_pnl = group['pnl_pct'].mean() * 100
                    color = '#3fb950' if avg_pnl >= 0 else '#da3633'
                    display(HTML(f"""
                    <div style="margin-left:8px; font-family:monospace; font-size:0.82em;">
                        {reason}: <b>{len(group)}</b> trades,
                        avg <span style="color:{color};">{avg_pnl:+.2f}%</span>
                    </div>
                    """))
