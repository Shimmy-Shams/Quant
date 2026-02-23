"""
News Sentiment — Tier 1 News Event Integration (Sentiment Penalty)

Instead of a binary "block trade if negative news" approach, this module
uses a **sentiment penalty system**:

  - Fetch recent headlines for a symbol (last 48h)
  - Score each headline using FinBERT (financial NLP model) or a
    keyword-based fallback
  - Compute an aggregate sentiment score (-1.0 to +1.0)
  - Return a **position size multiplier** (0.5 to 1.0):
      - Strongly negative sentiment → 0.5× (halve position)
      - Neutral / positive → 1.0× (no penalty)

Design principles:
  - No trade is fully blocked by sentiment alone — only position size is adjusted
  - The penalty is conservative (50% min, not 0%)
  - For backtest: uses historical Alpaca news (if available via API) or
    synthetic sentiment from price action as a proxy
  - Sentiment scores are cached (1-hour TTL) to avoid rate limit issues

Data sources:
  - Alpaca News API (free, included with trading account)
  - FinBERT sentiment model (optional, pip install transformers torch)
  - Keyword-based fallback (zero dependencies)

Usage:
    from data.news_sentiment import NewsSentiment

    ns = NewsSentiment(project_root)
    multiplier = ns.get_sentiment_multiplier("CRWD")
    adjusted_size = base_position_size * multiplier
"""

import json
import logging
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger("news_sentiment")

# ── Keyword-based sentiment (zero-dependency fallback) ──────────────────

# Strongly negative keywords (likely material impact)
_NEGATIVE_KEYWORDS = [
    r"\bfraud\b", r"\blawsuit\b", r"\bsec\s+investigat", r"\bindictment\b",
    r"\brecall\b", r"\bdata\s+breach\b", r"\bhack\b", r"\bcyber\s*attack\b",
    r"\bbankrupt", r"\bdefault\b", r"\bdowngrade\b", r"\bdelisted\b",
    r"\bclass\s+action\b", r"\bwhistleblower\b", r"\bsubpoena\b",
    r"\bmisses?\s+estimates?\b", r"\bmisses?\s+expectations?\b",
    r"\brevenue\s+miss\b", r"\bearnings?\s+miss\b",
    r"\bprofit\s+warning\b", r"\bguidance\s+cut\b", r"\blower\w*\s+guidance\b",
    r"\bceo\s+resign", r"\bcfo\s+resign", r"\bceo\s+depart",
    r"\bceo\s+fired\b", r"\bceo\s+ousted\b",
    r"\bfda\s+reject", r"\bfda\s+fail", r"\bclinical\s+trial\s+fail",
    r"\bshort\s+seller\b", r"\bshort\s+report\b",
]

# Positive keywords (likely favorable, used to offset)
_POSITIVE_KEYWORDS = [
    r"\bbeat\w*\s+estimates?\b", r"\bbeat\w*\s+expectations?\b",
    r"\bupgrade\b", r"\bstrong\s+earnings\b", r"\brecord\s+revenue\b",
    r"\bfda\s+approv", r"\bbuyback\b", r"\brepurchase\b",
    r"\bdividend\s+increas", r"\bracquir\b", r"\bmerger\b",
    r"\bguidance\s+rais", r"\bhigher\s+guidance\b",
]

_NEG_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _NEGATIVE_KEYWORDS]
_POS_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _POSITIVE_KEYWORDS]


def keyword_sentiment(headline: str) -> float:
    """
    Simple keyword-based sentiment scorer.

    Returns:
        Score between -1.0 and 1.0. Negative = bearish, positive = bullish.
    """
    neg_hits = sum(1 for p in _NEG_PATTERNS if p.search(headline))
    pos_hits = sum(1 for p in _POS_PATTERNS if p.search(headline))

    if neg_hits == 0 and pos_hits == 0:
        return 0.0  # Neutral

    # Net score, clamped to [-1, 1]
    raw = (pos_hits - neg_hits) / max(neg_hits + pos_hits, 1)
    return max(-1.0, min(1.0, raw))


# ── FinBERT Sentiment (optional, higher accuracy) ──────────────────────

_finbert_pipeline = None
_finbert_available = None


def _load_finbert():
    """Lazy-load FinBERT model. Returns None if not available."""
    global _finbert_pipeline, _finbert_available
    if _finbert_available is not None:
        return _finbert_pipeline

    try:
        from transformers import pipeline as hf_pipeline

        _finbert_pipeline = hf_pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            device=-1,  # CPU only (no GPU needed for our volume)
            truncation=True,
            max_length=512,
        )
        _finbert_available = True
        logger.info("FinBERT model loaded successfully")
        return _finbert_pipeline
    except Exception as e:
        _finbert_available = False
        logger.info(f"FinBERT not available ({e}), using keyword fallback")
        return None


def finbert_sentiment(headline: str) -> float:
    """
    Score a headline using FinBERT.

    Returns:
        Score between -1.0 and 1.0. Returns 0.0 if FinBERT unavailable.
    """
    pipe = _load_finbert()
    if pipe is None:
        return keyword_sentiment(headline)

    try:
        result = pipe(headline[:512])[0]
        label = result["label"].lower()
        score = result["score"]

        if label == "negative":
            return -score
        elif label == "positive":
            return score
        else:
            return 0.0
    except Exception:
        return keyword_sentiment(headline)


# ── Main Sentiment Module ──────────────────────────────────────────────


class NewsSentiment:
    """
    News sentiment analysis with position-size penalty system.

    Flow:
      1. Fetch recent headlines for a symbol
      2. Score each headline (FinBERT or keyword fallback)
      3. Aggregate into a single sentiment score
      4. Convert to a position-size multiplier (0.5 – 1.0)
    """

    def __init__(
        self,
        project_root: Path,
        use_finbert: bool = False,
        cache_ttl_minutes: int = 60,
        penalty_floor: float = 0.5,
        negative_threshold: float = -0.3,
    ):
        """
        Args:
            project_root: Path to project root.
            use_finbert: If True, try to use FinBERT. Falls back to keywords.
            cache_ttl_minutes: How long to cache sentiment scores.
            penalty_floor: Minimum position-size multiplier (0.5 = halve at worst).
            negative_threshold: Aggregate score below this → penalty applied.
        """
        self.project_root = project_root
        self.cache_file = project_root / "data" / "snapshots" / "sentiment_cache.json"
        self.use_finbert = use_finbert
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)
        self.penalty_floor = penalty_floor
        self.negative_threshold = negative_threshold
        self._cache: Dict = {}
        self._load_cache()

    # ─── Cache ─────────────────────────────────────────────────────────

    def _load_cache(self) -> None:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r") as f:
                    self._cache = json.load(f)
            except Exception:
                self._cache = {}

    def _save_cache(self) -> None:
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.cache_file, "w") as f:
                json.dump(self._cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save sentiment cache: {e}")

    def _is_cached(self, symbol: str) -> bool:
        entry = self._cache.get(symbol)
        if not entry or "fetched_at" not in entry:
            return False
        fetched = datetime.fromisoformat(entry["fetched_at"])
        return (datetime.now() - fetched) < self.cache_ttl

    # ─── News Fetching ─────────────────────────────────────────────────

    def fetch_news(
        self, symbol: str, lookback_hours: int = 48
    ) -> List[Dict]:
        """
        Fetch recent news headlines for a symbol.

        Tries Alpaca News API first, then yfinance news as fallback.

        Returns:
            List of {"headline": str, "datetime": str, "source": str}
        """
        articles = self._fetch_alpaca_news(symbol, lookback_hours)
        if not articles:
            articles = self._fetch_yfinance_news(symbol)
        return articles

    def _fetch_alpaca_news(
        self, symbol: str, lookback_hours: int = 48
    ) -> List[Dict]:
        """Fetch from Alpaca News API."""
        try:
            from alpaca.data.requests import NewsRequest
            from alpaca.data.historical.news import NewsClient
            import os

            api_key = os.environ.get("ALPACA_API_KEY", "")
            secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
            if not api_key:
                return []

            client = NewsClient(api_key=api_key, secret_key=secret_key)
            start = datetime.now() - timedelta(hours=lookback_hours)

            request = NewsRequest(
                symbols=symbol,
                start=start.isoformat() + "Z",
                limit=20,
                sort="desc",
            )
            news = client.get_news(request)

            articles = []
            for article in news.news:
                articles.append({
                    "headline": article.headline,
                    "datetime": str(article.created_at),
                    "source": article.source,
                })
            return articles

        except Exception as e:
            logger.debug(f"Alpaca news fetch failed for {symbol}: {e}")
            return []

    def _fetch_yfinance_news(self, symbol: str) -> List[Dict]:
        """Fallback: fetch news from yfinance."""
        try:
            import yfinance as yf

            t = yf.Ticker(symbol)
            news = t.news or []
            articles = []
            for article in news[:20]:
                title = article.get("title", "")
                if title:
                    articles.append({
                        "headline": title,
                        "datetime": str(
                            datetime.fromtimestamp(article.get("providerPublishTime", 0))
                        ),
                        "source": article.get("publisher", "unknown"),
                    })
            return articles
        except Exception as e:
            logger.debug(f"yfinance news fetch failed for {symbol}: {e}")
            return []

    # ─── Scoring ───────────────────────────────────────────────────────

    def score_headlines(self, headlines: List[str]) -> float:
        """
        Aggregate sentiment score across multiple headlines.

        Args:
            headlines: List of headline strings.

        Returns:
            Aggregate score between -1.0 and 1.0.
        """
        if not headlines:
            return 0.0

        scorer = finbert_sentiment if self.use_finbert else keyword_sentiment
        scores = [scorer(h) for h in headlines]

        # Weighted average: more recent headlines have more weight
        # (assumes headlines are in reverse chronological order)
        weights = [1.0 / (i + 1) for i in range(len(scores))]
        total_weight = sum(weights)
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight

        return max(-1.0, min(1.0, weighted_score))

    def get_sentiment_score(self, symbol: str) -> float:
        """
        Get aggregate sentiment score for a symbol.

        Uses cache if available and fresh; otherwise fetches new data.

        Returns:
            Score between -1.0 (very bearish) and 1.0 (very bullish).
        """
        if self._is_cached(symbol):
            return self._cache[symbol].get("score", 0.0)

        articles = self.fetch_news(symbol)
        headlines = [a["headline"] for a in articles]
        score = self.score_headlines(headlines)

        # Cache the result
        self._cache[symbol] = {
            "score": score,
            "n_articles": len(articles),
            "fetched_at": datetime.now().isoformat(),
        }
        self._save_cache()
        return score

    def get_sentiment_multiplier(self, symbol: str) -> float:
        """
        Convert sentiment score into a position-size multiplier.

        Returns:
            Float between penalty_floor (0.5) and 1.0.
            - 1.0 = no penalty (neutral or positive sentiment)
            - 0.5 = maximum penalty (strongly negative sentiment)
        """
        score = self.get_sentiment_score(symbol)

        if score >= self.negative_threshold:
            return 1.0  # Neutral or positive — no penalty

        # Linear interpolation from threshold to -1.0
        # At threshold → 1.0, at -1.0 → penalty_floor
        range_width = abs(self.negative_threshold - (-1.0))
        distance = abs(score - self.negative_threshold)
        penalty_pct = min(distance / range_width, 1.0)

        multiplier = 1.0 - penalty_pct * (1.0 - self.penalty_floor)
        return max(self.penalty_floor, min(1.0, multiplier))

    # ─── Backtest Support ──────────────────────────────────────────────

    def build_backtest_sentiment(
        self,
        price_df: pd.DataFrame,
        lookback_days: int = 5,
        drop_threshold: float = -0.08,
    ) -> pd.DataFrame:
        """
        Build a synthetic sentiment multiplier DataFrame for backtesting.

        Since we don't have historical news headlines for the past,
        we use **price-action proxy**: if a stock dropped > drop_threshold
        in the last N days, it's a proxy for "negative news event" and
        gets a reduced sentiment multiplier.

        This is a conservative approximation — in practice, large drops
        are usually accompanied by negative news.

        Args:
            price_df: Historical prices (DatetimeIndex × symbols).
            lookback_days: Window for measuring recent drop.
            drop_threshold: If return over lookback < this, apply penalty.

        Returns:
            DataFrame (same shape as price_df) with multiplier values (0.5–1.0).
        """
        # Calculate rolling returns over lookback window
        returns = price_df.pct_change(periods=lookback_days)

        # Default multiplier = 1.0 (no penalty)
        multipliers = pd.DataFrame(1.0, index=price_df.index, columns=price_df.columns)

        # Where returns are significantly negative → apply penalty
        # Linear scale: at drop_threshold → slight penalty, at 2× → full penalty
        neg_mask = returns < drop_threshold
        if neg_mask.any().any():
            # How far below threshold (0 to 1 scale)
            excess_drop = (drop_threshold - returns).clip(lower=0)
            max_excess = abs(drop_threshold)  # At 2× the threshold, full penalty
            penalty_pct = (excess_drop / max_excess).clip(upper=1.0)
            multipliers = multipliers - penalty_pct * (1.0 - self.penalty_floor)
            multipliers = multipliers.clip(lower=self.penalty_floor, upper=1.0)

        return multipliers

    # ─── Batch Operations ──────────────────────────────────────────────

    def refresh_batch(
        self, symbols: List[str], delay: float = 0.5
    ) -> Dict[str, float]:
        """
        Fetch sentiment for multiple symbols.

        Returns:
            Dict mapping symbol → multiplier.
        """
        results = {}
        for sym in symbols:
            if not self._is_cached(sym):
                multiplier = self.get_sentiment_multiplier(sym)
                results[sym] = multiplier
                time.sleep(delay)
            else:
                results[sym] = self.get_sentiment_multiplier(sym)
        return results
