"""
Sentiment Analysis Module

Analyzes news sentiment for market signals using both naive keyword matching
and advanced sentiment analysis techniques.
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, Any, List
from datetime import datetime, timedelta

from .base import BaseModule, ModuleOutput, ModuleContract

class SentimentAnalysisModule(BaseModule):
    """Module for analyzing news sentiment and generating market signals"""

    def define_contract(self) -> ModuleContract:
        return ModuleContract(
            name="sentiment_analysis",
            version="1.0.0",
            description="Analyzes news sentiment for market signals",
            input_schema={
                "news": "pd.DataFrame[published, title, summary]",
                "tickers": "List[str]"
            },
            output_schema={
                "sentiment_signals": "Dict[ticker, float]",
                "sentiment_features": "pd.DataFrame",
                "sentiment_summary": "Dict[str, Any]"
            },
            performance_sla={
                "max_latency_ms": 500.0,
                "min_confidence": 0.6
            },
            dependencies=[]
        )

    def process(self, inputs: Dict[str, Any]) -> ModuleOutput:
        """Analyze sentiment from news data"""
        news_df = inputs['news']
        tickers = inputs['tickers']

        # Validate inputs
        if news_df.empty:
            return ModuleOutput(
                data={
                    "sentiment_signals": {},
                    "sentiment_features": pd.DataFrame(),
                    "sentiment_summary": {"reason": "no_news_data"}
                },
                metadata={"reason": "no_news_data"},
                confidence=0.0
            )

        if not tickers:
            return ModuleOutput(
                data={
                    "sentiment_signals": {},
                    "sentiment_features": pd.DataFrame(),
                    "sentiment_summary": {"reason": "no_tickers"}
                },
                metadata={"reason": "no_tickers"},
                confidence=0.0
            )

        # Choose sentiment method based on config
        sentiment_method = self.config.get('sentiment_method', 'naive')

        if sentiment_method == 'naive':
            sentiment_data = self._naive_sentiment_analysis(news_df, tickers)
        elif sentiment_method == 'enhanced':
            sentiment_data = self._enhanced_sentiment_analysis(news_df, tickers)
        else:
            raise ValueError(f"Unknown sentiment method: {sentiment_method}")

        # Generate sentiment signals
        sentiment_signals = self._generate_sentiment_signals(sentiment_data)

        # Create features dataframe
        sentiment_features = self._create_sentiment_features(sentiment_data)

        # Calculate summary statistics
        sentiment_summary = self._calculate_sentiment_summary(sentiment_data, news_df)

        # Calculate confidence based on data quality
        confidence = self._calculate_confidence(sentiment_data, news_df, tickers)

        metadata = {
            "sentiment_method": sentiment_method,
            "news_articles_processed": len(news_df),
            "tickers_with_sentiment": len(sentiment_signals),
            "total_tickers": len(tickers),
            "news_date_range": self._get_news_date_range(news_df)
        }

        return ModuleOutput(
            data={
                "sentiment_signals": sentiment_signals,
                "sentiment_features": sentiment_features,
                "sentiment_summary": sentiment_summary
            },
            metadata=metadata,
            confidence=confidence
        )

    def test_module(self) -> Dict[str, Any]:
        """Test the sentiment analysis module with synthetic data"""
        # Generate test data
        test_news, test_tickers = self._generate_test_data()

        # Test processing
        result = self.process({'news': test_news, 'tickers': test_tickers})

        # Validate outputs
        signals = result.data['sentiment_signals']
        features_df = result.data['sentiment_features']
        summary = result.data['sentiment_summary']

        tests_passed = 0
        total_tests = 6

        # Test 1: Signals generated
        if signals and len(signals) > 0:
            tests_passed += 1

        # Test 2: All signal values are valid (between -1 and 1)
        all_valid = True
        for signal_value in signals.values():
            if not isinstance(signal_value, (int, float)) or not -1 <= signal_value <= 1:
                all_valid = False
                break
        if all_valid:
            tests_passed += 1

        # Test 3: Features dataframe has correct structure
        if not features_df.empty and 'ticker' in features_df.columns:
            tests_passed += 1

        # Test 4: Summary contains expected keys
        expected_keys = ['total_articles', 'avg_sentiment', 'sentiment_distribution']
        if all(key in summary for key in expected_keys):
            tests_passed += 1

        # Test 5: Positive news creates positive sentiment
        if any(s > 0 for s in signals.values()):
            tests_passed += 1

        # Test 6: Confidence is reasonable
        if 0.3 <= result.confidence <= 1.0:
            tests_passed += 1

        return {
            "status": "PASS" if tests_passed >= 4 else "FAIL",
            "tests_passed": tests_passed,
            "total_tests": total_tests,
            "signals_generated": len(signals),
            "confidence": result.confidence
        }

    def _generate_test_inputs(self) -> Dict[str, Any]:
        """Generate synthetic test data for benchmarking"""
        test_news, test_tickers = self._generate_test_data()
        return {"news": test_news, "tickers": test_tickers}

    def _generate_test_data(self) -> tuple:
        """Generate synthetic news data for testing"""
        # Test tickers
        tickers = ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'META']

        # Generate test news articles
        positive_templates = [
            "{company} reports strong quarterly earnings",
            "{company} wins major contract deal",
            "{company} announces record growth",
            "{company} raises guidance for next quarter"
        ]

        negative_templates = [
            "{company} cuts workforce amid slowdown",
            "{company} faces investigation over practices",
            "{company} reports profit warning",
            "{company} sees weak demand"
        ]

        neutral_templates = [
            "{company} announces new product launch",
            "{company} schedules earnings call",
            "{company} appoints new board member"
        ]

        news_data = []
        base_date = datetime.now() - timedelta(days=7)

        for i in range(50):  # Generate 50 articles
            # Random ticker
            ticker = np.random.choice(tickers)
            company = ticker  # Simplified

            # Random sentiment
            sentiment_type = np.random.choice(['positive', 'negative', 'neutral'], p=[0.4, 0.3, 0.3])

            if sentiment_type == 'positive':
                title = np.random.choice(positive_templates).format(company=company)
            elif sentiment_type == 'negative':
                title = np.random.choice(negative_templates).format(company=company)
            else:
                title = np.random.choice(neutral_templates).format(company=company)

            # Random date within last week
            article_date = base_date + timedelta(days=np.random.randint(0, 7))

            news_data.append({
                'published': article_date,
                'title': title,
                'summary': f"Additional details about {company} and market conditions."
            })

        return pd.DataFrame(news_data), tickers

    def _naive_sentiment_analysis(self, news_df: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
        """Original naive sentiment analysis implementation"""
        # Compile regex patterns
        POS = re.compile(r"\b(strong|record|raises|wins|contract|growth|beats|exceeds|positive|gain|up|bullish|optimistic|success)\b", re.I)
        NEG = re.compile(r"\b(profit warning|cuts|loss|weak|layoffs|investigation|down|falls|disappoints|bearish|pessimistic|decline|drop)\b", re.I)

        out = []
        for _, row in news_df.iterrows():
            text = f"{row.get('title', '')} {row.get('summary', '')}"
            score = 0

            # Count positive and negative words
            pos_matches = len(POS.findall(text))
            neg_matches = len(NEG.findall(text))

            score = pos_matches - neg_matches

            # Normalize score to [-1, 1] range
            if score != 0:
                score = np.tanh(score / 3)  # Dampen extreme scores

            # Find ticker mentions
            hits = []
            for ticker in tickers:
                ticker_base = ticker.split(".")[0].replace("-", "")
                text_clean = text.replace("-", "").upper()
                if ticker_base in text_clean:
                    hits.append(ticker)

            if not hits:
                continue

            for ticker in hits:
                out.append({
                    "ticker": ticker,
                    "published": row.get("published"),
                    "title": row.get("title", ""),
                    "sent_score": score,
                    "pos_words": pos_matches,
                    "neg_words": neg_matches
                })

        if not out:
            return pd.DataFrame(columns=["ticker", "date", "sent_score", "article_count"])

        df = pd.DataFrame(out)
        df["date"] = pd.to_datetime(df["published"]).dt.date

        # Aggregate by date and ticker
        sentiment_agg = df.groupby(["date", "ticker"]).agg({
            "sent_score": "mean",
            "pos_words": "sum",
            "neg_words": "sum",
            "title": "count"
        }).reset_index()

        sentiment_agg.rename(columns={"title": "article_count"}, inplace=True)

        return sentiment_agg

    def _enhanced_sentiment_analysis(self, news_df: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
        """Enhanced sentiment analysis with more sophisticated techniques"""
        # This could integrate with actual NLP libraries or LLM APIs
        # For now, implement an improved version of naive analysis

        # Enhanced word patterns
        STRONG_POS = re.compile(r"\b(excellent|outstanding|exceptional|breakthrough|surge|soars|skyrockets)\b", re.I)
        POS = re.compile(r"\b(strong|good|positive|growth|increase|rise|up|gain|success|win|beat)\b", re.I)
        NEG = re.compile(r"\b(weak|poor|negative|decline|decrease|fall|down|loss|fail|miss)\b", re.I)
        STRONG_NEG = re.compile(r"\b(terrible|awful|catastrophic|plummet|crash|collapse|disaster)\b", re.I)

        # Context patterns
        EARNINGS_CONTEXT = re.compile(r"\b(earnings|revenue|profit|eps|guidance)\b", re.I)
        MARKET_CONTEXT = re.compile(r"\b(market|stock|share|trading|investor)\b", re.I)

        out = []
        for _, row in news_df.iterrows():
            text = f"{row.get('title', '')} {row.get('summary', '')}"

            # Calculate weighted sentiment score
            score = 0
            score += len(STRONG_POS.findall(text)) * 2    # Strong positive: +2
            score += len(POS.findall(text)) * 1           # Positive: +1
            score -= len(NEG.findall(text)) * 1           # Negative: -1
            score -= len(STRONG_NEG.findall(text)) * 2    # Strong negative: -2

            # Context weighting
            context_weight = 1.0
            if EARNINGS_CONTEXT.search(text):
                context_weight *= 1.5  # Earnings news is more important
            if MARKET_CONTEXT.search(text):
                context_weight *= 1.2  # Market-related news gets boost

            score *= context_weight

            # Normalize to [-1, 1]
            score = np.tanh(score / 5)

            # Find ticker mentions (improved matching)
            hits = self._find_ticker_mentions(text, tickers)

            if not hits:
                continue

            for ticker in hits:
                out.append({
                    "ticker": ticker,
                    "published": row.get("published"),
                    "title": row.get("title", ""),
                    "sent_score": score,
                    "context_weight": context_weight
                })

        if not out:
            return pd.DataFrame(columns=["ticker", "date", "sent_score", "article_count"])

        df = pd.DataFrame(out)
        df["date"] = pd.to_datetime(df["published"]).dt.date

        # Weighted aggregation by date and ticker
        sentiment_agg = df.groupby(["date", "ticker"]).agg({
            "sent_score": lambda x: np.average(x, weights=df.loc[x.index, "context_weight"]),
            "context_weight": "mean",
            "title": "count"
        }).reset_index()

        sentiment_agg.rename(columns={"title": "article_count"}, inplace=True)

        return sentiment_agg

    def _find_ticker_mentions(self, text: str, tickers: List[str]) -> List[str]:
        """Improved ticker mention detection"""
        hits = []
        text_upper = text.upper()

        for ticker in tickers:
            # Try multiple matching strategies
            ticker_base = ticker.split(".")[0].replace("-", "")

            # Exact match
            if ticker_base in text_upper:
                hits.append(ticker)
                continue

            # Fuzzy matching for common variations
            # This could be expanded with more sophisticated NER
            variations = [
                ticker_base.replace("_", ""),
                ticker_base.replace("B", ""),  # Handle Swedish B-shares
                ticker_base.replace("A", "")   # Handle Swedish A-shares
            ]

            for variation in variations:
                if len(variation) >= 3 and variation in text_upper:
                    hits.append(ticker)
                    break

        return list(set(hits))  # Remove duplicates

    def _generate_sentiment_signals(self, sentiment_data: pd.DataFrame) -> Dict[str, float]:
        """Generate sentiment signals from processed data"""
        if sentiment_data.empty:
            return {}

        signals = {}
        sentiment_window = self.config.get('sentiment_window', 7)  # Days

        # Get recent sentiment for each ticker
        cutoff_date = datetime.now().date() - timedelta(days=sentiment_window)

        for ticker in sentiment_data['ticker'].unique():
            ticker_data = sentiment_data[sentiment_data['ticker'] == ticker]

            # Filter to recent data
            recent_data = ticker_data[ticker_data['date'] >= cutoff_date]

            if recent_data.empty:
                continue

            # Calculate weighted average sentiment
            if 'article_count' in recent_data.columns:
                # Weight by article count
                weights = recent_data['article_count']
                avg_sentiment = np.average(recent_data['sent_score'], weights=weights)
            else:
                avg_sentiment = recent_data['sent_score'].mean()

            # Apply threshold
            sentiment_threshold = self.config.get('sentiment_threshold', 0.1)
            if abs(avg_sentiment) >= sentiment_threshold:
                signals[ticker] = float(avg_sentiment)

        return signals

    def _create_sentiment_features(self, sentiment_data: pd.DataFrame) -> pd.DataFrame:
        """Create sentiment features dataframe"""
        if sentiment_data.empty:
            return pd.DataFrame()

        # Get latest sentiment for each ticker
        latest_data = sentiment_data.sort_values('date').groupby('ticker').tail(1)

        features = []
        for _, row in latest_data.iterrows():
            feature_row = {
                'ticker': row['ticker'],
                'date': row['date'],
                'sentiment_score': row['sent_score'],
                'article_count': row.get('article_count', 1)
            }

            # Add additional features if available
            if 'pos_words' in row:
                feature_row['positive_words'] = row['pos_words']
                feature_row['negative_words'] = row['neg_words']

            if 'context_weight' in row:
                feature_row['context_weight'] = row['context_weight']

            features.append(feature_row)

        return pd.DataFrame(features)

    def _calculate_sentiment_summary(self, sentiment_data: pd.DataFrame, news_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate summary statistics for sentiment analysis"""
        if sentiment_data.empty:
            return {
                "total_articles": len(news_df),
                "avg_sentiment": 0.0,
                "sentiment_distribution": {"positive": 0, "negative": 0, "neutral": 0}
            }

        summary = {
            "total_articles": len(news_df),
            "articles_with_sentiment": len(sentiment_data),
            "avg_sentiment": float(sentiment_data['sent_score'].mean()),
            "sentiment_std": float(sentiment_data['sent_score'].std()),
            "tickers_covered": int(sentiment_data['ticker'].nunique())
        }

        # Sentiment distribution
        positive_count = (sentiment_data['sent_score'] > 0.1).sum()
        negative_count = (sentiment_data['sent_score'] < -0.1).sum()
        neutral_count = len(sentiment_data) - positive_count - negative_count

        summary["sentiment_distribution"] = {
            "positive": int(positive_count),
            "negative": int(negative_count),
            "neutral": int(neutral_count)
        }

        # Date range analysis
        if 'date' in sentiment_data.columns:
            summary["date_range"] = {
                "start": str(sentiment_data['date'].min()),
                "end": str(sentiment_data['date'].max()),
                "days_covered": int((sentiment_data['date'].max() - sentiment_data['date'].min()).days) + 1
            }

        return summary

    def _calculate_confidence(self, sentiment_data: pd.DataFrame, news_df: pd.DataFrame, tickers: List[str]) -> float:
        """Calculate confidence score based on data quality"""
        if news_df.empty or not tickers:
            return 0.0

        # Base confidence from data coverage
        articles_with_sentiment = len(sentiment_data)
        total_articles = len(news_df)
        coverage_conf = min(1.0, articles_with_sentiment / max(1, total_articles))

        # Ticker coverage confidence
        tickers_with_sentiment = sentiment_data['ticker'].nunique() if not sentiment_data.empty else 0
        ticker_coverage_conf = tickers_with_sentiment / len(tickers)

        # Recency confidence
        if not sentiment_data.empty and 'date' in sentiment_data.columns:
            latest_date = sentiment_data['date'].max()
            days_old = (datetime.now().date() - latest_date).days
            recency_conf = max(0.1, 1.0 - (days_old / 30))  # Decay over 30 days
        else:
            recency_conf = 0.1

        # Article count confidence
        if not sentiment_data.empty and 'article_count' in sentiment_data.columns:
            avg_articles = sentiment_data['article_count'].mean()
            article_conf = min(1.0, avg_articles / 5)  # Ideal: 5+ articles per ticker
        else:
            article_conf = 0.5

        # Combined confidence
        confidence = (
            coverage_conf * 0.3 +
            ticker_coverage_conf * 0.3 +
            recency_conf * 0.2 +
            article_conf * 0.2
        )

        return min(1.0, confidence)

    def _get_news_date_range(self, news_df: pd.DataFrame) -> Dict[str, str]:
        """Get date range of news articles"""
        if news_df.empty or 'published' not in news_df.columns:
            return {"start": "N/A", "end": "N/A"}

        dates = pd.to_datetime(news_df['published'])
        return {
            "start": str(dates.min().date()),
            "end": str(dates.max().date())
        }