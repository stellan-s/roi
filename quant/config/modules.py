"""
Python module version of modules.yaml for easier importing
"""

modules = {
    "technical_indicators": {
        "enabled": True,
        "description": "Computes technical analysis indicators from price data",
        "sma_enabled": True,
        "sma_short": 20,
        "sma_long": 50,
        "momentum_enabled": True,
        "momentum_window": 21,
        "rsi_enabled": False,
        "rsi_window": 14,
        "volume_trend_enabled": True,
        "volume_window": 20,
        "volatility_enabled": True,
        "volatility_window": 20,
        "min_data_points": 50
    },
    "sentiment_analysis": {
        "enabled": True,
        "description": "Analyzes news sentiment for market signals",
        "sentiment_method": "naive",  # "naive" or "enhanced"
        "sentiment_window": 7,        # Days to aggregate sentiment
        "sentiment_threshold": 0.1,   # Minimum sentiment strength
        "confidence_threshold": 0.5
    },
    "regime_detection": {
        "enabled": True,
        "description": "Detects market regimes (Bull/Bear/Neutral)",
        "lookback_days": 60,
        "volatility_window": 20,
        "trend_window": 50,
        "vix_integration": True,
        "vol_low_threshold": 0.12,
        "vol_high_threshold": 0.30,
        "return_bull_threshold": 0.003,
        "return_bear_threshold": -0.003,
        "drawdown_bear_threshold": -0.15
    },
    "risk_management": {
        "enabled": True,
        "description": "Comprehensive risk analysis and management",
        "risk_free_rate": 0.02,
        "high_tail_risk_threshold": 0.05,
        "high_volatility_threshold": 0.30,
        "confidence_level": 0.95,
        "min_data_points": 30
    },
    "portfolio_management": {
        "enabled": True,
        "description": "Portfolio management with optimization and risk controls",
        "max_weight_per_stock": 0.15,
        "max_positions": 20,
        "min_position_size": 0.02,
        "max_single_regime_exposure": 0.80,
        "bear_market_allocation": 0.70,
        "transaction_cost_bps": 10,
        "target_total_allocation": 0.85
    }
}
