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
        "enabled": False,  # Not implemented yet
        "description": "Analyzes news sentiment for market signals"
    },
    "regime_detection": {
        "enabled": False,  # Not implemented yet
        "description": "Detects market regimes (Bull/Bear/Neutral)"
    },
    "risk_management": {
        "enabled": False,  # Not implemented yet
        "description": "Computes risk metrics and position sizing"
    },
    "portfolio_management": {
        "enabled": False,  # Not implemented yet
        "description": "Manages portfolio allocation and rebalancing"
    }
}