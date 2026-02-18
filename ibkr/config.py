"""
IBKR Module - Configuration
===========================
Configuration settings for Interactive Brokers API connection.
"""

# ===== CONNECTION SETTINGS =====
HOST = "172.22.112.1"       # Windows vEthernet IP (WSL → Windows)
PAPER_PORT = 7497           # Paper trading port (TWS default)
LIVE_PORT = 7496            # Live trading port (TWS default)
CLIENT_ID = 1               # Unique client ID for this application
CONNECTION_TIMEOUT = 30     # Timeout in seconds for API connection

# Use paper trading by default for safety
CURRENT_PORT = PAPER_PORT

# ===== ETF SYMBOL MAPPING (UCITS - EU Compliant) =====
# Using UCITS ETFs tradeable by EU retail investors (PRIIPs compliant)
# Trading on Euronext Amsterdam (AEB) - better liquidity for EU traders

# IBKR tickers (used for order execution)
ETF_MAPPING = {
    'SP500': 'SXR8',        # iShares Core S&P 500 UCITS (Xetra)
    'GOLD_OZ_USD': 'SGLD',  # iShares Physical Gold ETC (AEB)
    'SmallCAP': 'IUSN',     # iShares MSCI World Small Cap UCITS (AEB)
    'US_REIT_VNQ': 'IUSP',  # iShares US Property Yield UCITS (AEB)
    'TREASURY_10Y': 'IDTL', # iShares $ Treasury Bond 7-10yr UCITS (AEB)
    'OBLIGATION': 'LQDE',   # iShares $ Corp Bond UCITS (AEB)
    'NASDAQ_100': 'SXRV',   # iShares Nasdaq 100 UCITS (Xetra ticker on AEB)
    'COMMODITIES': 'EXXY',  # iShares Diversified Commodity Swap UCITS (AEB)
}

# Yahoo Finance tickers (used for price lookup - need exchange suffix)
YFINANCE_MAPPING = {
    'SP500': 'SXR8.DE',        # Xetra (Germany)
    'GOLD_OZ_USD': 'SGLD.L',   # London (Gold ETC)
    'SmallCAP': 'IUSN.AS',     # Amsterdam
    'US_REIT_VNQ': 'IUSP.L',   # London
    'TREASURY_10Y': 'IDTL.L',  # London
    'OBLIGATION': 'LQDE.L',    # London
    'NASDAQ_100': 'SXRV.DE',   # Germany (Xetra)
    'COMMODITIES': 'EXXY.DE',  # Germany
}


# ===== SAFETY LIMITS (Paper Trading) =====
MAX_ORDER_VALUE_USD = 500000    # Maximum value per single order (for 1M portfolio)
MAX_TOTAL_POSITION_USD = 1500000  # Maximum total portfolio value
MIN_ORDER_SIZE_USD = 50         # Minimum order to avoid dust

# ===== EXECUTION SETTINGS =====
DRY_RUN_DEFAULT = False         # Execute orders on paper account
ORDER_TYPE = "MKT"              # Market orders for simplicity
REBALANCE_THRESHOLD = 0.02      # 2% - Only rebalance if delta > threshold

# ===== SCHEDULING (Market Hours) =====
# US Market Hours (Eastern Time)
US_MARKET_OPEN_ET = "09:30"     # 9:30 AM ET = 15:30 CET
US_MARKET_CLOSE_ET = "16:00"    # 4:00 PM ET = 22:00 CET

# DAG Schedule: 16:00 CET (15:00 UTC) - 30 min after US market open
# Good liquidity, reasonable spreads
DAG_SCHEDULE_CRON = "0 15 * * 1-5"  # Mon-Fri at 15:00 UTC = 16:00 CET

# Execution timing options:
# - "immediate": Execute right after backtest (pre-market if before open)
# - "market_open": Wait for US market open (15:35 CET)
# - "manual": Don't auto-execute, wait for manual trigger
EXECUTION_TIMING = "market_open"

# Skip execution on weekends/holidays
SKIP_WEEKENDS = True
SKIP_US_HOLIDAYS = True

