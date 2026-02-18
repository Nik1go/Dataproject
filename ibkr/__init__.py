"""
IBKR Module
===========
Interactive Brokers integration for automated strategy execution.
"""

from .config import (
    HOST,
    PAPER_PORT,
    LIVE_PORT,
    CLIENT_ID,
    CURRENT_PORT,
    ETF_MAPPING,
    REBALANCE_THRESHOLD,
    DRY_RUN_DEFAULT,
)

from .connection import IBKRConnection
from .portfolio import PortfolioManager
from .orders import OrderManager
from .executor import execute_strategy

__all__ = [
    'HOST',
    'PAPER_PORT', 
    'LIVE_PORT',
    'CLIENT_ID',
    'CURRENT_PORT',
    'ETF_MAPPING',
    'REBALANCE_THRESHOLD',
    'DRY_RUN_DEFAULT',
    'IBKRConnection',
    'PortfolioManager',
    'OrderManager',
    'execute_strategy',
]

