"""
IBKR Portfolio Manager
======================
Reads current positions and calculates portfolio weights from IBKR.
"""

from typing import Dict, Optional
import logging

from ib_insync import IB, Stock

from .connection import IBKRConnection
from .config import ETF_MAPPING, HOST, CURRENT_PORT, CLIENT_ID, CONNECTION_TIMEOUT

logger = logging.getLogger(__name__)


class PortfolioManager:
    """
    Manages portfolio reading from IBKR.
    
    Usage:
        pm = PortfolioManager()
        if pm.connect():
            weights = pm.get_current_weights()
            value = pm.get_portfolio_value()
            pm.disconnect()
    """
    
    # Reverse mapping: IBKR symbol -> internal name
    SYMBOL_TO_ASSET = {v: k for k, v in ETF_MAPPING.items()}
    
    def __init__(self, host: str = HOST, port: int = CURRENT_PORT, 
                 client_id: int = CLIENT_ID, timeout: int = CONNECTION_TIMEOUT):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.timeout = timeout
        self.ib: Optional[IB] = None
        
    def connect(self) -> bool:
        """Connect to IBKR."""
        try:
            self.ib = IB()
            self.ib.connect(self.host, self.port, clientId=self.client_id, timeout=self.timeout)
            logger.info(f"PortfolioManager connected to {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"PortfolioManager connection failed: {e}")
            self.ib = None
            return False
    
    def disconnect(self):
        """Disconnect from IBKR."""
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()
            logger.info("PortfolioManager disconnected")
        self.ib = None
    
    def get_positions(self) -> Dict[str, Dict]:
        """
        Get current positions from IBKR.
        
        Returns:
            Dict mapping internal asset names to position info:
            {
                'SP500': {'symbol': 'SPY', 'shares': 10, 'avg_cost': 450.0, 'market_value': 4600.0},
                ...
            }
        """
        if not self.ib or not self.ib.isConnected():
            raise ConnectionError("Not connected to IBKR")
        
        positions = {}
        
        # Get portfolio items (includes market value)
        portfolio_items = self.ib.portfolio()
        
        for item in portfolio_items:
            symbol = item.contract.symbol
            
            # Check if this is one of our tracked assets
            if symbol in self.SYMBOL_TO_ASSET:
                asset_name = self.SYMBOL_TO_ASSET[symbol]
                positions[asset_name] = {
                    'symbol': symbol,
                    'shares': item.position,
                    'avg_cost': item.averageCost,
                    'market_value': item.marketValue,
                    'unrealized_pnl': item.unrealizedPNL
                }
                logger.debug(f"Position: {asset_name} ({symbol}): {item.position} shares, ${item.marketValue:.2f}")
        
        return positions
    
    def get_portfolio_value(self) -> float:
        """
        Get total portfolio value (cash + positions).
        
        Returns:
            Total portfolio value in account base currency.
        """
        if not self.ib or not self.ib.isConnected():
            raise ConnectionError("Not connected to IBKR")
        
        # Request account summary
        account_values = self.ib.accountSummary()
        
        # Try to find NetLiquidation in various currencies (EUR first, then USD, then BASE)
        for currency in ['EUR', 'USD', 'BASE']:
            for av in account_values:
                if av.tag == 'NetLiquidation' and av.currency == currency:
                    value = float(av.value)
                    if value > 0:
                        logger.info(f"Portfolio value: {value:.2f} {currency}")
                        return value
        
        # Fallback: sum positions + cash
        positions = self.get_positions()
        total_positions = sum(p['market_value'] for p in positions.values())
        
        # Get cash balance (try EUR first, then USD)
        cash = 0.0
        for currency in ['EUR', 'USD']:
            for av in account_values:
                if av.tag == 'TotalCashValue' and av.currency == currency:
                    cash = float(av.value)
                    if cash > 0:
                        break
            if cash > 0:
                break
        
        return total_positions + cash
    
    def get_current_weights(self) -> Dict[str, float]:
        """
        Calculate current portfolio weights.
        
        Returns:
            Dict mapping asset names to weights (0.0 to 1.0).
            Assets not held have weight 0.0.
        """
        if not self.ib or not self.ib.isConnected():
            raise ConnectionError("Not connected to IBKR")
        
        positions = self.get_positions()
        total_value = self.get_portfolio_value()
        
        if total_value <= 0:
            logger.warning("Portfolio value is zero or negative")
            return {asset: 0.0 for asset in ETF_MAPPING.keys()}
        
        weights = {}
        
        # Calculate weight for each asset
        for asset_name in ETF_MAPPING.keys():
            if asset_name in positions:
                market_value = positions[asset_name]['market_value']
                weights[asset_name] = market_value / total_value
            else:
                weights[asset_name] = 0.0
        
        logger.info(f"Current weights: {weights}")
        return weights
    
    def get_cash_balance(self) -> float:
        """Get available cash balance in account base currency."""
        if not self.ib or not self.ib.isConnected():
            raise ConnectionError("Not connected to IBKR")
        
        account_values = self.ib.accountSummary()
        
        # Try EUR first (for European accounts), then USD
        for currency in ['EUR', 'USD']:
            for av in account_values:
                if av.tag == 'TotalCashValue' and av.currency == currency:
                    value = float(av.value)
                    if value > 0:
                        return value
        
        return 0.0



def test_portfolio():
    """Quick test to read portfolio from IBKR."""
    pm = PortfolioManager()
    
    if pm.connect():
        try:
            print(f"📊 Portfolio Value: ${pm.get_portfolio_value():,.2f}")
            print(f"💵 Cash Balance: ${pm.get_cash_balance():,.2f}")
            print(f"📈 Positions: {pm.get_positions()}")
            print(f"⚖️ Weights: {pm.get_current_weights()}")
        finally:
            pm.disconnect()
    else:
        print("❌ Failed to connect to IBKR")


if __name__ == "__main__":
    test_portfolio()
