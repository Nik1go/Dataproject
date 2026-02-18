"""
IBKR Order Manager
==================
Calculates rebalancing orders and executes them via IBKR.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import logging

from ib_insync import IB, Stock, MarketOrder, LimitOrder

from .connection import IBKRConnection
from .config import (
    ETF_MAPPING, HOST, CURRENT_PORT, CLIENT_ID, CONNECTION_TIMEOUT,
    REBALANCE_THRESHOLD, MAX_ORDER_VALUE_USD, MIN_ORDER_SIZE_USD,
    ORDER_TYPE
)

logger = logging.getLogger(__name__)


@dataclass
class RebalanceOrder:
    """Represents a rebalancing order to execute."""
    asset_name: str
    symbol: str
    action: str  # 'BUY' or 'SELL'
    shares: int
    estimated_value: float
    current_weight: float
    target_weight: float
    weight_delta: float


class OrderManager:
    """
    Manages order calculation and execution for portfolio rebalancing.
    
    The rebalancing logic:
    1. Compare current weights to target weights
    2. Only rebalance if delta > threshold (default 2%)
    3. Execute SELL orders first to free up cash
    4. Then execute BUY orders
    """
    
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
            self.ib.reqMarketDataType(3)
            logger.info(f"OrderManager connected to {self.host}:{self.port} (using delayed market data)")
            return True
        except Exception as e:
            logger.error(f"OrderManager connection failed: {e}")
            self.ib = None
            return False
    
    def disconnect(self):
        """Disconnect from IBKR."""
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()
            logger.info("OrderManager disconnected")
        self.ib = None
    
    def get_current_price(self, asset_name: str) -> Optional[float]:
        """
        Get current market price for an asset using IBKR delayed data.
        Uses IBKR API only to ensure ticker consistency with order execution.
        
        Args:
            asset_name: Internal asset name (e.g., 'SP500')
            
        Returns:
            Current price or None if unavailable.
        """
        ibkr_symbol = ETF_MAPPING.get(asset_name)
        
        if not ibkr_symbol:
            logger.error(f"No ETF mapping for {asset_name}")
            return None
        
        if not self.ib or not self.ib.isConnected():
            logger.error("Not connected to IBKR")
            return None
        
        try:
            # Create contract (same as used for orders)
            contract = Stock(ibkr_symbol, 'SMART', 'EUR')
            qualified = self.ib.qualifyContracts(contract)
            
            if not qualified or not contract.conId:
                logger.error(f"❌ Contract not found for {asset_name} ({ibkr_symbol})")
                return None
            
            logger.info(f"Contract found: {ibkr_symbol} conId={contract.conId}")
            
            # Request delayed market data
            ticker = self.ib.reqMktData(contract, '', False, False)
            self.ib.sleep(3)  # Wait for delayed data
            
            # Try market price first
            price = ticker.marketPrice()
            self.ib.cancelMktData(contract)
            
            if price and price > 0 and not (price != price):  # Check for NaN
                logger.info(f"{asset_name} ({ibkr_symbol}): IBKR price = €{price:.2f}")
                return price
            
            # Try close price as fallback
            if ticker.close and ticker.close > 0:
                logger.info(f"{asset_name} ({ibkr_symbol}): IBKR close = €{ticker.close:.2f}")
                return ticker.close
                
            logger.warning(f"No price data for {asset_name} ({ibkr_symbol})")
            return None
            
        except Exception as e:
            logger.error(f"IBKR price fetch failed for {ibkr_symbol}: {e}")
            return None
    
    def calculate_rebalance_orders(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        portfolio_value: float,
        threshold: float = REBALANCE_THRESHOLD
    ) -> List[RebalanceOrder]:
        """
        Calculate orders needed to rebalance portfolio.
        
        Args:
            current_weights: Current portfolio weights by asset
            target_weights: Target portfolio weights by asset
            portfolio_value: Total portfolio value in USD
            threshold: Minimum weight difference to trigger rebalance (default 2%)
            
        Returns:
            List of RebalanceOrder objects (SELL orders first, then BUY)
        """
        if not self.ib or not self.ib.isConnected():
            raise ConnectionError("Not connected to IBKR")
        
        orders = []
        
        for asset_name, target_weight in target_weights.items():
            current_weight = current_weights.get(asset_name, 0.0)
            weight_delta = target_weight - current_weight
            
            # Skip if delta is below threshold
            if abs(weight_delta) < threshold:
                logger.debug(f"{asset_name}: delta {weight_delta:.2%} < threshold {threshold:.2%}, skipping")
                continue
            
            # Calculate order value
            order_value = abs(weight_delta) * portfolio_value
            
            # Skip dust orders
            if order_value < MIN_ORDER_SIZE_USD:
                logger.debug(f"{asset_name}: order value ${order_value:.2f} < min ${MIN_ORDER_SIZE_USD}, skipping")
                continue
            
            # Cap order value for safety
            if order_value > MAX_ORDER_VALUE_USD:
                logger.warning(f"{asset_name}: order value ${order_value:.2f} exceeds max ${MAX_ORDER_VALUE_USD}, capping")
                order_value = MAX_ORDER_VALUE_USD
            
            # Get symbol and current price
            symbol = ETF_MAPPING.get(asset_name)
            if not symbol:
                logger.error(f"No symbol mapping for {asset_name}")
                continue
            
            price = self.get_current_price(asset_name)  # Now uses asset_name
            if not price:
                logger.error(f"Could not get price for {asset_name}, skipping")
                continue
            
            # Calculate shares
            shares = int(order_value / price)
            if shares < 1:
                logger.debug(f"{asset_name}: calculated shares < 1, skipping")
                continue
            
            action = 'BUY' if weight_delta > 0 else 'SELL'
            
            order = RebalanceOrder(
                asset_name=asset_name,
                symbol=symbol,
                action=action,
                shares=shares,
                estimated_value=shares * price,
                current_weight=current_weight,
                target_weight=target_weight,
                weight_delta=weight_delta
            )
            orders.append(order)
            
            logger.info(
                f"Order: {action} {shares} {symbol} "
                f"(~${order.estimated_value:.2f}) "
                f"[{current_weight:.1%} → {target_weight:.1%}]"
            )
        
        # Sort: SELL orders first (to free up cash), then BUY
        orders.sort(key=lambda x: (0 if x.action == 'SELL' else 1, -x.estimated_value))
        
        return orders
    
    def execute_orders(
        self,
        orders: List[RebalanceOrder],
        dry_run: bool = False
    ) -> Dict[str, any]:
        """
        Execute a list of orders.
        
        Args:
            orders: List of RebalanceOrder objects
            dry_run: If True, only log orders without executing
            
        Returns:
            Execution summary dict
        """
        if not self.ib or not self.ib.isConnected():
            raise ConnectionError("Not connected to IBKR")
        
        results = {
            'executed': [],
            'failed': [],
            'skipped': [],
            'dry_run': dry_run
        }
        
        for order in orders:
            try:
                if dry_run:
                    logger.info(f"[DRY RUN] Would {order.action} {order.shares} {order.symbol}")
                    results['skipped'].append({
                        'symbol': order.symbol,
                        'action': order.action,
                        'shares': order.shares,
                        'reason': 'dry_run'
                    })
                    continue
                
                # Create contract (SMART routing, EUR for European UCITS ETFs)
                contract = Stock(order.symbol, 'SMART', 'EUR')
                qualified = self.ib.qualifyContracts(contract)
                
                # Check if contract was found
                if not qualified or not contract.conId:
                    logger.error(f"❌ Contract not found for {order.symbol} - skipping order")
                    results['failed'].append({
                        'symbol': order.symbol,
                        'action': order.action,
                        'shares': order.shares,
                        'reason': 'contract_not_found'
                    })
                    continue
                
                logger.info(f"Contract qualified: {order.symbol} conId={contract.conId}")
                
                # Create order
                if ORDER_TYPE == 'MKT':
                    ib_order = MarketOrder(order.action, order.shares)
                else:
                    # For limit orders, use current price
                    price = self.get_current_price(order.asset_name)
                    ib_order = LimitOrder(order.action, order.shares, price)
                
                # Place order
                trade = self.ib.placeOrder(contract, ib_order)
                self.ib.sleep(1)  # Wait for order to be processed
                
                logger.info(
                    f"✅ Executed: {order.action} {order.shares} {order.symbol} "
                    f"- Status: {trade.orderStatus.status}"
                )
                
                results['executed'].append({
                    'symbol': order.symbol,
                    'action': order.action,
                    'shares': order.shares,
                    'status': trade.orderStatus.status,
                    'order_id': trade.order.orderId
                })
                
            except Exception as e:
                logger.error(f"❌ Failed to execute {order.action} {order.shares} {order.symbol}: {e}")
                results['failed'].append({
                    'symbol': order.symbol,
                    'action': order.action,
                    'shares': order.shares,
                    'error': str(e)
                })
        
        return results


def test_order_calculation():
    """Test order calculation without executing."""
    # Mock data for testing
    current_weights = {
        'SP500': 0.35,
        'GOLD_OZ_USD': 0.25,
        'TREASURY_10Y': 0.40,
        'SmallCAP': 0.0,
        'US_REIT_VNQ': 0.0,
        'OBLIGATION': 0.0,
        'NASDAQ_100': 0.0,
        'COMMODITIES': 0.0
    }
    
    # Target: Q1 allocation
    target_weights = {
        'SP500': 0.30,
        'NASDAQ_100': 0.40,
        'SmallCAP': 0.30,
        'GOLD_OZ_USD': 0.0,
        'TREASURY_10Y': 0.0,
        'US_REIT_VNQ': 0.0,
        'OBLIGATION': 0.0,
        'COMMODITIES': 0.0
    }
    
    portfolio_value = 10000.0
    
    om = OrderManager()
    if om.connect():
        try:
            orders = om.calculate_rebalance_orders(
                current_weights, target_weights, portfolio_value
            )
            print(f"\n📋 Orders to execute ({len(orders)} total):")
            for o in orders:
                print(f"  {o.action} {o.shares} {o.symbol} (~${o.estimated_value:.2f})")
        finally:
            om.disconnect()
    else:
        print("❌ Failed to connect to IBKR")


if __name__ == "__main__":
    test_order_calculation()
