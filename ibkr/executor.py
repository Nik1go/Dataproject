"""
IBKR Strategy Executor
======================
Main orchestrator that reads backtest results and executes trades via IBKR.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd

from .portfolio import PortfolioManager
from .orders import OrderManager
from .config import (
    ETF_MAPPING, REBALANCE_THRESHOLD, HOST, CURRENT_PORT, CLIENT_ID
)

# Strategy allocations (same as backtest_strategy.py)
ALLOCATIONS = {
    # Q1 (Growth): Risk-on
    1: {
        'NASDAQ_100': 0.4,
        'SmallCAP': 0.3,
        'SP500': 0.3,
        'US_REIT_VNQ': 0.0,
        'GOLD_OZ_USD': 0.0,
        'TREASURY_10Y': 0.0,
        'OBLIGATION': 0.0,
        'COMMODITIES': 0.0
    },
    # Q2 (Inflation): Quality & Real Assets
    2: {
        'SP500': 0.4,
        'GOLD_OZ_USD': 0.3,
        'COMMODITIES': 0.2,
        'NASDAQ_100': 0.1,
        'SmallCAP': 0.0,
        'TREASURY_10Y': 0.0,
        'US_REIT_VNQ': 0.0,
        'OBLIGATION': 0.0
    },
    # Q3 (Stagflation): Full Defensive
    3: {
        'GOLD_OZ_USD': 0.6,
        'COMMODITIES': 0.2,
        'TREASURY_10Y': 0.2,
        'NASDAQ_100': 0.0,
        'SP500': 0.0,
        'SmallCAP': 0.0,
        'US_REIT_VNQ': 0.0,
        'OBLIGATION': 0.0
    },
    # Q4 (Deflation/Crash): Bunker Mode
    4: {
        'TREASURY_10Y': 0.6,
        'GOLD_OZ_USD': 0.4,
        'SP500': 0.0,
        'NASDAQ_100': 0.0,
        'SmallCAP': 0.0,
        'US_REIT_VNQ': 0.0,
        'OBLIGATION': 0.0,
        'COMMODITIES': 0.0
    },
}

logger = logging.getLogger(__name__)


def get_current_quadrant(backtest_output_dir: str) -> int:
    """
    Read the current (latest) quadrant from backtest results.
    
    Args:
        backtest_output_dir: Path to backtest results directory
        
    Returns:
        Current quadrant (1-4)
    """
    timeseries_path = os.path.join(backtest_output_dir, "backtest_timeseries.csv")
    
    if not os.path.exists(timeseries_path):
        raise FileNotFoundError(f"Backtest timeseries not found: {timeseries_path}")
    
    df = pd.read_csv(timeseries_path, parse_dates=['date'])
    df = df.sort_values('date')
    
    # Get the latest quadrant
    latest_quadrant = int(df['smooth_quadrant'].iloc[-1])
    latest_date = df['date'].iloc[-1]
    
    logger.info(f"Current quadrant: Q{latest_quadrant} (as of {latest_date.date()})")
    
    return latest_quadrant


def get_target_weights(quadrant: int) -> Dict[str, float]:
    """
    Get target portfolio weights for a given quadrant.
    
    Args:
        quadrant: Economic quadrant (1-4)
        
    Returns:
        Dict of asset weights
    """
    if quadrant not in ALLOCATIONS:
        raise ValueError(f"Invalid quadrant: {quadrant}. Must be 1-4.")
    
    return ALLOCATIONS[quadrant]


def execute_strategy(
    backtest_output_dir: str,
    dry_run: bool = False,
    rebalance_threshold: float = REBALANCE_THRESHOLD,
    execution_log_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main execution function: reads backtest results and executes trades.
    
    This function:
    1. Reads the current quadrant from backtest results
    2. Gets target allocation for that quadrant
    3. Reads current positions from IBKR
    4. Calculates rebalancing orders (if delta > threshold)
    5. Executes orders
    
    Args:
        backtest_output_dir: Path to backtest results (contains backtest_timeseries.csv)
        dry_run: If True, only log orders without executing
        rebalance_threshold: Minimum weight difference to trigger rebalance
        execution_log_dir: Directory to save execution logs (default: data/US/execution_logs)
        
    Returns:
        Execution summary dict
    """
    execution_start = datetime.now()
    
    result = {
        'timestamp': execution_start.isoformat(),
        'dry_run': dry_run,
        'success': False,
        'error': None,
        'quadrant': None,
        'target_weights': None,
        'current_weights': None,
        'portfolio_value': None,
        'orders': [],
        'execution_result': None
    }
    
    # Set up logging directory
    if execution_log_dir is None:
        base_dir = os.path.dirname(backtest_output_dir)
        execution_log_dir = os.path.join(base_dir, "execution_logs")
    os.makedirs(execution_log_dir, exist_ok=True)
    
    try:
        # 1. Get current quadrant from backtest
        quadrant = get_current_quadrant(backtest_output_dir)
        result['quadrant'] = quadrant
        
        # 2. Get target weights for this quadrant
        target_weights = get_target_weights(quadrant)
        result['target_weights'] = target_weights
        logger.info(f"Target weights for Q{quadrant}: {target_weights}")
        
        # 3. Connect to IBKR and get current positions
        pm = PortfolioManager()
        om = OrderManager()
        
        if not pm.connect():
            raise ConnectionError("Failed to connect PortfolioManager to IBKR")
        
        try:
            current_weights = pm.get_current_weights()
            portfolio_value = pm.get_portfolio_value()
            result['current_weights'] = current_weights
            result['portfolio_value'] = portfolio_value
            
            logger.info(f"Portfolio value: ${portfolio_value:,.2f}")
            logger.info(f"Current weights: {current_weights}")
        finally:
            pm.disconnect()
        
        # 4. Calculate rebalancing orders
        if not om.connect():
            raise ConnectionError("Failed to connect OrderManager to IBKR")
        
        try:
            orders = om.calculate_rebalance_orders(
                current_weights=current_weights,
                target_weights=target_weights,
                portfolio_value=portfolio_value,
                threshold=rebalance_threshold
            )
            
            result['orders'] = [
                {
                    'asset': o.asset_name,
                    'symbol': o.symbol,
                    'action': o.action,
                    'shares': o.shares,
                    'estimated_value': o.estimated_value,
                    'weight_delta': o.weight_delta
                }
                for o in orders
            ]
            
            if not orders:
                logger.info("No rebalancing needed - all positions within threshold")
                result['success'] = True
                result['execution_result'] = {'message': 'No rebalancing needed'}
            else:
                # 5. Execute orders
                execution_result = om.execute_orders(orders, dry_run=dry_run)
                result['execution_result'] = execution_result
                result['success'] = len(execution_result.get('failed', [])) == 0
                
        finally:
            om.disconnect()
        
    except Exception as e:
        logger.error(f"Strategy execution failed: {e}")
        result['error'] = str(e)
        result['success'] = False
    
    # Save execution log
    execution_end = datetime.now()
    result['duration_seconds'] = (execution_end - execution_start).total_seconds()
    
    log_filename = f"execution_{execution_start.strftime('%Y-%m-%d_%H%M%S')}.json"
    log_path = os.path.join(execution_log_dir, log_filename)
    
    with open(log_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    logger.info(f"Execution log saved to: {log_path}")
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"IBKR Strategy Execution Summary")
    print(f"{'='*50}")
    print(f"Timestamp: {result['timestamp']}")
    print(f"Quadrant: Q{result['quadrant']}")
    print(f"Portfolio Value: ${result.get('portfolio_value', 0):,.2f}")
    print(f"Dry Run: {dry_run}")
    print(f"Orders: {len(result['orders'])}")
    print(f"Success: {result['success']}")
    if result['error']:
        print(f"Error: {result['error']}")
    print(f"{'='*50}\n")
    
    return result


# Airflow-compatible function
def airflow_execute_strategy(**kwargs) -> Dict[str, Any]:
    """
    Wrapper for Airflow PythonOperator.
    
    Reads parameters from op_kwargs.
    """
    backtest_output_dir = kwargs.get('backtest_output_dir')
    dry_run = kwargs.get('dry_run', False)
    rebalance_threshold = kwargs.get('rebalance_threshold', REBALANCE_THRESHOLD)
    
    return execute_strategy(
        backtest_output_dir=backtest_output_dir,
        dry_run=dry_run,
        rebalance_threshold=rebalance_threshold
    )


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m ibkr.executor <backtest_output_dir> [--dry-run]")
        print("Example: python -m ibkr.executor data/US/backtest_results")
        sys.exit(1)
    
    backtest_dir = sys.argv[1]
    is_dry_run = '--dry-run' in sys.argv
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    result = execute_strategy(backtest_dir, dry_run=is_dry_run)
    
    if not result['success']:
        sys.exit(1)
