"""
IBKR Connection Manager
=======================
Handles connection and disconnection to Interactive Brokers TWS/Gateway.
"""

from ib_insync import IB, util
from typing import Optional
import logging

from .config import HOST, CURRENT_PORT, CLIENT_ID, CONNECTION_TIMEOUT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IBKRConnection:
    """
    Connection manager for Interactive Brokers.
    
    Usage:
        # As context manager (recommended)
        with IBKRConnection() as ib:
            positions = ib.positions()
        
        # Manual control
        conn = IBKRConnection()
        if conn.connect():
            # do stuff
            conn.disconnect()
    """
    
    def __init__(self, host: str = HOST, port: int = CURRENT_PORT, 
                 client_id: int = CLIENT_ID, timeout: int = CONNECTION_TIMEOUT):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.timeout = timeout
        self.ib: Optional[IB] = None
    
    def connect(self) -> bool:
        """
        Connect to TWS/IB Gateway.
        
        Returns:
            True if connection successful, False otherwise.
        """
        try:
            self.ib = IB()
            self.ib.connect(self.host, self.port, clientId=self.client_id, timeout=self.timeout)
            logger.info(f"✅ Connected to IBKR at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            self.ib = None
            return False
    
    def disconnect(self):
        """Disconnect from TWS/IB Gateway."""
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()
            logger.info("🔌 Disconnected from IBKR")
        self.ib = None
    
    def is_connected(self) -> bool:
        """Check if currently connected."""
        return self.ib is not None and self.ib.isConnected()
    
    def __enter__(self) -> IB:
        """Context manager entry."""
        if not self.connect():
            raise ConnectionError("Failed to connect to IBKR")
        return self.ib
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
        return False


def test_connection() -> bool:
    """
    Quick test to verify IBKR connection works.
    
    Returns:
        True if connection test passes.
    """
    print(f"🔗 Testing connection to IBKR at {HOST}:{CURRENT_PORT}...")
    
    conn = IBKRConnection()
    if conn.connect():
        print(f"✅ Connection successful!")
        
        # Get account info
        accounts = conn.ib.managedAccounts()
        print(f"📊 Managed accounts: {accounts}")
        
        # Get server time
        server_time = conn.ib.reqCurrentTime()
        print(f"🕐 Server time: {server_time}")
        
        conn.disconnect()
        return True
    else:
        print("❌ Connection failed. Check that TWS is running and API is enabled.")
        return False


if __name__ == "__main__":
    test_connection()
