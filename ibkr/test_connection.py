"""
IBKR Connection Test Script
Run with: python -m ibkr.test_connection
"""
from ib_insync import IB
import logging

# Enable detailed logging
logging.basicConfig(level=logging.DEBUG)

# Configuration - WSL to Windows connection
IBKR_HOST = '172.22.112.1'  # Windows vEthernet IP (destination)
IBKR_PORT = 7497            # Paper trading port
CLIENT_ID = 2
TIMEOUT = 30                # Longer timeout for handshake

print(f"🔗 Connecting to IBKR Paper Trading...")
print(f"   Host: {IBKR_HOST}:{IBKR_PORT}, ClientID: {CLIENT_ID}, Timeout: {TIMEOUT}s")

ib = IB()

try:
    ib.connect(IBKR_HOST, IBKR_PORT, clientId=CLIENT_ID, timeout=TIMEOUT)
    print("✅ Connected successfully!")
    
    print("\n=== ACCOUNT INFO ===")
    accounts = ib.managedAccounts()
    print(f"Account: {accounts[0] if accounts else 'N/A'}")
    
    summary = ib.accountSummary()
    for s in summary:
        if s.tag in ['NetLiquidation', 'TotalCashValue', 'AvailableFunds', 'BuyingPower']:
            print(f"  {s.tag}: {s.value} {s.currency}")
    
    print("\n=== POSITIONS ===")
    positions = ib.positions()
    if positions:
        for pos in positions:
            print(f"  {pos.contract.symbol}: {pos.position} shares @ avg {pos.avgCost:.2f}")
    else:
        print("  Aucune position (compte paper vide)")
    
except Exception as e:
    print(f"❌ Connection failed: {type(e).__name__}: {e}")
    print("\n🔧 Troubleshooting tips:")
    print("   1. Is TWS running and logged in?")
    print("   2. In TWS: File → Global Configuration → API → Settings:")
    print("      - 'Enable ActiveX and Socket Clients' = checked")
    print("      - 'Read-Only API' = UNCHECKED (important!)")
    print("      - 'Socket port' = 7497")
    print("      - 'Allow connections from localhost only' = unchecked")
    print(f"   3. Trusted IPs should include: 172.22.122.13")
finally:
    if ib.isConnected():
        ib.disconnect()
        print("\n✅ Test completed!")
