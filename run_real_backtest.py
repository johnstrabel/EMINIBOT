"""
Run Backtest on Real ES Data
"""

import pandas as pd
from src.analysis.backtester import Backtester

print("="*70)
print("RUNNING BACKTEST ON REAL ES DATA")
print("="*70)

# Load the real ES data
print("\n📊 Loading data...")
df = pd.read_csv('data/historical/ES_yahoo_20251013_to_20251020.csv')

print(f"✅ Loaded {len(df)} bars")
print(f"   Date range: {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")
print(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

# Initialize backtester
backtester = Backtester(
    initial_capital=50000,
    commission_per_contract=1.0,
    slippage_ticks=1
)

# Run backtest
print("\n🚀 Running backtest...")
results = backtester.run(df, verbose=True)

# Get trade log
trade_log = backtester.get_trade_log()

if len(trade_log) > 0:
    print("\n" + "="*70)
    print("DETAILED TRADE LOG")
    print("="*70)
    
    print("\n📋 All Trades:")
    print(trade_log[['trade_id', 'side', 'entry_price', 'exit_price', 'total_pnl', 'exit_reason']].to_string())
    
    # Win/Loss breakdown
    wins = trade_log[trade_log['total_pnl'] > 0]
    losses = trade_log[trade_log['total_pnl'] <= 0]
    
    print(f"\n📊 Trade Breakdown:")
    print(f"   Winning trades: {len(wins)}")
    print(f"   Losing trades: {len(losses)}")
    
    if len(wins) > 0:
        print(f"\n🟢 Best trade: ${wins['total_pnl'].max():.2f}")
        best = wins.loc[wins['total_pnl'].idxmax()]
        print(f"   {best['trade_id']}: {best['side'].upper()} from ${best['entry_price']:.2f} to ${best['exit_price']:.2f}")
    
    if len(losses) > 0:
        print(f"\n🔴 Worst trade: ${losses['total_pnl'].min():.2f}")
        worst = losses.loc[losses['total_pnl'].idxmin()]
        print(f"   {worst['trade_id']}: {worst['side'].upper()} from ${worst['entry_price']:.2f} to ${worst['exit_price']:.2f}")
else:
    print("\n⚠️ No trades executed - signals may need adjustment")

print("\n" + "="*70)
print("BACKTEST COMPLETE!")
print("="*70)