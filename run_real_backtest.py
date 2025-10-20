"""
Run Backtest on Real ES Data
Automatically uses the most recent data file
"""

import pandas as pd
from pathlib import Path
from src.analysis.backtester import Backtester

print("="*70)
print("RUNNING BACKTEST ON REAL ES DATA")
print("="*70)

# Find the most recent data file
data_dir = Path('data/historical')
data_files = list(data_dir.glob('ES_yahoo_*.csv'))

if not data_files:
    print("\n❌ ERROR: No data files found!")
    print("   Run: python download_extended_data.py")
    exit(1)

# Get the largest (most recent/most data) file
latest_file = max(data_files, key=lambda f: f.stat().st_size)

print(f"\n📊 Loading data from: {latest_file.name}")
df = pd.read_csv(latest_file)

print(f"✅ Loaded {len(df):,} bars")
print(f"   Date range: {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")
print(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

# Initialize backtester
print("\n🚀 Running backtest...")
backtester = Backtester(
    initial_capital=50000,
    commission_per_contract=1.0,
    slippage_ticks=1
)

# Run backtest
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
    
    # Exit reason breakdown
    print(f"\n📊 Exit Reasons:")
    exit_reasons = trade_log['exit_reason'].value_counts()
    for reason, count in exit_reasons.items():
        print(f"   {reason}: {count}")
    
else:
    print("\n⚠️ No trades executed - signals may need adjustment")

print("\n" + "="*70)
print("BACKTEST COMPLETE!")
print("="*70)