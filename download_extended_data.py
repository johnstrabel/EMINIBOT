"""
Download Extended ES Data for Backtesting
Downloads 60 days of 1-minute E-mini ES data
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time

print("="*70)
print("DOWNLOADING EXTENDED ES DATA")
print("="*70)

# Calculate date range (60 days back)
end_date = datetime.now()
start_date = end_date - timedelta(days=60)

print(f"\n📅 Date Range:")
print(f"   Start: {start_date.strftime('%Y-%m-%d')}")
print(f"   End: {end_date.strftime('%Y-%m-%d')}")
print(f"   Duration: 60 days")

# Download data in chunks (Yahoo Finance limits)
print("\n📊 Downloading ES futures data...")
print("   (This may take 2-5 minutes...)\n")

all_data = []
chunk_size = 7  # Download 7 days at a time

current_start = start_date
chunk_num = 1

while current_start < end_date:
    current_end = min(current_start + timedelta(days=chunk_size), end_date)
    
    print(f"   Chunk {chunk_num}: {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}", end="")
    
    try:
        # Download ES=F (E-mini S&P 500 futures)
        ticker = yf.Ticker("ES=F")
        df = ticker.history(
            start=current_start.strftime('%Y-%m-%d'),
            end=current_end.strftime('%Y-%m-%d'),
            interval='1m'
        )
        
        if len(df) > 0:
            all_data.append(df)
            print(f" ✅ ({len(df)} bars)")
        else:
            print(f" ⚠️ (no data)")
        
        # Small delay to avoid rate limiting
        time.sleep(1)
        
    except Exception as e:
        print(f" ❌ Error: {e}")
    
    current_start = current_end
    chunk_num += 1

# Combine all chunks
if all_data:
    print("\n📦 Combining data chunks...")
    combined_df = pd.concat(all_data)
    combined_df = combined_df.sort_index()
    
    # Remove duplicates
    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
    
    # Reset index to make datetime a column
    combined_df.reset_index(inplace=True)
    combined_df.rename(columns={'index': 'datetime'}, inplace=True)
    
    # Clean column names (lowercase)
    combined_df.columns = [col.lower() for col in combined_df.columns]
    
    # Save to CSV
    filename = f"data/historical/ES_yahoo_60days_{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}.csv"
    combined_df.to_csv(filename, index=False)
    
    print(f"\n✅ SUCCESS!")
    print(f"   Total bars: {len(combined_df):,}")
    print(f"   Date range: {combined_df['datetime'].iloc[0]} to {combined_df['datetime'].iloc[-1]}")
    print(f"   Price range: ${combined_df['close'].min():.2f} - ${combined_df['close'].max():.2f}")
    print(f"   Saved to: {filename}")
    
else:
    print("\n❌ ERROR: No data downloaded!")
    print("   Check your internet connection and try again.")

print("\n" + "="*70)