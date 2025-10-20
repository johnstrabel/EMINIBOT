import pandas as pd

df = pd.read_csv('data/historical/ES_yahoo_60days_20250821_to_20251020.csv')

print("="*70)
print("DATA QUALITY CHECK")
print("="*70)

# Check for missing data
print(f"\n📊 Total bars: {len(df)}")
print(f"Date range: {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")

# Check for NaN values
print(f"\nMissing values:")
print(df.isnull().sum())

# Check volume distribution
print(f"\nVolume stats:")
print(f"  Mean: {df['volume'].mean():.0f}")
print(f"  Min: {df['volume'].min():.0f}")
print(f"  Max: {df['volume'].max():.0f}")
print(f"  Zero volume bars: {(df['volume'] == 0).sum()}")

# Check by date
print(f"\nBars per day:")
df['date'] = pd.to_datetime(df['datetime']).dt.date
daily_counts = df.groupby('date').size()
print(daily_counts)

print("\n" + "="*70)