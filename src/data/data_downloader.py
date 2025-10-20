"""
Data Downloader for EMINIBOT
Downloads historical E-mini ES futures data from multiple sources
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config import paths


class DataDownloader:
    """
    Downloads historical E-mini ES futures data
    
    Supported sources:
    - Yahoo Finance (free, limited history)
    - Polygon.io (paid, high quality)
    - Custom CSV upload
    """
    
    def __init__(self):
        """Initialize data downloader"""
        self.data_dir = paths.HISTORICAL_DATA_DIR
        paths.ensure_directories_exist()
    
    def download_yahoo(
        self,
        start_date: str,
        end_date: str,
        interval: str = '1m',
        symbol: str = 'ES=F'
    ) -> pd.DataFrame:
        """
        Download data from Yahoo Finance
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval (1m, 5m, 15m, 1h, 1d)
            symbol: Ticker symbol (default ES=F for E-mini S&P 500)
            
        Returns:
            DataFrame with OHLCV data
        
        Note: Yahoo Finance limits 1-minute data to ~7 days at a time
        """
        try:
            import yfinance as yf
        except ImportError:
            print("❌ yfinance not installed. Installing now...")
            os.system('pip install yfinance')
            import yfinance as yf
        
        print(f"\n📊 Downloading from Yahoo Finance...")
        print(f"   Symbol: {symbol}")
        print(f"   Interval: {interval}")
        print(f"   Range: {start_date} to {end_date}")
        
        # Download data
        df = yf.download(
            symbol,
            start=start_date,
            end=end_date,
            interval=interval,
            progress=False
        )
        
        if len(df) == 0:
            print("⚠️ No data returned from Yahoo Finance")
            return pd.DataFrame()
        
        # Clean and format data
        df = self._format_yahoo_data(df)
        
        print(f"✅ Downloaded {len(df)} bars")
        print(f"   Date range: {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")
        print(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        
        return df
    
    def download_polygon(
        self,
        start_date: str,
        end_date: str,
        timespan: str = 'minute',
        api_key: str = None
    ) -> pd.DataFrame:
        """
        Download data from Polygon.io (requires API key)
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            timespan: Timespan (minute, hour, day)
            api_key: Polygon.io API key (or set POLYGON_API_KEY env variable)
            
        Returns:
            DataFrame with OHLCV data
        
        Note: Requires Polygon.io account ($29/month for Starter)
        """
        if api_key is None:
            api_key = os.getenv('POLYGON_API_KEY')
        
        if not api_key:
            print("❌ Polygon.io API key required!")
            print("   Get one at: https://polygon.io/")
            print("   Set as environment variable: POLYGON_API_KEY")
            return pd.DataFrame()
        
        try:
            from polygon import RESTClient
        except ImportError:
            print("❌ polygon-api-client not installed. Installing now...")
            os.system('pip install polygon-api-client')
            from polygon import RESTClient
        
        print(f"\n📊 Downloading from Polygon.io...")
        print(f"   Timespan: {timespan}")
        print(f"   Range: {start_date} to {end_date}")
        
        # Initialize client
        client = RESTClient(api_key)
        
        # ES futures ticker format for Polygon
        ticker = "ES"  # Will need to adjust based on contract month
        
        # Download data
        aggs = client.get_aggs(
            ticker=ticker,
            multiplier=1,
            timespan=timespan,
            from_=start_date,
            to=end_date
        )
        
        if not aggs:
            print("⚠️ No data returned from Polygon.io")
            return pd.DataFrame()
        
        # Convert to DataFrame
        data = []
        for agg in aggs:
            data.append({
                'datetime': datetime.fromtimestamp(agg.timestamp / 1000),
                'open': agg.open,
                'high': agg.high,
                'low': agg.low,
                'close': agg.close,
                'volume': agg.volume
            })
        
        df = pd.DataFrame(data)
        
        print(f"✅ Downloaded {len(df)} bars")
        print(f"   Date range: {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")
        
        return df
    
    def _format_yahoo_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Format Yahoo Finance data"""
        df = df.copy()
        
        # Handle MultiIndex columns (when downloading multiple tickers)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Reset index to get datetime as column
        df = df.reset_index()
        
        # Rename columns to lowercase
        if hasattr(df.columns, 'str'):
            df.columns = df.columns.str.lower()
        else:
            df.columns = [col.lower() if isinstance(col, str) else col for col in df.columns]
        
        # Rename 'date' or 'datetime' column
        if 'date' in df.columns:
            df = df.rename(columns={'date': 'datetime'})
        
        # Ensure datetime type
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Select and order columns
        columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        df = df[columns]
        
        # Remove any NaN values
        df = df.dropna()
        
        return df
    
    def load_csv(self, filepath: str) -> pd.DataFrame:
        """
        Load data from CSV file
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame with OHLCV data
        """
        print(f"\n📊 Loading from CSV: {filepath}")
        
        df = pd.read_csv(filepath)
        
        # Try to identify datetime column
        datetime_cols = ['datetime', 'date', 'timestamp', 'time']
        datetime_col = None
        
        for col in datetime_cols:
            if col in df.columns:
                datetime_col = col
                break
        
        if datetime_col:
            df['datetime'] = pd.to_datetime(df[datetime_col])
        else:
            print("⚠️ No datetime column found in CSV")
            return pd.DataFrame()
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                print(f"⚠️ Missing required column: {col}")
                return pd.DataFrame()
        
        # Select columns
        df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
        
        print(f"✅ Loaded {len(df)} bars")
        return df
    
    def save_data(
        self,
        df: pd.DataFrame,
        filename: str = None,
        source: str = 'yahoo'
    ) -> str:
        """
        Save data to CSV
        
        Args:
            df: DataFrame to save
            filename: Optional custom filename
            source: Data source name
            
        Returns:
            Path to saved file
        """
        if filename is None:
            start_date = df['datetime'].iloc[0].strftime('%Y%m%d')
            end_date = df['datetime'].iloc[-1].strftime('%Y%m%d')
            filename = f"ES_{source}_{start_date}_to_{end_date}.csv"
        
        filepath = self.data_dir / filename
        df.to_csv(filepath, index=False)
        
        print(f"\n💾 Saved data to: {filepath}")
        return str(filepath)
    
    def download_multiple_periods(
        self,
        start_date: str,
        end_date: str,
        interval: str = '1m',
        days_per_chunk: int = 7
    ) -> pd.DataFrame:
        """
        Download data in chunks (for Yahoo Finance 1-minute limitation)
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval
            days_per_chunk: Days per download chunk (default 7 for Yahoo 1m data)
            
        Returns:
            Combined DataFrame
        """
        print(f"\n📊 Downloading data in {days_per_chunk}-day chunks...")
        
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        all_data = []
        current = start
        
        while current < end:
            chunk_end = min(current + timedelta(days=days_per_chunk), end)
            
            print(f"\n   Chunk: {current.date()} to {chunk_end.date()}")
            
            chunk_data = self.download_yahoo(
                start_date=current.strftime('%Y-%m-%d'),
                end_date=chunk_end.strftime('%Y-%m-%d'),
                interval=interval
            )
            
            if len(chunk_data) > 0:
                all_data.append(chunk_data)
            
            current = chunk_end
        
        if not all_data:
            print("⚠️ No data downloaded")
            return pd.DataFrame()
        
        # Combine all chunks
        df = pd.concat(all_data, ignore_index=True)
        df = df.drop_duplicates(subset=['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)
        
        print(f"\n✅ Total downloaded: {len(df)} bars")
        
        return df


# ============================================================================
# TESTING & EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("ES FUTURES DATA DOWNLOADER")
    print("="*70)
    
    downloader = DataDownloader()
    
    # Example 1: Download recent data from Yahoo Finance
    print("\n📊 Example 1: Download 7 days of 1-minute data")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    df = downloader.download_yahoo(
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        interval='1m'
    )
    
    if len(df) > 0:
        print("\n📋 Sample data:")
        print(df.head())
        
        # Save the data
        filepath = downloader.save_data(df, source='yahoo')
        print(f"\n✅ Data ready for backtesting!")
        print(f"   Load with: pd.read_csv('{filepath}')")
    
    # Example 2: Show how to download multiple months
    print("\n\n📊 Example 2: How to download 2-3 months of data")
    print("   (This would take several minutes, so it's commented out)")
    print("""
    # Download last 60 days in 7-day chunks
    df_long = downloader.download_multiple_periods(
        start_date=(datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d'),
        end_date=datetime.now().strftime('%Y-%m-%d'),
        interval='1m',
        days_per_chunk=7
    )
    
    if len(df_long) > 0:
        downloader.save_data(df_long, source='yahoo_60days')
    """)
    
    print("\n" + "="*70)
    print("READY TO DOWNLOAD DATA!")
    print("="*70)
    print("\n💡 NEXT STEPS:")
    print("   1. Run this script to download data")
    print("   2. Load the CSV in your backtester")
    print("   3. Run full backtest on real ES data!")
    print("\n📝 For Polygon.io (paid, better quality):")
    print("   1. Sign up at https://polygon.io/ ($29/month)")
    print("   2. Get API key")
    print("   3. Set environment variable: POLYGON_API_KEY=your_key")
    print("   4. Use downloader.download_polygon()")