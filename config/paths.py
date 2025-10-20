"""
File Paths Configuration for EMINIBOT
Centralized path management for data, logs, and models
"""

from pathlib import Path
import os
from datetime import datetime


# ============================================================================
# ROOT DIRECTORY
# ============================================================================

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / 'data'


# ============================================================================
# DATA DIRECTORIES
# ============================================================================

# Raw data (downloaded from broker/data provider)
RAW_DATA_DIR = DATA_ROOT / 'raw'

# Processed data (cleaned, resampled, with indicators)
PROCESSED_DATA_DIR = DATA_ROOT / 'processed'

# Historical data for backtesting
HISTORICAL_DATA_DIR = DATA_ROOT / 'historical'

# Real-time data cache
REALTIME_DATA_DIR = DATA_ROOT / 'realtime'


# ============================================================================
# LOG DIRECTORIES
# ============================================================================

LOG_DIR = DATA_ROOT / 'logs'

# Trade logs (every executed trade)
TRADE_LOG_DIR = LOG_DIR / 'trades'

# Performance logs (daily/weekly summaries)
PERFORMANCE_LOG_DIR = LOG_DIR / 'performance'

# System logs (errors, warnings, info)
SYSTEM_LOG_DIR = LOG_DIR / 'system'

# Backtest logs
BACKTEST_LOG_DIR = LOG_DIR / 'backtests'


# ============================================================================
# MODEL DIRECTORIES
# ============================================================================

MODEL_DIR = DATA_ROOT / 'models'

# Saved models (if using ML in future)
SAVED_MODELS_DIR = MODEL_DIR / 'saved'

# Model parameters
MODEL_PARAMS_DIR = MODEL_DIR / 'parameters'


# ============================================================================
# RESULTS DIRECTORIES
# ============================================================================

RESULTS_DIR = DATA_ROOT / 'results'

# Backtest results
BACKTEST_RESULTS_DIR = RESULTS_DIR / 'backtests'

# Paper trading results
PAPER_RESULTS_DIR = RESULTS_DIR / 'paper_trading'

# Live trading results
LIVE_RESULTS_DIR = RESULTS_DIR / 'live_trading'


# ============================================================================
# REPORTS DIRECTORIES
# ============================================================================

REPORTS_DIR = DATA_ROOT / 'reports'

# Daily reports
DAILY_REPORTS_DIR = REPORTS_DIR / 'daily'

# Weekly reports
WEEKLY_REPORTS_DIR = REPORTS_DIR / 'weekly'

# Monthly reports
MONTHLY_REPORTS_DIR = REPORTS_DIR / 'monthly'


# ============================================================================
# FILE PATTERNS
# ============================================================================

# Data file patterns
RAW_DATA_PATTERN = 'ES_{date}_{timeframe}.csv'
PROCESSED_DATA_PATTERN = 'ES_{date}_{timeframe}_processed.csv'
HISTORICAL_DATA_PATTERN = 'ES_{start_date}_to_{end_date}_{timeframe}.csv'

# Log file patterns
TRADE_LOG_PATTERN = 'trades_{date}.log'
PERFORMANCE_LOG_PATTERN = 'performance_{date}.log'
SYSTEM_LOG_PATTERN = 'system_{date}.log'
BACKTEST_LOG_PATTERN = 'backtest_{timestamp}.log'

# Report file patterns
DAILY_REPORT_PATTERN = 'daily_report_{date}.html'
WEEKLY_REPORT_PATTERN = 'weekly_report_{week_start}.html'
MONTHLY_REPORT_PATTERN = 'monthly_report_{year}_{month}.html'


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def ensure_directories_exist():
    """Create all necessary directories if they don't exist"""
    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        HISTORICAL_DATA_DIR,
        REALTIME_DATA_DIR,
        TRADE_LOG_DIR,
        PERFORMANCE_LOG_DIR,
        SYSTEM_LOG_DIR,
        BACKTEST_LOG_DIR,
        SAVED_MODELS_DIR,
        MODEL_PARAMS_DIR,
        BACKTEST_RESULTS_DIR,
        PAPER_RESULTS_DIR,
        LIVE_RESULTS_DIR,
        DAILY_REPORTS_DIR,
        WEEKLY_REPORTS_DIR,
        MONTHLY_REPORTS_DIR,
    ]
    
    created = []
    for directory in directories:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            created.append(directory)
    
    if created:
        print(f"✅ Created {len(created)} directories")
    
    return created


def get_raw_data_path(date: str, timeframe: str = '1min') -> Path:
    """
    Get path for raw data file
    
    Args:
        date: Date string (YYYY-MM-DD)
        timeframe: Timeframe (1min, 5min, etc.)
    
    Returns:
        Path to raw data file
    """
    filename = RAW_DATA_PATTERN.format(date=date, timeframe=timeframe)
    return RAW_DATA_DIR / filename


def get_processed_data_path(date: str, timeframe: str = '1min') -> Path:
    """Get path for processed data file"""
    filename = PROCESSED_DATA_PATTERN.format(date=date, timeframe=timeframe)
    return PROCESSED_DATA_DIR / filename


def get_historical_data_path(start_date: str, end_date: str, timeframe: str = '1min') -> Path:
    """Get path for historical data file"""
    filename = HISTORICAL_DATA_PATTERN.format(
        start_date=start_date,
        end_date=end_date,
        timeframe=timeframe
    )
    return HISTORICAL_DATA_DIR / filename


def get_trade_log_path(date: str = None) -> Path:
    """Get path for trade log file"""
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')
    
    filename = TRADE_LOG_PATTERN.format(date=date)
    return TRADE_LOG_DIR / filename


def get_performance_log_path(date: str = None) -> Path:
    """Get path for performance log file"""
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')
    
    filename = PERFORMANCE_LOG_PATTERN.format(date=date)
    return PERFORMANCE_LOG_DIR / filename


def get_system_log_path(date: str = None) -> Path:
    """Get path for system log file"""
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')
    
    filename = SYSTEM_LOG_PATTERN.format(date=date)
    return SYSTEM_LOG_DIR / filename


def get_backtest_log_path(timestamp: str = None) -> Path:
    """Get path for backtest log file"""
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    filename = BACKTEST_LOG_PATTERN.format(timestamp=timestamp)
    return BACKTEST_LOG_DIR / filename


def get_daily_report_path(date: str = None) -> Path:
    """Get path for daily report"""
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')
    
    filename = DAILY_REPORT_PATTERN.format(date=date)
    return DAILY_REPORTS_DIR / filename


def get_weekly_report_path(week_start: str = None) -> Path:
    """Get path for weekly report"""
    if week_start is None:
        # Get Monday of current week
        now = datetime.now()
        monday = now - pd.Timedelta(days=now.weekday())
        week_start = monday.strftime('%Y-%m-%d')
    
    filename = WEEKLY_REPORT_PATTERN.format(week_start=week_start)
    return WEEKLY_REPORTS_DIR / filename


def get_monthly_report_path(year: int = None, month: int = None) -> Path:
    """Get path for monthly report"""
    if year is None or month is None:
        now = datetime.now()
        year = now.year
        month = now.month
    
    filename = MONTHLY_REPORT_PATTERN.format(year=year, month=f"{month:02d}")
    return MONTHLY_REPORTS_DIR / filename


def list_data_files(directory: Path, pattern: str = "*.csv") -> list:
    """
    List all data files in a directory
    
    Args:
        directory: Directory to search
        pattern: File pattern to match
    
    Returns:
        List of file paths
    """
    if not directory.exists():
        return []
    
    return sorted(directory.glob(pattern))


def get_latest_data_file(directory: Path, pattern: str = "*.csv") -> Path:
    """Get most recent data file"""
    files = list_data_files(directory, pattern)
    return files[-1] if files else None


def clean_old_logs(days_to_keep: int = 30):
    """
    Clean up old log files
    
    Args:
        days_to_keep: Number of days of logs to keep
    """
    import time
    
    cutoff_time = time.time() - (days_to_keep * 86400)
    deleted = []
    
    log_dirs = [TRADE_LOG_DIR, PERFORMANCE_LOG_DIR, SYSTEM_LOG_DIR]
    
    for log_dir in log_dirs:
        if not log_dir.exists():
            continue
        
        for log_file in log_dir.glob("*.log"):
            if log_file.stat().st_mtime < cutoff_time:
                log_file.unlink()
                deleted.append(log_file)
    
    if deleted:
        print(f"🗑️ Deleted {len(deleted)} old log files")
    
    return deleted


def get_disk_usage() -> dict:
    """Get disk usage for data directories"""
    import shutil
    
    usage = {}
    
    directories = {
        'raw_data': RAW_DATA_DIR,
        'processed_data': PROCESSED_DATA_DIR,
        'historical_data': HISTORICAL_DATA_DIR,
        'logs': LOG_DIR,
        'models': MODEL_DIR,
        'results': RESULTS_DIR,
    }
    
    for name, directory in directories.items():
        if directory.exists():
            total_size = sum(f.stat().st_size for f in directory.rglob('*') if f.is_file())
            usage[name] = {
                'bytes': total_size,
                'mb': total_size / (1024 * 1024),
                'gb': total_size / (1024 * 1024 * 1024),
            }
        else:
            usage[name] = {'bytes': 0, 'mb': 0, 'gb': 0}
    
    return usage


def print_directory_structure():
    """Print the directory structure"""
    print("\n" + "="*70)
    print("PROJECT DIRECTORY STRUCTURE")
    print("="*70)
    
    print(f"\n📂 Project Root: {PROJECT_ROOT}")
    
    print("\n📊 DATA DIRECTORIES:")
    print(f"  Raw Data: {RAW_DATA_DIR}")
    print(f"  Processed Data: {PROCESSED_DATA_DIR}")
    print(f"  Historical Data: {HISTORICAL_DATA_DIR}")
    print(f"  Real-time Data: {REALTIME_DATA_DIR}")
    
    print("\n📝 LOG DIRECTORIES:")
    print(f"  Trade Logs: {TRADE_LOG_DIR}")
    print(f"  Performance Logs: {PERFORMANCE_LOG_DIR}")
    print(f"  System Logs: {SYSTEM_LOG_DIR}")
    print(f"  Backtest Logs: {BACKTEST_LOG_DIR}")
    
    print("\n🤖 MODEL DIRECTORIES:")
    print(f"  Saved Models: {SAVED_MODELS_DIR}")
    print(f"  Model Params: {MODEL_PARAMS_DIR}")
    
    print("\n📈 RESULTS DIRECTORIES:")
    print(f"  Backtest Results: {BACKTEST_RESULTS_DIR}")
    print(f"  Paper Trading: {PAPER_RESULTS_DIR}")
    print(f"  Live Trading: {LIVE_RESULTS_DIR}")
    
    print("\n📄 REPORTS DIRECTORIES:")
    print(f"  Daily Reports: {DAILY_REPORTS_DIR}")
    print(f"  Weekly Reports: {WEEKLY_REPORTS_DIR}")
    print(f"  Monthly Reports: {MONTHLY_REPORTS_DIR}")
    
    print("\n" + "="*70 + "\n")


def print_disk_usage():
    """Print disk usage for data directories"""
    usage = get_disk_usage()
    
    print("\n" + "="*70)
    print("DISK USAGE")
    print("="*70 + "\n")
    
    for name, sizes in usage.items():
        print(f"{name:20s}: {sizes['mb']:>8.2f} MB ({sizes['gb']:.3f} GB)")
    
    total_mb = sum(s['mb'] for s in usage.values())
    total_gb = sum(s['gb'] for s in usage.values())
    
    print(f"\n{'TOTAL':20s}: {total_mb:>8.2f} MB ({total_gb:.3f} GB)")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    print("Testing paths configuration...")
    
    # Create directories
    print("\n📁 Creating directories...")
    ensure_directories_exist()
    
    # Print structure
    print_directory_structure()
    
    # Print disk usage
    print_disk_usage()
    
    # Test path generation
    print("\n🧪 TESTING PATH GENERATION:")
    print(f"  Raw data path: {get_raw_data_path('2024-01-01', '1min')}")
    print(f"  Trade log path: {get_trade_log_path()}")
    print(f"  Backtest log path: {get_backtest_log_path()}")
    print(f"  Daily report path: {get_daily_report_path()}")
    
    print("\n✅ Paths configuration complete!")