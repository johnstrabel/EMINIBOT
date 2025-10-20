"""
Global Settings for EMINIBOT
Core configuration for trading operations
"""

from datetime import time
import os
from pathlib import Path


# ============================================================================
# TRADING MODE
# ============================================================================
TRADING_MODE = os.getenv('TRADING_MODE', 'paper')  # 'paper', 'live', 'backtest'


# ============================================================================
# ACCOUNT SETTINGS
# ============================================================================
ACCOUNT_SIZE = 50000  # Starting capital for prop firm challenge
MAX_DAILY_LOSS_PCT = 3.0  # Max 3% daily loss (prop firm rule)
MAX_DAILY_LOSS = ACCOUNT_SIZE * (MAX_DAILY_LOSS_PCT / 100)  # $1,500


# ============================================================================
# INSTRUMENT SPECIFICATIONS - E-mini S&P 500 (ES)
# ============================================================================
SYMBOL = 'ES'
CONTRACT_NAME = 'E-mini S&P 500 Futures'
EXCHANGE = 'CME'

# Price specifications
TICK_SIZE = 0.25  # Minimum price increment (1 tick)
TICK_VALUE = 12.50  # Dollar value per tick ($12.50)
POINT_VALUE = 50.00  # Dollar value per full point ($50)
BIG_POINT_VALUE = POINT_VALUE  # Alias for clarity

# Contract specifications
CONTRACT_MULTIPLIER = 50  # $50 per index point
MICRO_CONTRACT = False  # False = standard ES, True = MES (micro)

# Margin requirements (approximate, varies by broker)
INTRADAY_MARGIN = 500  # ~$500 per contract for intraday
OVERNIGHT_MARGIN = 12000  # ~$12,000 per contract overnight


# ============================================================================
# RISK MANAGEMENT
# ============================================================================

# Position sizing
RISK_PER_TRADE_PCT = 1.0  # Risk 1% of account per trade
MAX_POSITION_SIZE = 2  # Max 2 contracts at once (conservative)
MIN_POSITION_SIZE = 1  # Always trade at least 1 contract

# Stop loss / Take profit
STOP_LOSS_TICKS = 8  # 8 ticks = $100 per contract
TAKE_PROFIT_1_TICKS = 12  # First target: 12 ticks = $150
TAKE_PROFIT_2_TICKS = 20  # Second target: 20 ticks = $250
PARTIAL_CLOSE_PCT = 0.5  # Close 50% at TP1, let rest run to TP2

# Trailing stop
USE_TRAILING_STOP = True
TRAIL_ACTIVATION_TICKS = 8  # Start trailing after 8 ticks profit
TRAIL_OFFSET_TICKS = 4  # Trail 4 ticks behind price

# Daily limits
MAX_TRADES_PER_DAY = 10  # Max 10 trades/day (prevent overtrading)
MAX_CONSECUTIVE_LOSSES = 3  # Stop after 3 losses in a row


# ============================================================================
# TRADE EXECUTION
# ============================================================================

# Order types
DEFAULT_ORDER_TYPE = 'LIMIT'  # 'MARKET', 'LIMIT', 'STOP'
SLIPPAGE_ALLOWANCE_TICKS = 2  # Allow 2 ticks slippage for limit orders

# Time in force
TIME_IN_FORCE = 'GTC'  # 'GTC' (Good Till Cancel) or 'DAY'

# Order validation
MIN_SIGNAL_STRENGTH = 60  # Minimum signal strength (0-100) to trade
MIN_INDICATORS_ALIGNED = 2  # Require at least 2 of 3 indicators agree


# ============================================================================
# BROKER / API SETTINGS
# ============================================================================

# NinjaTrader settings
NINJATRADER_HOST = os.getenv('NINJATRADER_HOST', 'localhost')
NINJATRADER_PORT = int(os.getenv('NINJATRADER_PORT', 36973))

# Rithmic settings (for live trading)
RITHMIC_API_KEY = os.getenv('RITHMIC_API_KEY', '')
RITHMIC_API_SECRET = os.getenv('RITHMIC_API_SECRET', '')

# Data provider
DATA_PROVIDER = os.getenv('DATA_PROVIDER', 'none')  # 'ninjatrader', 'rithmic', 'ib'


# ============================================================================
# PERFORMANCE TRACKING
# ============================================================================

# Metrics to track
TRACK_METRICS = [
    'total_trades',
    'winning_trades',
    'losing_trades',
    'win_rate',
    'profit_factor',
    'sharpe_ratio',
    'max_drawdown',
    'avg_win',
    'avg_loss',
    'largest_win',
    'largest_loss',
    'consecutive_wins',
    'consecutive_losses',
]

# Reporting frequency
DAILY_REPORT = True
WEEKLY_REPORT = True
TRADE_LOG_ENABLED = True


# ============================================================================
# SAFETY FEATURES
# ============================================================================

# Circuit breakers
ENABLE_CIRCUIT_BREAKER = True
CIRCUIT_BREAKER_LOSS_PCT = 2.0  # Pause trading at 2% daily loss

# Kill switch
ENABLE_KILL_SWITCH = True  # Emergency stop all trading
KILL_SWITCH_LOSS_PCT = 2.5  # Hard stop at 2.5% loss

# Pre-trade validation
VALIDATE_ORDERS = True  # Check order validity before submission
REQUIRE_SESSION_CHECK = True  # Verify session before trading


# ============================================================================
# TIMEZONE
# ============================================================================
TIMEZONE = 'America/Chicago'  # Central Time (where CME is located)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def ticks_to_dollars(ticks: int, num_contracts: int = 1) -> float:
    """Convert ticks to dollar value"""
    return ticks * TICK_VALUE * num_contracts


def dollars_to_ticks(dollars: float) -> int:
    """Convert dollar value to ticks"""
    return int(dollars / TICK_VALUE)


def points_to_dollars(points: float, num_contracts: int = 1) -> float:
    """Convert points to dollar value"""
    return points * POINT_VALUE * num_contracts


def calculate_position_size(account_balance: float, risk_pct: float = None) -> int:
    """
    Calculate position size based on account balance and risk
    
    Args:
        account_balance: Current account balance
        risk_pct: Risk percentage (defaults to RISK_PER_TRADE_PCT)
    
    Returns:
        Number of contracts to trade
    """
    if risk_pct is None:
        risk_pct = RISK_PER_TRADE_PCT
    
    risk_dollars = account_balance * (risk_pct / 100)
    stop_loss_dollars = STOP_LOSS_TICKS * TICK_VALUE
    
    position_size = int(risk_dollars / stop_loss_dollars)
    
    # Apply limits
    position_size = max(MIN_POSITION_SIZE, position_size)
    position_size = min(MAX_POSITION_SIZE, position_size)
    
    return position_size


def get_risk_reward_ratio() -> float:
    """Calculate risk:reward ratio"""
    risk = STOP_LOSS_TICKS * TICK_VALUE
    
    # Weighted average reward (50% at TP1, 50% at TP2)
    reward = (TAKE_PROFIT_1_TICKS * PARTIAL_CLOSE_PCT + 
              TAKE_PROFIT_2_TICKS * (1 - PARTIAL_CLOSE_PCT)) * TICK_VALUE
    
    return reward / risk if risk > 0 else 0


def validate_settings() -> bool:
    """Validate configuration settings"""
    errors = []
    
    # Check critical settings
    if STOP_LOSS_TICKS <= 0:
        errors.append("STOP_LOSS_TICKS must be positive")
    
    if TAKE_PROFIT_1_TICKS <= STOP_LOSS_TICKS:
        errors.append("TAKE_PROFIT_1_TICKS must be greater than STOP_LOSS_TICKS")
    
    if RISK_PER_TRADE_PCT <= 0 or RISK_PER_TRADE_PCT > 5:
        errors.append("RISK_PER_TRADE_PCT must be between 0 and 5")
    
    if MAX_DAILY_LOSS_PCT <= 0:
        errors.append("MAX_DAILY_LOSS_PCT must be positive")
    
    if MIN_POSITION_SIZE > MAX_POSITION_SIZE:
        errors.append("MIN_POSITION_SIZE cannot be greater than MAX_POSITION_SIZE")
    
    if errors:
        print("❌ Configuration Errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True


# ============================================================================
# DISPLAY SETTINGS (for startup)
# ============================================================================

def print_settings():
    """Print key settings on startup"""
    print("\n" + "="*70)
    print("EMINIBOT - TRADING CONFIGURATION")
    print("="*70)
    
    print(f"\n🎯 TRADING MODE: {TRADING_MODE.upper()}")
    print(f"💰 Account Size: ${ACCOUNT_SIZE:,.0f}")
    print(f"📊 Symbol: {SYMBOL} ({CONTRACT_NAME})")
    
    print(f"\n⚠️ RISK MANAGEMENT:")
    print(f"  Risk per trade: {RISK_PER_TRADE_PCT}% (${ACCOUNT_SIZE * RISK_PER_TRADE_PCT / 100:,.0f})")
    print(f"  Max daily loss: {MAX_DAILY_LOSS_PCT}% (${MAX_DAILY_LOSS:,.0f})")
    print(f"  Position size: {MIN_POSITION_SIZE}-{MAX_POSITION_SIZE} contracts")
    print(f"  Max trades/day: {MAX_TRADES_PER_DAY}")
    
    print(f"\n🎲 TRADE PARAMETERS:")
    print(f"  Stop Loss: {STOP_LOSS_TICKS} ticks (${ticks_to_dollars(STOP_LOSS_TICKS):,.2f} per contract)")
    print(f"  Take Profit 1: {TAKE_PROFIT_1_TICKS} ticks (${ticks_to_dollars(TAKE_PROFIT_1_TICKS):,.2f})")
    print(f"  Take Profit 2: {TAKE_PROFIT_2_TICKS} ticks (${ticks_to_dollars(TAKE_PROFIT_2_TICKS):,.2f})")
    print(f"  Risk:Reward Ratio: 1:{get_risk_reward_ratio():.2f}")
    
    if USE_TRAILING_STOP:
        print(f"  Trailing Stop: Activate at +{TRAIL_ACTIVATION_TICKS} ticks, trail by {TRAIL_OFFSET_TICKS} ticks")
    
    print(f"\n🔧 EXECUTION:")
    print(f"  Min Signal Strength: {MIN_SIGNAL_STRENGTH}/100")
    print(f"  Min Indicators Aligned: {MIN_INDICATORS_ALIGNED}/3")
    print(f"  Order Type: {DEFAULT_ORDER_TYPE}")
    
    print(f"\n🛡️ SAFETY:")
    print(f"  Circuit Breaker: {'✅ Enabled' if ENABLE_CIRCUIT_BREAKER else '❌ Disabled'} @ {CIRCUIT_BREAKER_LOSS_PCT}%")
    print(f"  Kill Switch: {'✅ Enabled' if ENABLE_KILL_SWITCH else '❌ Disabled'} @ {KILL_SWITCH_LOSS_PCT}%")
    
    print("\n" + "="*70 + "\n")


# ============================================================================
# VALIDATE ON IMPORT
# ============================================================================

if __name__ == "__main__":
    # Test configuration
    print("Testing configuration...")
    
    if validate_settings():
        print("✅ Configuration is valid!\n")
        print_settings()
        
        # Test helper functions
        print("\n📊 HELPER FUNCTION TESTS:")
        print(f"  8 ticks = ${ticks_to_dollars(8):.2f}")
        print(f"  $100 = {dollars_to_ticks(100)} ticks")
        print(f"  1 point = ${points_to_dollars(1):.2f}")
        print(f"  Position size for $50K account = {calculate_position_size(50000)} contracts")
        print(f"  Risk:Reward = 1:{get_risk_reward_ratio():.2f}")
    else:
        print("❌ Configuration validation failed!")