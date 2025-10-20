"""
Session Times for E-mini ES Futures
Defines trading sessions and market hours
"""

from datetime import time, datetime
import pytz


# ============================================================================
# TIMEZONE
# ============================================================================
TIMEZONE = pytz.timezone('America/Chicago')  # CME is in Central Time


# ============================================================================
# MARKET HOURS (Central Time)
# ============================================================================

# E-mini ES trades nearly 24 hours
# Sunday 6:00 PM - Friday 5:00 PM (with brief maintenance windows)

MARKET_OPEN = time(18, 0)   # 6:00 PM Sunday
MARKET_CLOSE = time(17, 0)  # 5:00 PM Friday

# Daily maintenance window
MAINTENANCE_START = time(17, 0)  # 5:00 PM
MAINTENANCE_END = time(18, 0)    # 6:00 PM


# ============================================================================
# TRADING SESSIONS
# ============================================================================

SESSIONS = {
    'overnight': {
        'name': 'Overnight',
        'start': time(18, 0),   # 6:00 PM
        'end': time(9, 30),     # 9:30 AM
        'description': 'Electronic trading session',
        'characteristics': [
            'Lower liquidity',
            'Wider spreads',
            'Less institutional activity',
            'Often gaps at RTH open',
        ],
        'risk_level': 'medium',
        'liquidity': 'low',
        'volatility': 'low',
    },
    
    'opening': {
        'name': 'Opening Range',
        'start': time(9, 30),   # 9:30 AM
        'end': time(10, 30),    # 10:30 AM
        'description': 'First hour of Regular Trading Hours',
        'characteristics': [
            'High volatility',
            'Institutional order flow',
            'Often establishes daily range',
            'Gap fills common',
        ],
        'risk_level': 'high',
        'liquidity': 'high',
        'volatility': 'high',
    },
    
    'midday': {
        'name': 'Midday Session',
        'start': time(10, 30),  # 10:30 AM
        'end': time(14, 0),     # 2:00 PM
        'description': 'Core trading session - best conditions',
        'characteristics': [
            'Peak liquidity',
            'Tightest spreads',
            'Clear trends',
            'Institutional participation',
            'Best risk:reward',
        ],
        'risk_level': 'low',
        'liquidity': 'peak',
        'volatility': 'moderate',
    },
    
    'closing': {
        'name': 'Closing Session',
        'start': time(14, 0),   # 2:00 PM
        'end': time(16, 0),     # 4:00 PM
        'description': 'Final 2 hours before equity close',
        'characteristics': [
            'Institutional rebalancing',
            'Index fund flows',
            'Increased volatility near close',
            'Good liquidity',
        ],
        'risk_level': 'medium',
        'liquidity': 'high',
        'volatility': 'moderate-high',
    },
    
    'after_hours': {
        'name': 'After Hours',
        'start': time(16, 0),   # 4:00 PM
        'end': time(18, 0),     # 6:00 PM
        'description': 'Post-equity close, pre-maintenance',
        'characteristics': [
            'Lower liquidity',
            'Wider spreads',
            'Less predictable',
            'Avoid trading',
        ],
        'risk_level': 'very_high',
        'liquidity': 'very_low',
        'volatility': 'unpredictable',
    },
}


# ============================================================================
# REGULAR TRADING HOURS (RTH)
# ============================================================================

RTH_START = time(9, 30)   # 9:30 AM
RTH_END = time(16, 0)     # 4:00 PM

# RTH sub-sessions
RTH_SESSIONS = ['opening', 'midday', 'closing']


# ============================================================================
# ELECTRONIC TRADING HOURS (ETH)
# ============================================================================

ETH_START = time(18, 0)   # 6:00 PM
ETH_END = time(17, 0)     # 5:00 PM next day

ETH_SESSIONS = ['overnight', 'after_hours']


# ============================================================================
# OPTIMAL TRADING WINDOWS
# ============================================================================

OPTIMAL_WINDOWS = {
    'primary': {
        'start': time(10, 30),   # 10:30 AM
        'end': time(14, 0),      # 2:00 PM
        'reason': 'Peak liquidity, clear trends, best conditions',
    },
    
    'secondary': {
        'start': time(14, 0),    # 2:00 PM
        'end': time(15, 30),     # 3:30 PM
        'reason': 'Good liquidity, institutional flows',
    },
    
    'avoid': {
        'ranges': [
            (time(9, 30), time(10, 0)),   # First 30 min (choppy)
            (time(15, 45), time(16, 0)),  # Last 15 min (unpredictable)
            (time(16, 0), time(18, 0)),   # After hours
            (time(12, 0), time(13, 0)),   # Lunch hour (lower volume)
        ],
        'reason': 'High volatility, low liquidity, or unpredictable',
    },
}


# ============================================================================
# SESSION RISK MULTIPLIERS
# ============================================================================

SESSION_RISK = {
    'overnight': 0.5,     # Half position size
    'opening': 0.75,      # 75% position size
    'midday': 1.0,        # Full position size
    'closing': 0.85,      # 85% position size
    'after_hours': 0.0,   # No trading
}


# ============================================================================
# SESSION QUALITY SCORES (0-100)
# ============================================================================

SESSION_QUALITY = {
    'overnight': 40,      # Below average
    'opening': 60,        # Good but volatile
    'midday': 100,        # Optimal
    'closing': 75,        # Good
    'after_hours': 0,     # Avoid
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_current_session(dt: datetime = None) -> str:
    """
    Get current trading session
    
    Args:
        dt: Datetime to check (defaults to now)
    
    Returns:
        Session name
    """
    if dt is None:
        dt = datetime.now(TIMEZONE)
    
    # Ensure timezone aware
    if dt.tzinfo is None:
        dt = TIMEZONE.localize(dt)
    else:
        dt = dt.astimezone(TIMEZONE)
    
    current_time = dt.time()
    
    # Check each session
    for session_name, session_info in SESSIONS.items():
        start = session_info['start']
        end = session_info['end']
        
        if session_name == 'overnight':
            # Overnight wraps around midnight
            if current_time >= start or current_time < end:
                return session_name
        else:
            if start <= current_time < end:
                return session_name
    
    return 'after_hours'


def is_rth(dt: datetime = None) -> bool:
    """Check if currently in Regular Trading Hours"""
    if dt is None:
        dt = datetime.now(TIMEZONE)
    
    session = get_current_session(dt)
    return session in RTH_SESSIONS


def is_optimal_window(dt: datetime = None) -> bool:
    """Check if currently in optimal trading window"""
    if dt is None:
        dt = datetime.now(TIMEZONE)
    
    if dt.tzinfo is None:
        dt = TIMEZONE.localize(dt)
    else:
        dt = dt.astimezone(TIMEZONE)
    
    current_time = dt.time()
    
    # Check primary window
    primary = OPTIMAL_WINDOWS['primary']
    if primary['start'] <= current_time < primary['end']:
        return True
    
    # Check secondary window
    secondary = OPTIMAL_WINDOWS['secondary']
    if secondary['start'] <= current_time < secondary['end']:
        return True
    
    return False


def should_avoid_trading(dt: datetime = None) -> bool:
    """Check if current time is in avoid window"""
    if dt is None:
        dt = datetime.now(TIMEZONE)
    
    if dt.tzinfo is None:
        dt = TIMEZONE.localize(dt)
    else:
        dt = dt.astimezone(TIMEZONE)
    
    current_time = dt.time()
    
    for start, end in OPTIMAL_WINDOWS['avoid']['ranges']:
        if start <= current_time < end:
            return True
    
    return False


def is_market_open(dt: datetime = None) -> bool:
    """Check if market is open"""
    if dt is None:
        dt = datetime.now(TIMEZONE)
    
    if dt.tzinfo is None:
        dt = TIMEZONE.localize(dt)
    else:
        dt = dt.astimezone(TIMEZONE)
    
    # Check for maintenance window
    current_time = dt.time()
    if MAINTENANCE_START <= current_time < MAINTENANCE_END:
        return False
    
    # Check day of week (closed Saturday)
    weekday = dt.weekday()
    if weekday == 5:  # Saturday
        return False
    
    return True


def get_session_info(session_name: str) -> dict:
    """Get detailed info about a session"""
    return SESSIONS.get(session_name, {})


def get_session_risk_multiplier(session_name: str) -> float:
    """Get risk multiplier for a session"""
    return SESSION_RISK.get(session_name, 0.0)


def get_session_quality(session_name: str) -> int:
    """Get quality score for a session"""
    return SESSION_QUALITY.get(session_name, 0)


def print_session_info():
    """Print all session information"""
    print("\n" + "="*70)
    print("E-MINI ES TRADING SESSIONS")
    print("="*70)
    
    print(f"\n⏰ TIMEZONE: {TIMEZONE}")
    print(f"📅 Market Hours: {MARKET_OPEN.strftime('%I:%M %p')} - {MARKET_CLOSE.strftime('%I:%M %p')}")
    print(f"🔧 Maintenance: {MAINTENANCE_START.strftime('%I:%M %p')} - {MAINTENANCE_END.strftime('%I:%M %p')}")
    
    print("\n📊 TRADING SESSIONS:")
    for session_name, info in SESSIONS.items():
        print(f"\n  {info['name'].upper()}:")
        print(f"    Time: {info['start'].strftime('%I:%M %p')} - {info['end'].strftime('%I:%M %p')}")
        print(f"    Risk Level: {info['risk_level']}")
        print(f"    Liquidity: {info['liquidity']}")
        print(f"    Quality Score: {SESSION_QUALITY[session_name]}/100")
        print(f"    Position Size: {SESSION_RISK[session_name]*100:.0f}%")
    
    print("\n🎯 OPTIMAL TRADING WINDOWS:")
    print(f"  Primary: {OPTIMAL_WINDOWS['primary']['start'].strftime('%I:%M %p')} - {OPTIMAL_WINDOWS['primary']['end'].strftime('%I:%M %p')}")
    print(f"    {OPTIMAL_WINDOWS['primary']['reason']}")
    print(f"  Secondary: {OPTIMAL_WINDOWS['secondary']['start'].strftime('%I:%M %p')} - {OPTIMAL_WINDOWS['secondary']['end'].strftime('%I:%M %p')}")
    print(f"    {OPTIMAL_WINDOWS['secondary']['reason']}")
    
    print("\n⚠️ AVOID TRADING:")
    for start, end in OPTIMAL_WINDOWS['avoid']['ranges']:
        print(f"  {start.strftime('%I:%M %p')} - {end.strftime('%I:%M %p')}")
    print(f"  Reason: {OPTIMAL_WINDOWS['avoid']['reason']}")
    
    print("\n" + "="*70 + "\n")


def print_current_status():
    """Print current market status"""
    now = datetime.now(TIMEZONE)
    
    print("\n" + "="*70)
    print("CURRENT MARKET STATUS")
    print("="*70)
    
    print(f"\n🕐 Current Time: {now.strftime('%A, %B %d, %Y %I:%M:%S %p %Z')}")
    
    market_open = is_market_open(now)
    print(f"📊 Market Status: {'🟢 OPEN' if market_open else '🔴 CLOSED'}")
    
    if market_open:
        session = get_current_session(now)
        session_info = get_session_info(session)
        
        print(f"\n📍 Current Session: {session_info['name']}")
        print(f"   Quality: {get_session_quality(session)}/100")
        print(f"   Position Size: {get_session_risk_multiplier(session)*100:.0f}%")
        print(f"   RTH: {'Yes' if is_rth(now) else 'No'}")
        print(f"   Optimal Window: {'✅ Yes' if is_optimal_window(now) else '❌ No'}")
        print(f"   Should Avoid: {'⚠️ Yes' if should_avoid_trading(now) else '✅ No'}")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    print("Testing session times...")
    
    print_session_info()
    print_current_status()
    
    print("✅ Session times configuration complete!")