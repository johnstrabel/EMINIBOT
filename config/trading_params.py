"""
Trading Parameters for EMINIBOT
Fine-tuning parameters for each indicator
"""


# ============================================================================
# ORDER FLOW PARAMETERS
# ============================================================================

ORDER_FLOW = {
    # Delta thresholds
    'delta_threshold': 1.5,  # Buy/sell ratio threshold (1.5:1)
    'strong_delta_threshold': 2.0,  # Strong imbalance (2:1)
    
    # CVD (Cumulative Volume Delta)
    'cvd_lookback': 20,  # Bars to look back for CVD calculation
    'cvd_trend_threshold': 500,  # Min CVD change for trend signal
    
    # Absorption detection
    'absorption_threshold': 500,  # Min contracts for absorption
    'absorption_volume_multiplier': 1.5,  # Volume must be 1.5x average
    'absorption_price_tolerance': 0.5,  # Max price move during absorption
    
    # Imbalance detection
    'imbalance_window': 3,  # Consecutive bars for imbalance
    'min_imbalance_bars': 2,  # Min bars to confirm
    
    # Signal weighting
    'weight': 0.40,  # 40% weight in combined signal
    
    # Signal strength multipliers
    'strong_buy_multiplier': 1.5,  # Boost strong buy signals
    'strong_sell_multiplier': 1.5,  # Boost strong sell signals
}


# ============================================================================
# VOLUME PROFILE PARAMETERS
# ============================================================================

VOLUME_PROFILE = {
    # Value area
    'value_area_pct': 0.70,  # 70% of volume defines value area
    'value_area_extension': 0.10,  # 10% tolerance for "near" value area
    
    # Volume profile bins
    'num_bins': 50,  # Number of price bins for profile
    'min_touches': 2,  # Min touches to confirm S/R level
    
    # VWAP
    'vwap_std_devs': [1, 2, 3],  # Standard deviation bands
    'vwap_weight': 0.5,  # Weight of VWAP in volume profile signal
    
    # Volume nodes
    'hvn_percentile': 80,  # High Volume Node threshold (80th percentile)
    'lvn_percentile': 20,  # Low Volume Node threshold (20th percentile)
    'node_tolerance': 0.002,  # 0.2% price tolerance for node detection
    
    # POC (Point of Control)
    'poc_magnet_strength': 2.0,  # How strongly POC attracts price
    'poc_tolerance': 0.001,  # 0.1% tolerance for "at POC"
    
    # Signal weighting
    'weight': 0.30,  # 30% weight in combined signal
    
    # Signal conditions
    'above_vah_bullish': True,  # Price above VAH = bullish
    'below_val_bearish': True,  # Price below VAL = bearish
    'poc_neutral': True,  # At POC = neutral (wait for direction)
}


# ============================================================================
# MARKET STRUCTURE PARAMETERS
# ============================================================================

MARKET_STRUCTURE = {
    # Swing point detection
    'swing_window': 5,  # Lookback/forward bars for swing points
    'min_swing_strength': 3,  # Min bars on each side for valid swing
    
    # Support/Resistance
    'level_tolerance': 0.002,  # 0.2% tolerance for level clustering
    'min_touches': 2,  # Min touches to confirm S/R level
    'level_strength_decay': 0.9,  # Decay factor for old levels
    
    # Breakout detection
    'breakout_confirmation': 2,  # Bars to confirm breakout
    'breakout_volume_multiplier': 1.3,  # Volume must be 1.3x average
    'false_breakout_threshold': 0.003,  # 0.3% max retrace for valid breakout
    
    # Retest detection
    'retest_tolerance': 0.002,  # 0.2% tolerance for retest
    'retest_confirmation': 1,  # Bars to confirm retest hold
    
    # Structure quality
    'quality_window': 20,  # Bars to assess structure quality
    'min_quality_score': 0.5,  # Min quality to trust structure (0-1)
    
    # Signal weighting
    'weight': 0.30,  # 30% weight in combined signal
    
    # Signal multipliers
    'breakout_multiplier': 1.3,  # Boost breakout signals
    'retest_multiplier': 1.5,  # Boost retest signals (higher confidence)
}


# ============================================================================
# SESSION PARAMETERS
# ============================================================================

SESSION = {
    # Session preferences
    'trade_overnight': True,   # ✅ CHANGED: Now trading overnight (was False)
    'trade_rth': True,  # Trade Regular Trading Hours
    'trade_opening': True,  # Trade opening session (9:30-10:30 AM)
    'trade_midday': True,  # Trade midday session (10:30 AM-2 PM)
    'trade_closing': True,  # Trade closing session (2-4 PM)
    'trade_after_hours': False,  # Don't trade after hours
    
    # Risk multipliers by session
    'session_risk_multipliers': {
        'overnight': 0.5,    # Half size (safer during low liquidity)
        'opening': 0.75,     # 75% size
        'midday': 1.0,       # Full size (best conditions)
        'closing': 0.85,     # 85% size
        'after_hours': 0.0,  # No trading
    },
    
    # Signal adjustments by session
    'session_signal_adjustments': {
        'overnight': 0.5,    # Reduce signal confidence
        'opening': 0.7,      # Lower confidence (choppy)
        'midday': 1.0,       # Full confidence
        'closing': 0.8,      # Good confidence
        'after_hours': 0.0,  # No trading
    },
    
    # Session transition handling
    'pause_on_transition': True,  # Don't trade during session changes
    'transition_pause_bars': 3,   # Wait 3 bars after transition
    
    # Session bias
    'use_session_bias': True,     # Use historical session performance
    'bias_lookback': 5,           # Look back 5 sessions
    'bias_weight': 0.2,           # 20% weight to bias
}


# ============================================================================
# SIGNAL COMBINATION PARAMETERS
# ============================================================================

SIGNAL_COMBINATION = {
    # Alignment requirements
    'min_indicators_aligned': 2,  # Need 2 of 3 indicators to agree
    'unanimous_boost': 1.3,       # Boost signal if all 3 agree
    
    # Signal strength calculation
    'base_strength': 50,          # Base signal strength
    'max_strength': 100,          # Maximum signal strength
    'min_strength': 0,            # Minimum signal strength
    
    # Confirmation requirements
    'require_order_flow': True,   # Order flow must confirm
    'require_volume_profile': False,  # VP confirmation optional
    'require_structure': False,   # Structure confirmation optional
    
    # Conflicting signals
    'conflict_threshold': 0.3,    # Max disagreement allowed (30%)
    'skip_conflicted': True,      # Skip trades with conflicted signals
    
    # Signal filtering
    'min_signal_strength': 40,    # ✅ CHANGED: Only take signals > 40/100 (was 50)
    'apply_session_filter': True, # Apply session-based filtering
    'apply_quality_filter': True, # Apply structure quality filter
}


# ============================================================================
# ENTRY RULES
# ============================================================================

ENTRY = {
    # Entry conditions
    'wait_for_confirmation': True,   # Wait 1 bar to confirm signal
    'require_price_action': True,    # Price must be moving in signal direction
    
    # Entry timing
    'max_bars_to_enter': 3,          # Enter within 3 bars or skip
    'avoid_session_transition': True, # Don't enter during transitions
    
    # Entry validation
    'check_spread': True,             # Verify spread is reasonable
    'max_spread_ticks': 2,            # Max 2 tick spread
    'check_volume': True,             # Verify adequate volume
    'min_volume_multiplier': 0.5,     # Volume must be 50%+ of average
    
    # Risk checks
    'validate_risk_reward': True,     # Check R:R before entry
    'min_risk_reward': 1.5,           # Minimum 1.5:1 R:R
    
    # Position management
    'scale_in': False,                # Don't scale into positions (yet)
    'max_entries_per_signal': 1,      # One entry per signal
}


# ============================================================================
# EXIT RULES
# ============================================================================

EXIT = {
    # Stop loss
    'use_hard_stop': True,            # Always use hard stop loss
    'move_to_breakeven': True,        # Move stop to breakeven after profit
    'breakeven_trigger_ticks': 8,     # Move to BE after 8 ticks profit
    'breakeven_offset_ticks': 1,      # Lock in 1 tick profit at BE
    
    # Take profit
    'use_multiple_targets': True,     # Use TP1 and TP2
    'tp1_percentage': 0.5,            # Close 50% at TP1
    'tp2_percentage': 0.5,            # Close 50% at TP2
    
    # Trailing stop
    'use_trailing': True,             # Use trailing stop
    'trail_activation_ticks': 8,      # Start trailing at 8 ticks profit
    'trail_offset_ticks': 4,          # Trail 4 ticks behind
    'trail_step_ticks': 2,            # Move trail in 2 tick increments
    
    # Time-based exits
    'max_hold_time_minutes': 180,     # Max 3 hours per trade
    'exit_before_close': True,        # Close all before market close
    'close_before_time': '15:45',     # Close by 3:45 PM (15 min buffer)
    
    # Exit on opposing signal
    'exit_on_opposite': True,         # Exit if opposite signal appears
    'opposite_strength_threshold': 70, # Only if opposing signal is strong
    
    # Emergency exits
    'exit_on_news': True,             # Exit if major news event
    'exit_on_halt': True,             # Exit if trading halt
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_indicator_weights():
    """Get indicator weights as dictionary"""
    return {
        'order_flow': ORDER_FLOW['weight'],
        'volume_profile': VOLUME_PROFILE['weight'],
        'market_structure': MARKET_STRUCTURE['weight'],
    }


def validate_weights():
    """Ensure indicator weights sum to 1.0"""
    weights = get_indicator_weights()
    total = sum(weights.values())
    
    if abs(total - 1.0) > 0.01:
        print(f"⚠️ Warning: Indicator weights sum to {total:.2f}, should be 1.0")
        return False
    return True


def print_parameters():
    """Print key parameters"""
    print("\n" + "="*70)
    print("TRADING PARAMETERS")
    print("="*70)
    
    print("\n📊 INDICATOR WEIGHTS:")
    weights = get_indicator_weights()
    for indicator, weight in weights.items():
        print(f"  {indicator}: {weight*100:.0f}%")
    
    print("\n📈 ORDER FLOW:")
    print(f"  Delta threshold: {ORDER_FLOW['delta_threshold']}")
    print(f"  CVD lookback: {ORDER_FLOW['cvd_lookback']} bars")
    print(f"  Absorption threshold: {ORDER_FLOW['absorption_threshold']} contracts")
    
    print("\n📊 VOLUME PROFILE:")
    print(f"  Value area: {VOLUME_PROFILE['value_area_pct']*100:.0f}%")
    print(f"  Price bins: {VOLUME_PROFILE['num_bins']}")
    print(f"  VWAP std devs: {VOLUME_PROFILE['vwap_std_devs']}")
    
    print("\n🏗️ MARKET STRUCTURE:")
    print(f"  Swing window: {MARKET_STRUCTURE['swing_window']} bars")
    print(f"  Breakout confirmation: {MARKET_STRUCTURE['breakout_confirmation']} bars")
    print(f"  Min quality score: {MARKET_STRUCTURE['min_quality_score']}")
    
    print("\n⏰ SESSION:")
    print(f"  Trade overnight: {SESSION['trade_overnight']}")
    print(f"  Trade RTH: {SESSION['trade_rth']}")
    print(f"  Pause on transition: {SESSION['pause_on_transition']}")
    
    print("\n🎯 ENTRY:")
    print(f"  Min indicators aligned: {SIGNAL_COMBINATION['min_indicators_aligned']}")
    print(f"  Min signal strength: {SIGNAL_COMBINATION['min_signal_strength']}")
    print(f"  Min R:R ratio: {ENTRY['min_risk_reward']}")
    
    print("\n🚪 EXIT:")
    print(f"  Multiple targets: {EXIT['use_multiple_targets']}")
    print(f"  Trailing stop: {EXIT['use_trailing']}")
    print(f"  Max hold time: {EXIT['max_hold_time_minutes']} minutes")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    print("Testing trading parameters...")
    
    if validate_weights():
        print("✅ Indicator weights validated!")
    else:
        print("❌ Indicator weight validation failed!")
    
    print_parameters()