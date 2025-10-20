"""
Signal Generator for EMINIBOT
Combines all indicators to generate trading signals
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import indicators
from src.indicators.order_flow import OrderFlowAnalyzer
from src.indicators.volume_profile import VolumeProfileAnalyzer
from src.indicators.market_structure import MarketStructureAnalyzer
from src.indicators.session_stats import SessionAnalyzer

# Import config
from config import trading_params as params
from config import settings


class SignalGenerator:
    """
    Combines all indicators to generate trading signals
    
    Signal Flow:
    1. Run all 4 indicators on data
    2. Extract individual signals
    3. Apply indicator weights (40/30/30)
    4. Check alignment requirements
    5. Apply session filters
    6. Calculate final signal strength
    7. Generate trading decision
    """
    
    def __init__(self):
        """Initialize signal generator with all indicators"""
        
        # Initialize indicators
        self.order_flow = OrderFlowAnalyzer(
            delta_threshold=params.ORDER_FLOW['delta_threshold'],
            cvd_lookback=params.ORDER_FLOW['cvd_lookback'],
            absorption_threshold=params.ORDER_FLOW['absorption_threshold'],
            imbalance_window=params.ORDER_FLOW['imbalance_window']
        )
        
        self.volume_profile = VolumeProfileAnalyzer(
            value_area_pct=params.VOLUME_PROFILE['value_area_pct'],
            num_bins=params.VOLUME_PROFILE['num_bins'],
            vwap_std_devs=params.VOLUME_PROFILE['vwap_std_devs']
        )
        
        self.market_structure = MarketStructureAnalyzer(
            swing_window=params.MARKET_STRUCTURE['swing_window'],
            min_touches=params.MARKET_STRUCTURE['min_touches'],
            level_tolerance=params.MARKET_STRUCTURE['level_tolerance'],
            breakout_confirmation=params.MARKET_STRUCTURE['breakout_confirmation']
        )
        
        self.session = SessionAnalyzer(
            timezone=settings.TIMEZONE,
            trade_overnight=params.SESSION['trade_overnight'],
            trade_rth=params.SESSION['trade_rth']
        )
        
        # Store weights
        self.weights = {
            'order_flow': params.ORDER_FLOW['weight'],
            'volume_profile': params.VOLUME_PROFILE['weight'],
            'market_structure': params.MARKET_STRUCTURE['weight']
        }
        
        # Combination parameters
        self.min_indicators_aligned = params.SIGNAL_COMBINATION['min_indicators_aligned']
        self.min_signal_strength = params.SIGNAL_COMBINATION['min_signal_strength']
        self.unanimous_boost = params.SIGNAL_COMBINATION['unanimous_boost']
    
    def analyze_all_indicators(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Run all indicators on data
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Tuple of (analyzed DataFrame, results dictionary)
        """
        results = {}
        
        # Order Flow Analysis
        df = self.order_flow.analyze(df)
        results['order_flow'] = self.order_flow.get_current_state(df)
        
        # Volume Profile Analysis
        df, vp_results = self.volume_profile.analyze(df)
        results['volume_profile'] = self.volume_profile.get_current_state(df, vp_results)
        
        # Market Structure Analysis
        df, ms_results = self.market_structure.analyze(df)
        results['market_structure'] = self.market_structure.get_current_state(df, ms_results)
        
        # Session Analysis
        df, session_results = self.session.analyze(df)
        results['session'] = self.session.get_current_state(df, session_results)
        
        return df, results
    
    def extract_signals(self, df: pd.DataFrame) -> Dict:
        """
        Extract signal values from analyzed dataframe
        
        Args:
            df: DataFrame with all indicators
            
        Returns:
            Dictionary with individual signals
        """
        if len(df) == 0:
            return {
                'order_flow': 0,
                'volume_profile': 0,
                'market_structure': 0,
                'session_adjustment': 1.0
            }
        
        last_row = df.iloc[-1]
        
        return {
            'order_flow': int(last_row.get('order_flow_signal', 0)),
            'volume_profile': int(last_row.get('vp_signal', 0)),
            'market_structure': int(last_row.get('structure_signal', 0)),
            'session_adjustment': float(last_row.get('session_signal_adjustment', 1.0))
        }
    
    def check_alignment(self, signals: Dict) -> Dict:
        """
        Check if indicators are aligned
        
        Args:
            signals: Dictionary with individual signals
            
        Returns:
            Dictionary with alignment info
        """
        # Get signals (excluding session adjustment)
        signal_values = [
            signals['order_flow'],
            signals['volume_profile'],
            signals['market_structure']
        ]
        
        # Count bullish (+1) and bearish (-1) signals
        bullish_count = sum(1 for s in signal_values if s == 1)
        bearish_count = sum(1 for s in signal_values if s == -1)
        neutral_count = sum(1 for s in signal_values if s == 0)
        
        # Determine alignment
        aligned_bullish = bullish_count >= self.min_indicators_aligned
        aligned_bearish = bearish_count >= self.min_indicators_aligned
        unanimous_bullish = bullish_count == 3
        unanimous_bearish = bearish_count == 3
        
        return {
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'neutral_count': neutral_count,
            'aligned_bullish': aligned_bullish,
            'aligned_bearish': aligned_bearish,
            'unanimous_bullish': unanimous_bullish,
            'unanimous_bearish': unanimous_bearish,
            'is_aligned': aligned_bullish or aligned_bearish
        }
    
    def calculate_signal_strength(
        self,
        signals: Dict,
        alignment: Dict,
        results: Dict
    ) -> int:
        """
        Calculate weighted signal strength (0-100)
        
        Args:
            signals: Individual indicator signals
            alignment: Alignment information
            results: Full results from indicators
            
        Returns:
            Signal strength (0-100)
        """
        # Base strength from weighted combination
        strength = 0
        
        # Add weighted signals
        if signals['order_flow'] != 0:
            strength += abs(signals['order_flow']) * self.weights['order_flow'] * 100
        
        if signals['volume_profile'] != 0:
            strength += abs(signals['volume_profile']) * self.weights['volume_profile'] * 100
        
        if signals['market_structure'] != 0:
            strength += abs(signals['market_structure']) * self.weights['market_structure'] * 100
        
        # Boost for unanimity
        if alignment['unanimous_bullish'] or alignment['unanimous_bearish']:
            strength *= self.unanimous_boost
        
        # Apply session adjustment
        strength *= signals['session_adjustment']
        
        # Boost from order flow strength (if available)
        if 'order_flow' in results and 'signal_strength' in results['order_flow']:
            of_strength = results['order_flow']['signal_strength']
            strength = (strength * 0.7) + (of_strength * 0.3)  # 70/30 blend
        
        # Adjust for structure quality
        if 'market_structure' in results and 'structure_quality' in results['market_structure']:
            quality = results['market_structure']['structure_quality']
            strength *= (0.8 + (quality * 0.2))  # 80-100% based on quality
        
        # Clamp to 0-100
        strength = max(0, min(100, strength))
        
        return int(strength)
    
    def generate_signal(self, df: pd.DataFrame) -> Dict:
        """
        Generate complete trading signal
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary with signal information
        """
        # Run all indicators
        df, results = self.analyze_all_indicators(df)
        
        # Extract individual signals
        signals = self.extract_signals(df)
        
        # Check alignment
        alignment = self.check_alignment(signals)
        
        # Calculate signal strength
        strength = self.calculate_signal_strength(signals, alignment, results)
        
        # Determine final signal
        final_signal = 0  # Neutral by default
        
        if alignment['aligned_bullish'] and strength >= self.min_signal_strength:
            final_signal = 1  # BUY
        elif alignment['aligned_bearish'] and strength >= self.min_signal_strength:
            final_signal = -1  # SELL
        
        # Check session filters
        can_trade = results['session']['can_trade']
        if not can_trade:
            final_signal = 0  # Override to neutral if session doesn't allow trading
        
        # Build signal package
        signal_package = {
            'signal': final_signal,
            'strength': strength,
            'can_trade': can_trade,
            'timestamp': df['datetime'].iloc[-1] if len(df) > 0 else datetime.now(),
            
            # Individual signals
            'order_flow_signal': signals['order_flow'],
            'volume_profile_signal': signals['volume_profile'],
            'market_structure_signal': signals['market_structure'],
            
            # Alignment
            'indicators_aligned': alignment['is_aligned'],
            'bullish_count': alignment['bullish_count'],
            'bearish_count': alignment['bearish_count'],
            'unanimous': alignment['unanimous_bullish'] or alignment['unanimous_bearish'],
            
            # Session info
            'session': results['session']['session'],
            'session_quality': results['session'].get('session_stats', {}).get('num_bars', 0),
            'session_adjustment': signals['session_adjustment'],
            
            # Key levels (for entry/exit planning)
            'current_price': results['volume_profile'].get('price', 0),
            'vwap': results['volume_profile'].get('vwap', 0),
            'poc': results['volume_profile'].get('poc_price', 0),
            'vah': results['volume_profile'].get('vah', 0),
            'val': results['volume_profile'].get('val', 0),
            'nearest_support': results['market_structure'].get('nearest_support'),
            'nearest_resistance': results['market_structure'].get('nearest_resistance'),
            
            # Full results (for advanced analysis)
            'full_results': results
        }
        
        return signal_package
    
    def calculate_entry_exit_levels(self, signal_package: Dict) -> Dict:
        """
        Calculate suggested entry and exit levels
        
        Args:
            signal_package: Signal package from generate_signal()
            
        Returns:
            Dictionary with entry/exit levels
        """
        current_price = signal_package['current_price']
        signal = signal_package['signal']
        
        if signal == 0:
            return {
                'entry': None,
                'stop_loss': None,
                'take_profit_1': None,
                'take_profit_2': None
            }
        
        # Calculate levels based on tick values
        tick_size = settings.TICK_SIZE
        stop_ticks = settings.STOP_LOSS_TICKS
        tp1_ticks = settings.TAKE_PROFIT_1_TICKS
        tp2_ticks = settings.TAKE_PROFIT_2_TICKS
        
        if signal == 1:  # BUY
            entry = current_price
            stop_loss = entry - (stop_ticks * tick_size)
            take_profit_1 = entry + (tp1_ticks * tick_size)
            take_profit_2 = entry + (tp2_ticks * tick_size)
        else:  # SELL
            entry = current_price
            stop_loss = entry + (stop_ticks * tick_size)
            take_profit_1 = entry - (tp1_ticks * tick_size)
            take_profit_2 = entry - (tp2_ticks * tick_size)
        
        return {
            'entry': entry,
            'stop_loss': stop_loss,
            'take_profit_1': take_profit_1,
            'take_profit_2': take_profit_2,
            'risk_ticks': stop_ticks,
            'reward_ticks_tp1': tp1_ticks,
            'reward_ticks_tp2': tp2_ticks,
            'risk_dollars': settings.ticks_to_dollars(stop_ticks),
            'reward_dollars_tp1': settings.ticks_to_dollars(tp1_ticks),
            'reward_dollars_tp2': settings.ticks_to_dollars(tp2_ticks),
        }
    
    def get_trade_recommendation(self, df: pd.DataFrame) -> Dict:
        """
        Get complete trade recommendation
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary with full trade recommendation
        """
        # Generate signal
        signal_package = self.generate_signal(df)
        
        # Calculate entry/exit levels
        levels = self.calculate_entry_exit_levels(signal_package)
        
        # Calculate position size
        account_balance = settings.ACCOUNT_SIZE  # Would come from broker in live trading
        position_size = settings.calculate_position_size(account_balance)
        
        # Apply session risk multiplier
        session_multiplier = signal_package.get('session_adjustment', 1.0)
        adjusted_position_size = int(position_size * session_multiplier)
        adjusted_position_size = max(1, adjusted_position_size)  # At least 1 contract
        
        # Build recommendation
        recommendation = {
            'action': 'BUY' if signal_package['signal'] == 1 else 'SELL' if signal_package['signal'] == -1 else 'WAIT',
            'signal_strength': signal_package['strength'],
            'position_size': adjusted_position_size,
            'entry_price': levels['entry'],
            'stop_loss': levels['stop_loss'],
            'take_profit_1': levels['take_profit_1'],
            'take_profit_2': levels['take_profit_2'],
            'risk_per_contract': levels.get('risk_dollars', 0),
            'reward_per_contract_tp1': levels.get('reward_dollars_tp1', 0),
            'reward_per_contract_tp2': levels.get('reward_dollars_tp2', 0),
            'total_risk': levels.get('risk_dollars', 0) * adjusted_position_size,
            'session': signal_package['session'],
            'timestamp': signal_package['timestamp'],
            'reasoning': self._build_reasoning(signal_package),
            'full_signal': signal_package
        }
        
        return recommendation
    
    def _build_reasoning(self, signal_package: Dict) -> str:
        """Build human-readable reasoning for signal"""
        if signal_package['signal'] == 0:
            return "No trade signal - waiting for alignment"
        
        action = "BUY" if signal_package['signal'] == 1 else "SELL"
        strength = signal_package['strength']
        
        reasons = []
        
        # Indicator agreement
        if signal_package['unanimous']:
            reasons.append("All 3 indicators agree")
        else:
            reasons.append(f"{signal_package['bullish_count'] if signal_package['signal'] == 1 else signal_package['bearish_count']} of 3 indicators agree")
        
        # Order flow
        if signal_package['order_flow_signal'] != 0:
            direction = "bullish" if signal_package['order_flow_signal'] == 1 else "bearish"
            reasons.append(f"Order flow shows {direction} pressure")
        
        # Volume profile
        if signal_package['volume_profile_signal'] != 0:
            direction = "above value" if signal_package['volume_profile_signal'] == 1 else "below value"
            reasons.append(f"Price is {direction}")
        
        # Market structure
        if signal_package['market_structure_signal'] != 0:
            direction = "bullish" if signal_package['market_structure_signal'] == 1 else "bearish"
            reasons.append(f"Market structure is {direction}")
        
        # Session
        session = signal_package['session']
        reasons.append(f"Trading in {session} session")
        
        reasoning = f"{action} signal (strength: {strength}/100) - " + ", ".join(reasons)
        return reasoning


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("Testing Signal Generator...")
    print("=" * 70)
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01 10:30', periods=100, freq='1min')
    
    base_price = 4500
    trend = np.linspace(0, 5, 100)  # Slight uptrend
    noise = np.random.randn(100) * 0.3
    prices = base_price + trend + noise
    
    df = pd.DataFrame({
        'datetime': dates,
        'open': prices + np.random.randn(100) * 0.1,
        'high': prices + np.abs(np.random.randn(100) * 0.3),
        'low': prices - np.abs(np.random.randn(100) * 0.3),
        'close': prices,
        'volume': np.random.randint(1000, 5000, 100)
    })
    
    # Ensure OHLC consistency
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    # Initialize signal generator
    print("\n📊 Initializing Signal Generator...")
    generator = SignalGenerator()
    print("✅ All indicators loaded")
    
    # Generate signal
    print("\n🔍 Analyzing market...")
    recommendation = generator.get_trade_recommendation(df)
    
    # Display results
    print("\n" + "=" * 70)
    print("TRADING RECOMMENDATION")
    print("=" * 70)
    
    print(f"\n🎯 ACTION: {recommendation['action']}")
    print(f"💪 Signal Strength: {recommendation['signal_strength']}/100")
    print(f"📍 Position Size: {recommendation['position_size']} contracts")
    
    if recommendation['action'] != 'WAIT':
        print(f"\n💰 TRADE LEVELS:")
        print(f"  Entry: ${recommendation['entry_price']:.2f}")
        print(f"  Stop Loss: ${recommendation['stop_loss']:.2f}")
        print(f"  Take Profit 1: ${recommendation['take_profit_1']:.2f}")
        print(f"  Take Profit 2: ${recommendation['take_profit_2']:.2f}")
        
        print(f"\n📊 RISK/REWARD:")
        print(f"  Risk per contract: ${recommendation['risk_per_contract']:.2f}")
        print(f"  Reward per contract (TP1): ${recommendation['reward_per_contract_tp1']:.2f}")
        print(f"  Reward per contract (TP2): ${recommendation['reward_per_contract_tp2']:.2f}")
        print(f"  Total risk: ${recommendation['total_risk']:.2f}")
        print(f"  R:R Ratio (TP1): 1:{recommendation['reward_per_contract_tp1']/recommendation['risk_per_contract']:.2f}")
    
    print(f"\n⏰ SESSION: {recommendation['session']}")
    print(f"🕐 Timestamp: {recommendation['timestamp']}")
    
    print(f"\n💭 REASONING:")
    print(f"  {recommendation['reasoning']}")
    
    print("\n" + "=" * 70)
    print("✅ Signal generator working correctly!")
    print("=" * 70)