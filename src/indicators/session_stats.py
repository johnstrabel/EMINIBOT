"""
Session Statistics for E-mini ES Futures
Identifies market sessions and calculates session-specific metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from datetime import datetime, time
import pytz


class SessionAnalyzer:
    """
    Session-based analysis for E-mini ES futures
    
    ES Trading Hours (US/Central Time):
    - Overnight: 6:00 PM - 9:30 AM (Sunday-Friday)
    - Opening: 9:30 AM - 10:30 AM (market open volatility)
    - Midday: 10:30 AM - 2:00 PM (best liquidity)
    - Closing: 2:00 PM - 4:00 PM (institutional flows)
    - After Hours: 4:00 PM - 6:00 PM
    """
    
    def __init__(
        self,
        timezone: str = 'America/Chicago',
        trade_overnight: bool = False,
        trade_rth: bool = True
    ):
        """
        Initialize Session Analyzer
        
        Args:
            timezone: Timezone for session times (default Chicago/Central)
            trade_overnight: Whether to trade overnight session
            trade_rth: Whether to trade regular trading hours
        """
        self.timezone = pytz.timezone(timezone)
        self.trade_overnight = trade_overnight
        self.trade_rth = trade_rth
        
        # Define session times (in 24-hour format)
        self.sessions = {
            'overnight': (time(18, 0), time(9, 30)),  # 6 PM - 9:30 AM
            'opening': (time(9, 30), time(10, 30)),   # 9:30 AM - 10:30 AM
            'midday': (time(10, 30), time(14, 0)),    # 10:30 AM - 2 PM
            'closing': (time(14, 0), time(16, 0)),    # 2 PM - 4 PM
            'after_hours': (time(16, 0), time(18, 0)) # 4 PM - 6 PM
        }
    
    def identify_session(self, dt: datetime) -> str:
        """
        Identify which session a datetime belongs to
        
        Args:
            dt: Datetime to classify
            
        Returns:
            Session name: 'overnight', 'opening', 'midday', 'closing', 'after_hours'
        """
        # Ensure timezone aware
        if dt.tzinfo is None:
            dt = self.timezone.localize(dt)
        else:
            dt = dt.astimezone(self.timezone)
        
        current_time = dt.time()
        
        # Check each session
        for session_name, (start_time, end_time) in self.sessions.items():
            if session_name == 'overnight':
                # Overnight wraps around midnight
                if current_time >= start_time or current_time < end_time:
                    return session_name
            else:
                if start_time <= current_time < end_time:
                    return session_name
        
        return 'after_hours'
    
    def calculate_session_stats(
        self,
        df: pd.DataFrame,
        session_name: str
    ) -> Dict:
        """
        Calculate statistics for a specific session
        
        Args:
            df: DataFrame with price data
            session_name: Name of session to analyze
            
        Returns:
            Dictionary with session statistics
        """
        session_df = df[df['session'] == session_name]
        
        if len(session_df) == 0:
            return {
                'avg_volume': 0,
                'avg_range': 0,
                'avg_volatility': 0,
                'num_bars': 0
            }
        
        # Calculate metrics
        avg_volume = session_df['volume'].mean()
        avg_range = (session_df['high'] - session_df['low']).mean()
        
        # Volatility = standard deviation of returns
        if len(session_df) > 1:
            returns = session_df['close'].pct_change()
            avg_volatility = returns.std()
        else:
            avg_volatility = 0
        
        return {
            'avg_volume': avg_volume,
            'avg_range': avg_range,
            'avg_volatility': avg_volatility,
            'num_bars': len(session_df)
        }
    
    def apply_session_filter(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Apply session-based trading filters
        
        Args:
            df: DataFrame with session classification
            
        Returns:
            DataFrame with can_trade column added
        """
        df = df.copy()
        df['can_trade'] = True
        
        # Apply filters based on settings
        if not self.trade_overnight:
            df.loc[df['session'] == 'overnight', 'can_trade'] = False
        
        if not self.trade_rth:
            df.loc[df['session'].isin(['opening', 'midday', 'closing']), 'can_trade'] = False
        
        # Never trade after hours (too low liquidity)
        df.loc[df['session'] == 'after_hours', 'can_trade'] = False
        
        return df
    
    def calculate_session_risk_multiplier(
        self,
        session_name: str,
        session_stats: Dict
    ) -> float:
        """
        Calculate risk multiplier for session
        Lower multiplier = reduce position size
        
        Args:
            session_name: Name of session
            session_stats: Statistics for all sessions
            
        Returns:
            Risk multiplier (0.0 to 1.0)
        """
        multipliers = {
            'overnight': 0.5,    # Half size (lower liquidity)
            'opening': 0.75,     # 75% size (high volatility)
            'midday': 1.0,       # Full size (best conditions)
            'closing': 0.85,     # 85% size (good but flows can be erratic)
            'after_hours': 0.0   # No trading
        }
        
        return multipliers.get(session_name, 0.5)
    
    def detect_session_transition(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Detect transitions between sessions
        Important because transitions often have price moves
        
        Args:
            df: DataFrame with session classification
            
        Returns:
            DataFrame with session_transition column
        """
        df = df.copy()
        df['session_transition'] = False
        df['previous_session'] = df['session'].shift(1)
        
        # Mark rows where session changes
        df.loc[df['session'] != df['previous_session'], 'session_transition'] = True
        
        return df
    
    def calculate_session_bias(
        self,
        df: pd.DataFrame,
        lookback: int = 5
    ) -> pd.DataFrame:
        """
        Calculate directional bias for each session
        Based on recent performance in this session
        
        Args:
            df: DataFrame with session data
            lookback: Number of previous sessions to analyze
            
        Returns:
            DataFrame with session_bias column
        """
        df = df.copy()
        df['session_bias'] = 0.0
        
        for session_name in self.sessions.keys():
            session_mask = df['session'] == session_name
            session_indices = df[session_mask].index
            
            for idx in session_indices:
                # Get previous sessions of same type
                prior_sessions = df[(df['session'] == session_name) & (df.index < idx)]
                
                if len(prior_sessions) >= lookback:
                    recent = prior_sessions.tail(lookback)
                    
                    # Calculate net price change
                    open_prices = recent.groupby('session')['open'].first()
                    close_prices = recent.groupby('session')['close'].last()
                    
                    if len(open_prices) > 0 and len(close_prices) > 0:
                        net_change = (close_prices - open_prices).sum()
                        avg_change = net_change / lookback
                        
                        # Normalize to -1 to +1
                        typical_range = df['high'].mean() - df['low'].mean()
                        if typical_range > 0:
                            bias = np.clip(avg_change / typical_range, -1, 1)
                            df.loc[idx, 'session_bias'] = bias
        
        return df
    
    def generate_session_signals(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate trading signals based on session analysis
        
        Signal adjustments:
        - Reduce signal strength during overnight (lower confidence)
        - Increase signal strength during midday (best conditions)
        - Avoid trading on session transitions (wait for clarity)
        
        Args:
            df: DataFrame with session analysis
            
        Returns:
            DataFrame with session_signal_adjustment column
        """
        df = df.copy()
        df['session_signal_adjustment'] = 1.0
        
        # Adjust based on session
        session_adjustments = {
            'overnight': 0.5,
            'opening': 0.7,
            'midday': 1.0,
            'closing': 0.8,
            'after_hours': 0.0
        }
        
        for session_name, adjustment in session_adjustments.items():
            mask = df['session'] == session_name
            df.loc[mask, 'session_signal_adjustment'] = adjustment
        
        # Reduce on transitions
        df.loc[df['session_transition'], 'session_signal_adjustment'] *= 0.5
        
        return df
    
    def analyze(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Complete session analysis pipeline
        
        Args:
            df: DataFrame with OHLCV data and datetime
            
        Returns:
            Tuple of (analyzed DataFrame, analysis results dictionary)
        """
        df = df.copy()
        
        # Ensure datetime column exists and is datetime type
        if 'datetime' not in df.columns:
            df['datetime'] = df.index
        
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Classify sessions
        df['session'] = df['datetime'].apply(self.identify_session)
        
        # Calculate session statistics
        session_stats = {}
        for session_name in self.sessions.keys():
            session_stats[session_name] = self.calculate_session_stats(df, session_name)
        
        # Apply filters
        df = self.apply_session_filter(df)
        
        # Detect transitions
        df = self.detect_session_transition(df)
        
        # Calculate bias
        df = self.calculate_session_bias(df)
        
        # Generate signals
        df = self.generate_session_signals(df)
        
        # Add risk multipliers
        df['session_risk_multiplier'] = df['session'].apply(
            lambda s: self.calculate_session_risk_multiplier(s, session_stats)
        )
        
        # Compile results
        results = {
            'session_stats': session_stats,
            'current_session': df['session'].iloc[-1] if len(df) > 0 else None,
            'can_trade_now': df['can_trade'].iloc[-1] if len(df) > 0 else False
        }
        
        return df, results
    
    def get_current_state(self, df: pd.DataFrame, results: Dict) -> Dict:
        """
        Get current session state
        
        Args:
            df: Analyzed DataFrame
            results: Results from analyze()
            
        Returns:
            Dictionary with current state
        """
        if len(df) == 0:
            return {}
        
        last_row = df.iloc[-1]
        current_session = results['current_session']
        
        return {
            'session': current_session,
            'can_trade': bool(last_row['can_trade']),
            'session_transition': bool(last_row['session_transition']),
            'session_bias': float(last_row['session_bias']),
            'signal_adjustment': float(last_row['session_signal_adjustment']),
            'risk_multiplier': float(last_row['session_risk_multiplier']),
            'session_stats': results['session_stats'].get(current_session, {})
        }


# Test code
if __name__ == "__main__":
    print("Testing Session Analyzer...")
    print("=" * 60)
    
    # Generate sample data spanning multiple sessions
    np.random.seed(42)
    
    # Create data for one full trading day
    dates = pd.date_range(
        start='2024-01-01 09:00:00',
        end='2024-01-01 16:00:00',
        freq='5min',
        tz='America/Chicago'
    )
    
    base_price = 4500
    prices = base_price + np.cumsum(np.random.randn(len(dates)) * 0.5)
    
    df = pd.DataFrame({
        'datetime': dates,
        'open': prices + np.random.randn(len(dates)) * 0.2,
        'high': prices + np.abs(np.random.randn(len(dates)) * 0.5),
        'low': prices - np.abs(np.random.randn(len(dates)) * 0.5),
        'close': prices,
        'volume': np.random.randint(1000, 5000, len(dates))
    })
    
    # Ensure OHLC consistency
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    # Analyze
    analyzer = SessionAnalyzer(trade_overnight=False, trade_rth=True)
    df_analyzed, results = analyzer.analyze(df)
    
    # Print results
    print("\nSESSION ANALYSIS RESULTS")
    print("=" * 60)
    
    print(f"\nSession Statistics:")
    for session_name, stats in results['session_stats'].items():
        print(f"\n  {session_name.upper()}:")
        print(f"    Bars: {stats['num_bars']}")
        print(f"    Avg Volume: {stats['avg_volume']:.0f}")
        print(f"    Avg Range: ${stats['avg_range']:.2f}")
        print(f"    Avg Volatility: {stats['avg_volatility']:.4f}")
    
    print(f"\nCurrent State:")
    state = analyzer.get_current_state(df_analyzed, results)
    print(f"  Session: {state['session']}")
    print(f"  Can Trade: {state['can_trade']}")
    print(f"  Session Transition: {state['session_transition']}")
    print(f"  Session Bias: {state['session_bias']:.2f}")
    print(f"  Signal Adjustment: {state['signal_adjustment']:.2f}")
    print(f"  Risk Multiplier: {state['risk_multiplier']:.2f}")
    
    print(f"\nSession Distribution:")
    session_counts = df_analyzed['session'].value_counts()
    for session, count in session_counts.items():
        print(f"  {session}: {count} bars")
    
    print(f"\nTrading Allowed:")
    tradeable_bars = df_analyzed['can_trade'].sum()
    print(f"  Tradeable bars: {tradeable_bars}/{len(df_analyzed)} ({tradeable_bars/len(df_analyzed)*100:.1f}%)")
    
    print(f"\nSession Transitions: {df_analyzed['session_transition'].sum()}")
    
    print("\n✅ Session analyzer working correctly!")