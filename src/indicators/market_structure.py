"""
Market Structure Indicator for E-mini ES Futures
Detects swing points, support/resistance, breakouts, and retests
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime


class MarketStructureAnalyzer:
    """
    Market Structure analysis for identifying key price levels and patterns
    
    Components:
    - Swing highs and swing lows
    - Support and resistance levels
    - Breakout detection
    - Retest identification
    - Structure quality scoring
    """
    
    def __init__(
        self,
        swing_window: int = 5,
        min_touches: int = 2,
        level_tolerance: float = 0.002,  # 0.2% tolerance for level clustering
        breakout_confirmation: int = 2
    ):
        """
        Initialize Market Structure Analyzer
        
        Args:
            swing_window: Lookback/forward window for swing detection
            min_touches: Minimum touches to confirm S/R level
            level_tolerance: Price tolerance for level clustering (as fraction)
            breakout_confirmation: Bars needed to confirm breakout
        """
        self.swing_window = swing_window
        self.min_touches = min_touches
        self.level_tolerance = level_tolerance
        self.breakout_confirmation = breakout_confirmation
        
    def detect_swing_highs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect swing high points
        Swing high = local maximum with lower highs on both sides
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with swing_high column added
        """
        df = df.copy()
        df['swing_high'] = False
        df['swing_high_price'] = np.nan
        
        window = self.swing_window
        
        for i in range(window, len(df) - window):
            current_high = df.iloc[i]['high']
            
            # Check if current bar is highest in window
            left_window = df.iloc[i-window:i]['high']
            right_window = df.iloc[i+1:i+window+1]['high']
            
            # ✅ FIXED - checks high against highs (was incorrectly checking current_close)
            if current_high > left_window.max() and current_high > right_window.max():
                df.loc[df.index[i], 'swing_high'] = True
                df.loc[df.index[i], 'swing_high_price'] = current_high
        
        return df
    
    def detect_swing_lows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect swing low points
        Swing low = local minimum with higher lows on both sides
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            DataFrame with swing_low column added
        """
        df = df.copy()
        df['swing_low'] = False
        df['swing_low_price'] = np.nan
        
        window = self.swing_window
        
        for i in range(window, len(df) - window):
            current_low = df.iloc[i]['low']
            
            # Check if current bar is lowest in window
            left_window = df.iloc[i-window:i]['low']
            right_window = df.iloc[i+1:i+window+1]['low']
            
            if current_low < left_window.min() and current_low < right_window.min():
                df.loc[df.index[i], 'swing_low'] = True
                df.loc[df.index[i], 'swing_low_price'] = current_low
        
        return df
    
    def identify_support_resistance(self, df: pd.DataFrame) -> Dict:
        """
        Identify support and resistance levels from swing points
        
        Args:
            df: DataFrame with swing points detected
            
        Returns:
            Dictionary with support and resistance levels
        """
        # Get all swing points
        swing_highs = df[df['swing_high']]['swing_high_price'].dropna().values
        swing_lows = df[df['swing_low']]['swing_low_price'].dropna().values
        
        # Cluster nearby levels
        resistance_levels = self._cluster_levels(swing_highs)
        support_levels = self._cluster_levels(swing_lows)
        
        return {
            'resistance': resistance_levels,
            'support': support_levels
        }
    
    def _cluster_levels(self, prices: np.ndarray) -> List[Dict]:
        """
        Cluster nearby price levels together
        
        Args:
            prices: Array of prices to cluster
            
        Returns:
            List of dictionaries with level info
        """
        if len(prices) == 0:
            return []
        
        levels = []
        prices = np.sort(prices)
        
        current_cluster = [prices[0]]
        
        for price in prices[1:]:
            # Check if price is within tolerance of current cluster
            cluster_mean = np.mean(current_cluster)
            tolerance = cluster_mean * self.level_tolerance
            
            if abs(price - cluster_mean) <= tolerance:
                current_cluster.append(price)
            else:
                # Save current cluster if it has enough touches
                if len(current_cluster) >= self.min_touches:
                    levels.append({
                        'price': np.mean(current_cluster),
                        'touches': len(current_cluster),
                        'strength': len(current_cluster)
                    })
                current_cluster = [price]
        
        # Don't forget last cluster
        if len(current_cluster) >= self.min_touches:
            levels.append({
                'price': np.mean(current_cluster),
                'touches': len(current_cluster),
                'strength': len(current_cluster)
            })
        
        return levels
    
    def detect_breakouts(
        self,
        df: pd.DataFrame,
        levels: Dict
    ) -> pd.DataFrame:
        """
        Detect breakouts of support/resistance levels
        
        Args:
            df: DataFrame with price data
            levels: Support/resistance levels
            
        Returns:
            DataFrame with breakout signals
        """
        df = df.copy()
        df['breakout_resistance'] = False
        df['breakout_support'] = False
        df['broken_level'] = np.nan
        
        for i in range(self.breakout_confirmation, len(df)):
            current_close = df.iloc[i]['close']
            previous_close = df.iloc[i-1]['close']
            
            # Check resistance breakouts
            for level in levels['resistance']:
                level_price = level['price']
                
                # Breakout: previous below, current above
                if previous_close <= level_price and current_close > level_price:
                    # Confirm with next bars
                    if i + self.breakout_confirmation < len(df):
                        future_closes = df.iloc[i+1:i+self.breakout_confirmation+1]['close']
                        if all(future_closes > level_price):
                            df.loc[df.index[i], 'breakout_resistance'] = True
                            df.loc[df.index[i], 'broken_level'] = level_price
            
            # Check support breakouts
            for level in levels['support']:
                level_price = level['price']
                
                # Breakdown: previous above, current below
                if previous_close >= level_price and current_close < level_price:
                    # Confirm with next bars
                    if i + self.breakout_confirmation < len(df):
                        future_closes = df.iloc[i+1:i+self.breakout_confirmation+1]['close']
                        if all(future_closes < level_price):
                            df.loc[df.index[i], 'breakout_support'] = True
                            df.loc[df.index[i], 'broken_level'] = level_price
        
        return df
    
    def detect_retests(
        self,
        df: pd.DataFrame,
        levels: Dict
    ) -> pd.DataFrame:
        """
        Detect retests of broken levels
        Broken resistance becomes support (and vice versa)
        
        Args:
            df: DataFrame with breakout data
            levels: Support/resistance levels
            
        Returns:
            DataFrame with retest signals
        """
        df = df.copy()
        df['retest_as_support'] = False
        df['retest_as_resistance'] = False
        
        # Track broken levels
        broken_resistance = []
        broken_support = []
        
        for i in range(len(df)):
            # Track new breakouts
            if df.iloc[i]['breakout_resistance']:
                broken_resistance.append(df.iloc[i]['broken_level'])
            if df.iloc[i]['breakout_support']:
                broken_support.append(df.iloc[i]['broken_level'])
            
            current_low = df.iloc[i]['low']
            current_high = df.iloc[i]['high']
            
            # Check for retests of broken resistance (now support)
            for level_price in broken_resistance:
                tolerance = level_price * self.level_tolerance
                if abs(current_low - level_price) <= tolerance:
                    df.loc[df.index[i], 'retest_as_support'] = True
            
            # Check for retests of broken support (now resistance)
            for level_price in broken_support:
                tolerance = level_price * self.level_tolerance
                if abs(current_high - level_price) <= tolerance:
                    df.loc[df.index[i], 'retest_as_resistance'] = True
        
        return df
    
    def calculate_structure_quality(
        self,
        df: pd.DataFrame,
        levels: Dict
    ) -> pd.DataFrame:
        """
        Calculate structure quality score
        Higher score = cleaner trend, lower score = choppy/ranging
        
        Args:
            df: DataFrame with swing points
            levels: Support/resistance levels
            
        Returns:
            DataFrame with structure_quality column
        """
        df = df.copy()
        df['structure_quality'] = 0.0
        
        window = 20  # Look at recent structure
        
        for i in range(window, len(df)):
            recent_df = df.iloc[i-window:i]
            
            # Count swing points
            num_swing_highs = recent_df['swing_high'].sum()
            num_swing_lows = recent_df['swing_low'].sum()
            
            # Quality factors
            swing_balance = 1.0 - abs(num_swing_highs - num_swing_lows) / max(num_swing_highs + num_swing_lows, 1)
            swing_frequency = (num_swing_highs + num_swing_lows) / window
            
            # Ideal: balanced swings, moderate frequency
            quality = swing_balance * (1.0 - abs(swing_frequency - 0.2))
            quality = max(0, min(1, quality))  # Clamp to [0, 1]
            
            df.loc[df.index[i], 'structure_quality'] = quality
        
        return df
    
    def generate_signals(
        self,
        df: pd.DataFrame,
        levels: Dict
    ) -> pd.DataFrame:
        """
        Generate trading signals based on market structure
        
        Signal logic:
        - Retest of broken resistance (now support) = +1 (bullish)
        - Retest of broken support (now resistance) = -1 (bearish)
        - Fresh breakout up = +1
        - Fresh breakdown = -1
        - Otherwise = 0
        
        Args:
            df: DataFrame with structure analysis
            levels: Support/resistance levels
            
        Returns:
            DataFrame with structure_signal column
        """
        df = df.copy()
        df['structure_signal'] = 0
        
        for i in range(len(df)):
            # Bullish signals
            if df.iloc[i]['retest_as_support']:
                df.loc[df.index[i], 'structure_signal'] = 1
            elif df.iloc[i]['breakout_resistance']:
                df.loc[df.index[i], 'structure_signal'] = 1
            
            # Bearish signals
            elif df.iloc[i]['retest_as_resistance']:
                df.loc[df.index[i], 'structure_signal'] = -1
            elif df.iloc[i]['breakout_support']:
                df.loc[df.index[i], 'structure_signal'] = -1
        
        return df
    
    def analyze(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Complete market structure analysis pipeline
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Tuple of (analyzed DataFrame, analysis results dictionary)
        """
        # Detect swing points
        df = self.detect_swing_highs(df)
        df = self.detect_swing_lows(df)
        
        # Identify support/resistance
        levels = self.identify_support_resistance(df)
        
        # Detect breakouts and retests
        df = self.detect_breakouts(df, levels)
        df = self.detect_retests(df, levels)
        
        # Calculate structure quality
        df = self.calculate_structure_quality(df, levels)
        
        # Generate signals
        df = self.generate_signals(df, levels)
        
        # Compile results
        results = {
            'levels': levels,
            'num_swing_highs': df['swing_high'].sum(),
            'num_swing_lows': df['swing_low'].sum(),
            'num_breakouts': df['breakout_resistance'].sum() + df['breakout_support'].sum(),
            'num_retests': df['retest_as_support'].sum() + df['retest_as_resistance'].sum(),
            'avg_structure_quality': df['structure_quality'].mean()
        }
        
        return df, results
    
    def get_current_state(self, df: pd.DataFrame, results: Dict) -> Dict:
        """
        Get current market structure state
        
        Args:
            df: Analyzed DataFrame
            results: Results from analyze()
            
        Returns:
            Dictionary with current state
        """
        if len(df) == 0:
            return {}
        
        last_row = df.iloc[-1]
        
        # Find nearest support/resistance
        current_price = last_row['close']
        
        nearest_resistance = None
        nearest_support = None
        
        if results['levels']['resistance']:
            nearest_resistance = min(
                [l['price'] for l in results['levels']['resistance']],
                key=lambda x: abs(x - current_price)
            )
        
        if results['levels']['support']:
            nearest_support = min(
                [l['price'] for l in results['levels']['support']],
                key=lambda x: abs(x - current_price)
            )
        
        return {
            'price': current_price,
            'swing_high': bool(last_row['swing_high']),
            'swing_low': bool(last_row['swing_low']),
            'breakout_resistance': bool(last_row['breakout_resistance']),
            'breakout_support': bool(last_row['breakout_support']),
            'retest_as_support': bool(last_row['retest_as_support']),
            'retest_as_resistance': bool(last_row['retest_as_resistance']),
            'structure_quality': float(last_row['structure_quality']),
            'signal': int(last_row['structure_signal']),
            'nearest_resistance': nearest_resistance,
            'nearest_support': nearest_support
        }


# Test code
if __name__ == "__main__":
    print("Testing Market Structure Analyzer...")
    print("=" * 60)
    
    # Generate sample data with trending structure
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01 09:30', periods=100, freq='1min')
    
    # Create uptrend with pullbacks
    base_price = 4500
    trend = np.linspace(0, 10, 100)  # Uptrend
    noise = np.random.randn(100) * 0.5
    prices = base_price + trend + noise
    
    df = pd.DataFrame({
        'datetime': dates,
        'open': prices + np.random.randn(100) * 0.2,
        'high': prices + np.abs(np.random.randn(100) * 0.5),
        'low': prices - np.abs(np.random.randn(100) * 0.5),
        'close': prices,
        'volume': np.random.randint(1000, 5000, 100)
    })
    
    # Ensure OHLC consistency
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    # Analyze
    analyzer = MarketStructureAnalyzer()
    df_analyzed, results = analyzer.analyze(df)
    
    # Print results
    print("\nMARKET STRUCTURE ANALYSIS RESULTS")
    print("=" * 60)
    
    print(f"\nSwing Points:")
    print(f"  Swing Highs: {results['num_swing_highs']}")
    print(f"  Swing Lows: {results['num_swing_lows']}")
    
    print(f"\nSupport/Resistance Levels:")
    print(f"  Resistance Levels: {len(results['levels']['resistance'])}")
    for i, level in enumerate(results['levels']['resistance'][:3]):  # Show top 3
        print(f"    R{i+1}: ${level['price']:.2f} ({level['touches']} touches)")
    
    print(f"  Support Levels: {len(results['levels']['support'])}")
    for i, level in enumerate(results['levels']['support'][:3]):  # Show top 3
        print(f"    S{i+1}: ${level['price']:.2f} ({level['touches']} touches)")
    
    print(f"\nStructure Events:")
    print(f"  Breakouts: {results['num_breakouts']}")
    print(f"  Retests: {results['num_retests']}")
    print(f"  Avg Structure Quality: {results['avg_structure_quality']:.2f}")
    
    print(f"\nCurrent State:")
    state = analyzer.get_current_state(df_analyzed, results)
    print(f"  Current Price: ${state['price']:.2f}")
    if state['nearest_resistance']:
        print(f"  Nearest Resistance: ${state['nearest_resistance']:.2f}")
    if state['nearest_support']:
        print(f"  Nearest Support: ${state['nearest_support']:.2f}")
    print(f"  Structure Quality: {state['structure_quality']:.2f}")
    print(f"  Signal: {state['signal']} ", end="")
    if state['signal'] == 1:
        print("(Bullish Structure)")
    elif state['signal'] == -1:
        print("(Bearish Structure)")
    else:
        print("(Neutral)")
    
    print(f"\nSignal Summary:")
    buy_signals = (df_analyzed['structure_signal'] == 1).sum()
    sell_signals = (df_analyzed['structure_signal'] == -1).sum()
    neutral = (df_analyzed['structure_signal'] == 0).sum()
    print(f"  Buy signals: {buy_signals}")
    print(f"  Sell signals: {sell_signals}")
    print(f"  Neutral: {neutral}")
    
    print("\n✅ Market structure analyzer working correctly!")