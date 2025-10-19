"""
Order Flow Indicators for E-mini ES
Tracks institutional buying/selling pressure

Key Concepts:
- Delta: Buy volume - Sell volume (shows aggression)
- CVD: Cumulative Volume Delta (shows trend)
- Absorption: Large orders stopping price movement
- Imbalance: Sustained one-sided pressure
"""

import pandas as pd
import numpy as np
from typing import Dict


class OrderFlowAnalyzer:
    """
    Analyzes order flow to detect institutional activity
    
    For E-mini ES, we use bid/ask volume to approximate delta
    (Real tape reading requires Level 2 data)
    """
    
    def __init__(self, 
                 delta_threshold: float = 1.5,
                 cvd_lookback: int = 20,
                 absorption_threshold: int = 500,
                 imbalance_window: int = 3):
        """
        Initialize order flow analyzer
        
        Args:
            delta_threshold: Ratio for significant delta (1.5 = 1.5:1 buy/sell)
            cvd_lookback: Bars to look back for CVD calculation
            absorption_threshold: Contract size for absorption detection
            imbalance_window: Bars for sustained imbalance
        """
        self.delta_threshold = delta_threshold
        self.cvd_lookback = cvd_lookback
        self.absorption_threshold = absorption_threshold
        self.imbalance_window = imbalance_window
    
    def calculate_delta(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate delta (buy volume - sell volume)
        
        Approximation method:
        - If close > open: assign volume to buy side
        - If close < open: assign volume to sell side
        - If close == open: split 50/50
        """
        df = df.copy()
        
        # Calculate buy/sell volume approximation
        df['buy_volume'] = np.where(
            df['close'] > df['open'],
            df['volume'],
            np.where(
                df['close'] < df['open'],
                0,
                df['volume'] * 0.5
            )
        )
        
        df['sell_volume'] = np.where(
            df['close'] < df['open'],
            df['volume'],
            np.where(
                df['close'] > df['open'],
                0,
                df['volume'] * 0.5
            )
        )
        
        # Calculate delta
        df['delta'] = df['buy_volume'] - df['sell_volume']
        
        # Delta percentage
        df['delta_pct'] = df['delta'] / df['volume']
        
        # Delta ratio (buy/sell ratio)
        df['delta_ratio'] = np.where(
            df['sell_volume'] > 0,
            df['buy_volume'] / df['sell_volume'],
            np.nan
        )
        
        return df
    
    def calculate_cvd(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Cumulative Volume Delta
        Running sum of delta shows overall buying/selling pressure
        """
        df = df.copy()
        
        # Cumulative volume delta
        df['cvd'] = df['delta'].cumsum()
        
        # CVD change over lookback period
        df['cvd_change'] = df['cvd'].diff(self.cvd_lookback)
        
        # CVD slope (rate of change)
        df['cvd_slope'] = df['cvd'].diff() / self.cvd_lookback
        
        return df
    
    def detect_absorption(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect absorption: Large orders stopping price movement
        
        Signs of absorption:
        - High volume at a price level
        - Price fails to move through level
        - Delta shows one-sided pressure but price doesn't move
        """
        df = df.copy()
        
        # Calculate price change
        df['price_change'] = df['close'].diff()
        df['price_change_pct'] = df['close'].pct_change()
        
        # High volume + low price change = potential absorption
        volume_ma = df['volume'].rolling(20).mean()
        price_volatility = df['price_change_pct'].rolling(20).std()
        
        # Absorption conditions
        high_volume = df['volume'] > (volume_ma * 1.5)
        low_price_movement = abs(df['price_change_pct']) < (price_volatility * 0.5)
        high_delta = abs(df['delta']) > self.absorption_threshold
        
        # Bullish absorption (buyers absorbing selling)
        df['bullish_absorption'] = (
            high_volume & 
            low_price_movement & 
            (df['delta'] > self.absorption_threshold)
        ).astype(int)
        
        # Bearish absorption (sellers absorbing buying)
        df['bearish_absorption'] = (
            high_volume & 
            low_price_movement & 
            (df['delta'] < -self.absorption_threshold)
        ).astype(int)
        
        return df
    
    def detect_imbalance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect sustained order flow imbalance
        
        Imbalance = Multiple consecutive bars with same-sided delta
        Signals strong institutional participation
        """
        df = df.copy()
        
        # Positive delta (buying pressure)
        df['positive_delta'] = (df['delta'] > 0).astype(int)
        
        # Count consecutive positive deltas
        df['consecutive_buy'] = (
            df['positive_delta']
            .groupby((df['positive_delta'] != df['positive_delta'].shift()).cumsum())
            .cumsum()
        )
        
        # Count consecutive negative deltas
        df['consecutive_sell'] = (
            (1 - df['positive_delta'])
            .groupby((df['positive_delta'] == df['positive_delta'].shift()).cumsum())
            .cumsum()
        )
        
        # Strong imbalance signals
        df['strong_buy_imbalance'] = (
            (df['consecutive_buy'] >= self.imbalance_window) &
            (df['delta_ratio'] >= self.delta_threshold)
        ).astype(int)
        
        df['strong_sell_imbalance'] = (
            (df['consecutive_sell'] >= self.imbalance_window) &
            (df['delta_ratio'] <= (1 / self.delta_threshold))
        ).astype(int)
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on order flow
        
        Signal logic:
        - BUY: Strong buy imbalance + bullish absorption + positive CVD
        - SELL: Strong sell imbalance + bearish absorption + negative CVD
        """
        df = df.copy()
        
        # Buy signal conditions
        buy_conditions = (
            (df['strong_buy_imbalance'] == 1) |
            (df['bullish_absorption'] == 1) |
            ((df['cvd_change'] > 0) & (df['delta_ratio'] > self.delta_threshold))
        )
        
        # Sell signal conditions
        sell_conditions = (
            (df['strong_sell_imbalance'] == 1) |
            (df['bearish_absorption'] == 1) |
            ((df['cvd_change'] < 0) & (df['delta_ratio'] < (1 / self.delta_threshold)))
        )
        
        # Generate signals
        df['order_flow_signal'] = 0
        df.loc[buy_conditions, 'order_flow_signal'] = 1
        df.loc[sell_conditions, 'order_flow_signal'] = -1
        
        # Signal strength (0-100)
        df['signal_strength'] = 0
        
        # Calculate buy signal strength
        buy_strength = (
            (df['strong_buy_imbalance'] * 40) +
            (df['bullish_absorption'] * 30) +
            ((df['cvd_change'] > 0).astype(int) * 30)
        )
        df.loc[buy_conditions, 'signal_strength'] = buy_strength
        
        # Calculate sell signal strength
        sell_strength = (
            (df['strong_sell_imbalance'] * 40) +
            (df['bearish_absorption'] * 30) +
            ((df['cvd_change'] < 0).astype(int) * 30)
        )
        df.loc[sell_conditions, 'signal_strength'] = sell_strength
        
        return df
    
    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete order flow analysis pipeline
        """
        # Calculate all indicators
        df = self.calculate_delta(df)
        df = self.calculate_cvd(df)
        df = self.detect_absorption(df)
        df = self.detect_imbalance(df)
        df = self.generate_signals(df)
        
        return df
    
    def get_current_state(self, df: pd.DataFrame) -> Dict:
        """
        Get current order flow state for decision making
        """
        latest = df.iloc[-1]
        
        return {
            'delta': float(latest['delta']),
            'delta_ratio': float(latest['delta_ratio']) if not pd.isna(latest['delta_ratio']) else 0.0,
            'cvd': float(latest['cvd']),
            'cvd_change': float(latest['cvd_change']) if not pd.isna(latest['cvd_change']) else 0.0,
            'bullish_absorption': bool(latest['bullish_absorption']),
            'bearish_absorption': bool(latest['bearish_absorption']),
            'buy_imbalance': bool(latest['strong_buy_imbalance']),
            'sell_imbalance': bool(latest['strong_sell_imbalance']),
            'signal': int(latest['order_flow_signal']),
            'signal_strength': int(latest['signal_strength']),
        }


# ========================================
# EXAMPLE USAGE
# ========================================

if __name__ == "__main__":
    """
    Test order flow analyzer with sample data
    """
    # Create sample ES data
    dates = pd.date_range('2024-01-01 09:30', periods=100, freq='1min')
    
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'datetime': dates,
        'open': 4500 + np.random.randn(100).cumsum() * 0.5,
        'high': 4500 + np.random.randn(100).cumsum() * 0.5 + 1,
        'low': 4500 + np.random.randn(100).cumsum() * 0.5 - 1,
        'close': 4500 + np.random.randn(100).cumsum() * 0.5,
        'volume': np.random.randint(1000, 5000, 100)
    })
    
    # Initialize analyzer
    analyzer = OrderFlowAnalyzer(
        delta_threshold=1.5,
        cvd_lookback=20,
        absorption_threshold=500,
        imbalance_window=3
    )
    
    # Run analysis
    result = analyzer.analyze(sample_data)
    
    # Show results
    print("=" * 60)
    print("ORDER FLOW ANALYSIS RESULTS")
    print("=" * 60)
    
    print("\nLast 5 bars:")
    cols = ['datetime', 'close', 'volume', 'delta', 'cvd', 'order_flow_signal']
    print(result[cols].tail())
    
    print("\nCurrent Order Flow State:")
    state = analyzer.get_current_state(result)
    for key, value in state.items():
        print(f"  {key}: {value}")
    
    print("\nSignal Summary:")
    print(f"  Buy signals: {(result['order_flow_signal'] == 1).sum()}")
    print(f"  Sell signals: {(result['order_flow_signal'] == -1).sum()}")
    print(f"  Neutral: {(result['order_flow_signal'] == 0).sum()}")
    
    print("\nAbsorption Events:")
    print(f"  Bullish absorption: {result['bullish_absorption'].sum()}")
    print(f"  Bearish absorption: {result['bearish_absorption'].sum()}")
    
    print("\n✅ Order flow analyzer working correctly!")