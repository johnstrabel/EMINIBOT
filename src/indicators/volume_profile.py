"""
Volume Profile Indicator for E-mini ES Futures
Identifies high-volume price zones, POC, Value Area, and VWAP
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from datetime import datetime


class VolumeProfileAnalyzer:
    """
    Volume Profile analysis for identifying key price levels
    
    Components:
    - Point of Control (POC): Price level with highest volume
    - Value Area: Price range containing 70% of volume
    - VWAP: Volume-weighted average price
    - Volume Nodes: High and low volume zones
    """
    
    def __init__(
        self,
        value_area_pct: float = 0.70,
        num_bins: int = 50,
        vwap_std_devs: list = [1, 2, 3]
    ):
        """
        Initialize Volume Profile Analyzer
        
        Args:
            value_area_pct: Percentage of volume for value area (default 70%)
            num_bins: Number of price bins for profile (default 50)
            vwap_std_devs: Standard deviation bands for VWAP
        """
        self.value_area_pct = value_area_pct
        self.num_bins = num_bins
        self.vwap_std_devs = vwap_std_devs if vwap_std_devs else [1, 2, 3]
        
    def calculate_poc(self, df: pd.DataFrame) -> Dict:
        """
        Calculate Point of Control (POC)
        POC = Price level with highest volume
        
        Args:
            df: DataFrame with 'close', 'high', 'low', 'volume' columns
            
        Returns:
            Dictionary with POC price and volume
        """
        if len(df) == 0:
            return {'poc_price': None, 'poc_volume': 0}
        
        # Create price bins
        price_min = df['low'].min()
        price_max = df['high'].max()
        bins = np.linspace(price_min, price_max, self.num_bins)
        
        # Assign volume to price bins
        volume_profile = np.zeros(len(bins) - 1)
        
        for idx, row in df.iterrows():
            # Distribute volume across price range for each bar
            bar_bins = np.digitize([row['low'], row['high']], bins)
            start_bin = max(0, bar_bins[0] - 1)
            end_bin = min(len(volume_profile), bar_bins[1])
            
            # Distribute volume evenly across bins
            num_bins_touched = end_bin - start_bin
            if num_bins_touched > 0:
                volume_per_bin = row['volume'] / num_bins_touched
                volume_profile[start_bin:end_bin] += volume_per_bin
        
        # Find POC
        poc_bin = np.argmax(volume_profile)
        poc_price = (bins[poc_bin] + bins[poc_bin + 1]) / 2
        poc_volume = volume_profile[poc_bin]
        
        return {
            'poc_price': poc_price,
            'poc_volume': poc_volume,
            'volume_profile': volume_profile,
            'price_bins': bins
        }
    
    def calculate_value_area(
        self,
        df: pd.DataFrame,
        poc_data: Dict
    ) -> Dict:
        """
        Calculate Value Area High (VAH) and Value Area Low (VAL)
        Value Area = Price range containing specified % of volume
        
        Args:
            df: DataFrame with price/volume data
            poc_data: Dictionary from calculate_poc()
            
        Returns:
            Dictionary with VAH, VAL, and value area volume
        """
        if poc_data['poc_price'] is None:
            return {
                'vah': None,
                'val': None,
                'value_area_volume': 0
            }
        
        volume_profile = poc_data['volume_profile']
        bins = poc_data['price_bins']
        poc_bin = np.argmax(volume_profile)
        
        total_volume = volume_profile.sum()
        target_volume = total_volume * self.value_area_pct
        
        # Expand from POC outward until we hit target volume
        value_area_volume = volume_profile[poc_bin]
        lower_bin = poc_bin
        upper_bin = poc_bin
        
        while value_area_volume < target_volume:
            # Check which direction has more volume
            lower_volume = volume_profile[lower_bin - 1] if lower_bin > 0 else 0
            upper_volume = volume_profile[upper_bin + 1] if upper_bin < len(volume_profile) - 1 else 0
            
            if lower_volume > upper_volume and lower_bin > 0:
                lower_bin -= 1
                value_area_volume += lower_volume
            elif upper_bin < len(volume_profile) - 1:
                upper_bin += 1
                value_area_volume += upper_volume
            else:
                break
        
        val = bins[lower_bin]
        vah = bins[upper_bin + 1]
        
        return {
            'vah': vah,
            'val': val,
            'value_area_volume': value_area_volume,
            'value_area_pct': value_area_volume / total_volume if total_volume > 0 else 0
        }
    
    def calculate_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate VWAP and standard deviation bands
        
        Args:
            df: DataFrame with 'close', 'high', 'low', 'volume' columns
            
        Returns:
            DataFrame with VWAP and band columns added
        """
        df = df.copy()
        
        # Typical price
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        # VWAP calculation
        df['vwap_cumsum'] = (df['typical_price'] * df['volume']).cumsum()
        df['volume_cumsum'] = df['volume'].cumsum()
        df['vwap'] = df['vwap_cumsum'] / df['volume_cumsum']
        
        # Standard deviation bands
        df['squared_diff'] = ((df['typical_price'] - df['vwap']) ** 2) * df['volume']
        df['squared_diff_cumsum'] = df['squared_diff'].cumsum()
        df['vwap_variance'] = df['squared_diff_cumsum'] / df['volume_cumsum']
        df['vwap_std'] = np.sqrt(df['vwap_variance'])
        
        # Create bands
        for std in self.vwap_std_devs:
            df[f'vwap_upper_{std}'] = df['vwap'] + (df['vwap_std'] * std)
            df[f'vwap_lower_{std}'] = df['vwap'] - (df['vwap_std'] * std)
        
        # Clean up intermediate columns
        df = df.drop([
            'typical_price', 'vwap_cumsum', 'volume_cumsum',
            'squared_diff', 'squared_diff_cumsum', 'vwap_variance'
        ], axis=1)
        
        return df
    
    def identify_volume_nodes(
        self,
        poc_data: Dict,
        percentile_high: float = 80,
        percentile_low: float = 20
    ) -> Dict:
        """
        Identify high volume nodes (HVN) and low volume nodes (LVN)
        
        Args:
            poc_data: Dictionary from calculate_poc()
            percentile_high: Percentile threshold for HVN (default 80)
            percentile_low: Percentile threshold for LVN (default 20)
            
        Returns:
            Dictionary with HVN and LVN price levels
        """
        volume_profile = poc_data['volume_profile']
        bins = poc_data['price_bins']
        
        high_threshold = np.percentile(volume_profile, percentile_high)
        low_threshold = np.percentile(volume_profile, percentile_low)
        
        hvn_prices = []
        lvn_prices = []
        
        for i, volume in enumerate(volume_profile):
            price = (bins[i] + bins[i + 1]) / 2
            
            if volume >= high_threshold:
                hvn_prices.append(price)
            elif volume <= low_threshold:
                lvn_prices.append(price)
        
        return {
            'hvn': hvn_prices,  # High Volume Nodes (support/resistance)
            'lvn': lvn_prices,  # Low Volume Nodes (breakout zones)
            'hvn_threshold': high_threshold,
            'lvn_threshold': low_threshold
        }
    
    def generate_signals(
        self,
        df: pd.DataFrame,
        poc_data: Dict,
        value_area: Dict,
        volume_nodes: Dict
    ) -> pd.DataFrame:
        """
        Generate trading signals based on volume profile
        
        Signal logic:
        - Price near POC = 0 (neutral, high activity)
        - Price above VAH = +1 (bullish, above value)
        - Price below VAL = -1 (bearish, below value)
        - Price at HVN = potential support/resistance
        - Price at LVN = potential breakout zone
        
        Args:
            df: DataFrame with price data
            poc_data: POC information
            value_area: Value area information
            volume_nodes: Volume node information
            
        Returns:
            DataFrame with signals added
        """
        df = df.copy()
        
        poc_price = poc_data['poc_price']
        vah = value_area['vah']
        val = value_area['val']
        
        # Initialize signals
        df['vp_signal'] = 0
        df['near_poc'] = False
        df['near_hvn'] = False
        df['near_lvn'] = False
        
        if poc_price is None or vah is None or val is None:
            return df
        
        poc_tolerance = (vah - val) * 0.1  # 10% of value area width
        
        for idx in df.index:
            price = df.loc[idx, 'close']
            
            # Check price relative to value area
            if price > vah:
                df.loc[idx, 'vp_signal'] = 1  # Above value
            elif price < val:
                df.loc[idx, 'vp_signal'] = -1  # Below value
            
            # Check if near POC
            if abs(price - poc_price) < poc_tolerance:
                df.loc[idx, 'near_poc'] = True
            
            # Check if near HVN
            for hvn_price in volume_nodes['hvn']:
                if abs(price - hvn_price) < poc_tolerance:
                    df.loc[idx, 'near_hvn'] = True
                    break
            
            # Check if near LVN
            for lvn_price in volume_nodes['lvn']:
                if abs(price - lvn_price) < poc_tolerance:
                    df.loc[idx, 'near_lvn'] = True
                    break
        
        return df
    
    def analyze(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Complete volume profile analysis pipeline
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Tuple of (analyzed DataFrame, analysis results dictionary)
        """
        # Calculate POC
        poc_data = self.calculate_poc(df)
        
        # Calculate Value Area
        value_area = self.calculate_value_area(df, poc_data)
        
        # Calculate VWAP
        df = self.calculate_vwap(df)
        
        # Identify volume nodes
        volume_nodes = self.identify_volume_nodes(poc_data)
        
        # Generate signals
        df = self.generate_signals(df, poc_data, value_area, volume_nodes)
        
        # Compile results
        results = {
            'poc': poc_data,
            'value_area': value_area,
            'volume_nodes': volume_nodes,
            'current_price': df['close'].iloc[-1] if len(df) > 0 else None,
            'vwap_current': df['vwap'].iloc[-1] if len(df) > 0 else None
        }
        
        return df, results
    
    def get_current_state(self, df: pd.DataFrame, results: Dict) -> Dict:
        """
        Get current volume profile state for last bar
        
        Args:
            df: Analyzed DataFrame
            results: Results from analyze()
            
        Returns:
            Dictionary with current state
        """
        if len(df) == 0:
            return {}
        
        last_row = df.iloc[-1]
        
        return {
            'price': last_row['close'],
            'vwap': last_row['vwap'],
            'vwap_upper_1': last_row['vwap_upper_1'],
            'vwap_lower_1': last_row['vwap_lower_1'],
            'poc_price': results['poc']['poc_price'],
            'vah': results['value_area']['vah'],
            'val': results['value_area']['val'],
            'signal': int(last_row['vp_signal']),
            'near_poc': bool(last_row['near_poc']),
            'near_hvn': bool(last_row['near_hvn']),
            'near_lvn': bool(last_row['near_lvn'])
        }


# Test code
if __name__ == "__main__":
    print("Testing Volume Profile Analyzer...")
    print("=" * 60)
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01 09:30', periods=100, freq='1min')
    
    base_price = 4500
    prices = base_price + np.cumsum(np.random.randn(100) * 0.5)
    
    df = pd.DataFrame({
        'datetime': dates,
        'open': prices + np.random.randn(100) * 0.2,
        'high': prices + np.abs(np.random.randn(100) * 0.5),
        'low': prices - np.abs(np.random.randn(100) * 0.5),
        'close': prices,
        'volume': np.random.randint(1000, 5000, 100)
    })
    
    # Ensure high >= close >= low
    df['high'] = df[['high', 'close']].max(axis=1)
    df['low'] = df[['low', 'close']].min(axis=1)
    
    # Analyze
    analyzer = VolumeProfileAnalyzer()
    df_analyzed, results = analyzer.analyze(df)
    
    # Print results
    print("\nVOLUME PROFILE ANALYSIS RESULTS")
    print("=" * 60)
    
    print(f"\nPoint of Control:")
    print(f"  POC Price: ${results['poc']['poc_price']:.2f}")
    print(f"  POC Volume: {results['poc']['poc_volume']:.0f}")
    
    print(f"\nValue Area:")
    print(f"  VAH (Value Area High): ${results['value_area']['vah']:.2f}")
    print(f"  VAL (Value Area Low): ${results['value_area']['val']:.2f}")
    print(f"  Value Area Width: ${results['value_area']['vah'] - results['value_area']['val']:.2f}")
    print(f"  Volume %: {results['value_area']['value_area_pct']:.1%}")
    
    print(f"\nVolume Nodes:")
    print(f"  High Volume Nodes (HVN): {len(results['volume_nodes']['hvn'])} zones")
    print(f"  Low Volume Nodes (LVN): {len(results['volume_nodes']['lvn'])} zones")
    
    print(f"\nCurrent State:")
    state = analyzer.get_current_state(df_analyzed, results)
    print(f"  Current Price: ${state['price']:.2f}")
    print(f"  VWAP: ${state['vwap']:.2f}")
    print(f"  Distance from VWAP: ${state['price'] - state['vwap']:.2f}")
    print(f"  Near POC: {state['near_poc']}")
    print(f"  Near HVN: {state['near_hvn']}")
    print(f"  Near LVN: {state['near_lvn']}")
    print(f"  Signal: {state['signal']} ", end="")
    if state['signal'] == 1:
        print("(Above Value Area)")
    elif state['signal'] == -1:
        print("(Below Value Area)")
    else:
        print("(In Value Area)")
    
    print(f"\nLast 5 bars with VWAP:")
    print(df_analyzed[['datetime', 'close', 'vwap', 'vwap_upper_1', 'vwap_lower_1', 'vp_signal']].tail())
    
    print("\n✅ Volume profile analyzer working correctly!")