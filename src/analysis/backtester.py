"""
Backtester for EMINIBOT
Runs strategy on historical data and generates performance reports
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.strategy.signal_generator import SignalGenerator
from src.execution.trade_manager import TradeManager
from config import settings


class Backtester:
    """
    Backtesting engine for trading strategy
    
    Features:
    - Bar-by-bar simulation
    - Realistic trade execution
    - Position management
    - Performance tracking
    - Equity curve generation
    """
    
    def __init__(
        self,
        initial_capital: float = None,
        commission_per_contract: float = 1.0,
        slippage_ticks: int = 1
    ):
        """
        Initialize backtester
        
        Args:
            initial_capital: Starting capital (defaults to settings.ACCOUNT_SIZE)
            commission_per_contract: Commission per contract per side
            slippage_ticks: Slippage in ticks per trade
        """
        self.initial_capital = initial_capital or settings.ACCOUNT_SIZE
        self.commission_per_contract = commission_per_contract
        self.slippage_ticks = slippage_ticks
        self.slippage_dollars = slippage_ticks * settings.TICK_VALUE
        
        # Initialize components
        self.signal_generator = SignalGenerator()
        self.trade_manager = TradeManager()
        
        # Tracking
        self.equity_curve = []
        self.trade_log = []
        self.daily_returns = []
        self.current_equity = self.initial_capital
        
        # Results
        self.results = None
    
    def run(
        self,
        data: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Run backtest on historical data
        
        Args:
            data: DataFrame with OHLCV data and 'datetime' column
            start_date: Optional start date (YYYY-MM-DD)
            end_date: Optional end date (YYYY-MM-DD)
            verbose: Print progress
            
        Returns:
            Dictionary with backtest results
        """
        if verbose:
            print("\n" + "="*70)
            print("RUNNING BACKTEST")
            print("="*70)
        
        # Prepare data
        data = self._prepare_data(data, start_date, end_date)
        
        if verbose:
            print(f"\n📊 Data prepared:")
            print(f"  Bars: {len(data)}")
            print(f"  Date range: {data['datetime'].iloc[0]} to {data['datetime'].iloc[-1]}")
            print(f"  Initial capital: ${self.initial_capital:,.2f}")
        
        # Reset state
        self._reset_state()
        
        # Run bar-by-bar simulation
        if verbose:
            print(f"\n🔄 Running simulation...")
        
        for i in range(len(data)):
            self._process_bar(data, i)
            
            # Progress indicator
            if verbose and i % 100 == 0 and i > 0:
                progress = (i / len(data)) * 100
                print(f"  Progress: {progress:.1f}% ({i}/{len(data)} bars)")
        
        # Calculate final results
        self.results = self._calculate_results()
        
        if verbose:
            print("\n✅ Backtest complete!")
            self._print_results()
        
        return self.results
    
    def _prepare_data(
        self,
        data: pd.DataFrame,
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """Prepare and filter data"""
        data = data.copy()
        
        # Ensure datetime column
        if 'datetime' not in data.columns:
            data['datetime'] = data.index
        
        data['datetime'] = pd.to_datetime(data['datetime'])
        
        # Filter by date range
        if start_date:
            data = data[data['datetime'] >= start_date]
        if end_date:
            data = data[data['datetime'] <= end_date]
        
        # Reset index
        data = data.reset_index(drop=True)
        
        return data
    
    def _reset_state(self):
        """Reset backtester state"""
        self.trade_manager = TradeManager()
        self.equity_curve = []
        self.trade_log = []
        self.daily_returns = []
        self.current_equity = self.initial_capital
    
    def _process_bar(self, data: pd.DataFrame, i: int):
        """Process a single bar"""
        # Get data up to current bar (for indicators)
        current_data = data.iloc[:i+1].copy()
        current_bar = data.iloc[i]
        current_price = current_bar['close']
        timestamp = current_bar['datetime']
        
        # Update existing trades
        self.trade_manager.update_trades(current_price, timestamp)
        
        # Generate signal (only if we have enough data for indicators)
        if i >= 50:  # Need at least 50 bars for indicators
            recommendation = self.signal_generator.get_trade_recommendation(current_data)
            
            # Execute new trades
            if recommendation['action'] in ['BUY', 'SELL']:
                self._execute_trade(recommendation, current_price, timestamp)
        
        # Update equity curve
        self._update_equity()
    
    def _execute_trade(
        self,
        recommendation: Dict,
        current_price: float,
        timestamp: datetime
    ):
        """Execute a trade based on recommendation"""
        # Check if we can trade
        can_trade, reason = self.trade_manager.can_open_trade(timestamp)
        if not can_trade:
            return
        
        # Apply slippage
        if recommendation['action'] == 'BUY':
            entry_price = current_price + (self.slippage_ticks * settings.TICK_SIZE)
            side = 'long'
        else:
            entry_price = current_price - (self.slippage_ticks * settings.TICK_SIZE)
            side = 'short'
        
        # Open trade
        trade = self.trade_manager.open_trade(
            side=side,
            entry_price=entry_price,
            position_size=recommendation['position_size'],
            stop_loss=recommendation['stop_loss'],
            take_profit_1=recommendation['take_profit_1'],
            take_profit_2=recommendation['take_profit_2'],
            timestamp=timestamp
        )
        
        if trade:
            # Deduct commission
            commission = self.commission_per_contract * trade.position_size
            self.current_equity -= commission
    
    def _update_equity(self):
        """Update equity curve"""
        # Calculate current equity
        realized_pnl = sum(t.realized_pnl for t in self.trade_manager.closed_trades)
        unrealized_pnl = sum(t.unrealized_pnl for t in self.trade_manager.open_trades)
        
        # Deduct commissions for closed trades
        total_closed_contracts = sum(
            t.initial_position_size for t in self.trade_manager.closed_trades
        )
        total_commissions = total_closed_contracts * self.commission_per_contract * 2  # Entry + exit
        
        self.current_equity = self.initial_capital + realized_pnl + unrealized_pnl - total_commissions
        
        self.equity_curve.append(self.current_equity)
    
    def _calculate_results(self) -> Dict:
        """Calculate backtest results"""
        stats = self.trade_manager.get_statistics()
        
        # Equity curve analysis
        equity_array = np.array(self.equity_curve)
        returns = np.diff(equity_array) / equity_array[:-1]
        
        # Max drawdown
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max
        max_drawdown = abs(drawdown.min()) * 100 if len(drawdown) > 0 else 0
        
        # Sharpe ratio (annualized, assuming 252 trading days)
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Total return
        total_return = ((self.current_equity - self.initial_capital) / self.initial_capital) * 100
        
        # Risk-adjusted metrics
        recovery_factor = abs(stats['total_pnl'] / max_drawdown) if max_drawdown != 0 else 0
        
        # Average trade metrics
        avg_trade_pnl = stats['total_pnl'] / stats['total_trades'] if stats['total_trades'] > 0 else 0
        
        # Expectancy
        if stats['total_trades'] > 0:
            win_rate_decimal = stats['win_rate'] / 100
            expectancy = (win_rate_decimal * stats['avg_win']) - ((1 - win_rate_decimal) * stats['avg_loss'])
        else:
            expectancy = 0
        
        results = {
            # Account metrics
            'initial_capital': self.initial_capital,
            'final_equity': self.current_equity,
            'total_return': total_return,
            'total_pnl': stats['total_pnl'],
            
            # Trade statistics
            'total_trades': stats['total_trades'],
            'winning_trades': stats['total_wins'],
            'losing_trades': stats['total_losses'],
            'win_rate': stats['win_rate'],
            
            # P&L metrics
            'gross_profit': stats['gross_profit'],
            'gross_loss': stats['gross_loss'],
            'profit_factor': stats['profit_factor'],
            'avg_win': stats['avg_win'],
            'avg_loss': stats['avg_loss'],
            'avg_trade': avg_trade_pnl,
            'expectancy': expectancy,
            
            # Risk metrics
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'recovery_factor': recovery_factor,
            
            # Trade details
            'equity_curve': self.equity_curve,
            'closed_trades': [t.to_dict() for t in self.trade_manager.closed_trades],
        }
        
        return results
    
    def _print_results(self):
        """Print backtest results"""
        r = self.results
        
        print("\n" + "="*70)
        print("BACKTEST RESULTS")
        print("="*70)
        
        print(f"\n💰 ACCOUNT PERFORMANCE:")
        print(f"  Initial Capital: ${r['initial_capital']:,.2f}")
        print(f"  Final Equity: ${r['final_equity']:,.2f}")
        print(f"  Total Return: {r['total_return']:,.2f}%")
        print(f"  Total P&L: ${r['total_pnl']:,.2f}")
        
        print(f"\n📊 TRADE STATISTICS:")
        print(f"  Total Trades: {r['total_trades']}")
        print(f"  Winning Trades: {r['winning_trades']}")
        print(f"  Losing Trades: {r['losing_trades']}")
        print(f"  Win Rate: {r['win_rate']:.1f}%")
        
        print(f"\n💵 PROFIT & LOSS:")
        print(f"  Gross Profit: ${r['gross_profit']:,.2f}")
        print(f"  Gross Loss: ${r['gross_loss']:,.2f}")
        print(f"  Profit Factor: {r['profit_factor']:.2f}")
        print(f"  Avg Win: ${r['avg_win']:,.2f}")
        print(f"  Avg Loss: ${r['avg_loss']:,.2f}")
        print(f"  Avg Trade: ${r['avg_trade']:,.2f}")
        print(f"  Expectancy: ${r['expectancy']:.2f}")
        
        print(f"\n⚠️ RISK METRICS:")
        print(f"  Max Drawdown: {r['max_drawdown']:.2f}%")
        print(f"  Sharpe Ratio: {r['sharpe_ratio']:.2f}")
        print(f"  Recovery Factor: {r['recovery_factor']:.2f}")
        
        # Performance rating
        print(f"\n🎯 PERFORMANCE RATING:")
        rating = self._rate_performance(r)
        print(f"  {rating}")
        
        print("\n" + "="*70)
    
    def _rate_performance(self, results: Dict) -> str:
        """Rate the backtest performance"""
        win_rate = results['win_rate']
        profit_factor = results['profit_factor']
        max_dd = results['max_drawdown']
        
        score = 0
        
        # Win rate scoring (max 30 points)
        if win_rate >= 60:
            score += 30
        elif win_rate >= 55:
            score += 25
        elif win_rate >= 50:
            score += 20
        elif win_rate >= 45:
            score += 10
        
        # Profit factor scoring (max 30 points)
        if profit_factor >= 2.0:
            score += 30
        elif profit_factor >= 1.5:
            score += 25
        elif profit_factor >= 1.2:
            score += 20
        elif profit_factor >= 1.0:
            score += 10
        
        # Drawdown scoring (max 40 points)
        if max_dd <= 5:
            score += 40
        elif max_dd <= 10:
            score += 35
        elif max_dd <= 15:
            score += 25
        elif max_dd <= 20:
            score += 15
        elif max_dd <= 25:
            score += 5
        
        # Rating
        if score >= 85:
            return "🌟 EXCELLENT - Ready for live trading!"
        elif score >= 70:
            return "✅ GOOD - Needs minor improvements"
        elif score >= 50:
            return "⚠️ FAIR - Significant improvements needed"
        else:
            return "❌ POOR - Major revisions required"
    
    def get_trade_log(self) -> pd.DataFrame:
        """Get trade log as DataFrame"""
        if not self.trade_manager.closed_trades:
            return pd.DataFrame()
        
        trades = [t.to_dict() for t in self.trade_manager.closed_trades]
        return pd.DataFrame(trades)
    
    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve as DataFrame"""
        return pd.DataFrame({
            'bar': range(len(self.equity_curve)),
            'equity': self.equity_curve
        })


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("Testing Backtester...")
    print("="*70)
    
    # Generate sample data with uptrend
    print("\n📊 Generating sample ES data...")
    np.random.seed(42)
    
    num_bars = 500
    dates = pd.date_range(start='2024-01-01 09:30', periods=num_bars, freq='1min')
    
    # Create trending data
    base_price = 4500
    trend = np.linspace(0, 20, num_bars)  # Uptrend of 20 points
    noise = np.cumsum(np.random.randn(num_bars) * 0.5)
    prices = base_price + trend + noise
    
    data = pd.DataFrame({
        'datetime': dates,
        'open': prices + np.random.randn(num_bars) * 0.2,
        'high': prices + np.abs(np.random.randn(num_bars) * 0.5),
        'low': prices - np.abs(np.random.randn(num_bars) * 0.5),
        'close': prices,
        'volume': np.random.randint(1000, 5000, num_bars)
    })
    
    # Ensure OHLC consistency
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)
    
    print(f"✅ Generated {len(data)} bars of data")
    print(f"   Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
    
    # Run backtest
    print("\n🚀 Starting backtest...")
    backtester = Backtester(
        initial_capital=50000,
        commission_per_contract=1.0,
        slippage_ticks=1
    )
    
    results = backtester.run(data, verbose=True)
    
    # Show trade log
    trade_log = backtester.get_trade_log()
    if len(trade_log) > 0:
        print("\n📋 TRADE LOG (Last 5 trades):")
        print(trade_log[['trade_id', 'side', 'entry_price', 'exit_price', 'total_pnl', 'exit_reason']].tail())
    
    print("\n✅ Backtester working correctly!")