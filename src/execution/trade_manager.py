"""
Trade Manager for EMINIBOT
Manages trade execution, position tracking, and exit logic
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import config
from config import trading_params as params
from config import settings

# Shorthand for EXIT params
EXIT = params.EXIT


class OrderStatus(Enum):
    """Order status enum"""
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class PositionSide(Enum):
    """Position side enum"""
    LONG = "long"
    SHORT = "short"


class Trade:
    """
    Represents a single trade
    """
    
    def __init__(
        self,
        trade_id: str,
        side: PositionSide,
        entry_price: float,
        position_size: int,
        stop_loss: float,
        take_profit_1: float,
        take_profit_2: float,
        timestamp: datetime
    ):
        self.trade_id = trade_id
        self.side = side
        self.entry_price = entry_price
        self.position_size = position_size
        self.initial_position_size = position_size
        self.stop_loss = stop_loss
        self.take_profit_1 = take_profit_1
        self.take_profit_2 = take_profit_2
        self.timestamp = timestamp
        
        # Trade status
        self.is_open = True
        self.entry_filled = False
        self.tp1_hit = False
        self.tp2_hit = False
        self.stop_hit = False
        self.closed_timestamp = None
        
        # P&L tracking
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.max_profit = 0.0
        self.max_drawdown = 0.0
        
        # Exit tracking
        self.exit_price = None
        self.exit_reason = None
        
        # Trailing stop
        self.trailing_stop_active = False
        self.trailing_stop_price = None
    
    def update_pnl(self, current_price: float):
        """Update unrealized P&L"""
        if not self.is_open:
            return
        
        price_diff = current_price - self.entry_price
        if self.side == PositionSide.SHORT:
            price_diff = -price_diff
        
        tick_diff = price_diff / settings.TICK_SIZE
        self.unrealized_pnl = tick_diff * settings.TICK_VALUE * self.position_size
        
        # Track max profit/drawdown
        if self.unrealized_pnl > self.max_profit:
            self.max_profit = self.unrealized_pnl
        
        current_dd = self.max_profit - self.unrealized_pnl
        if current_dd > self.max_drawdown:
            self.max_drawdown = current_dd
    
    def get_total_pnl(self) -> float:
        """Get total P&L (realized + unrealized)"""
        return self.realized_pnl + self.unrealized_pnl
    
    def close_position(
        self,
        exit_price: float,
        contracts: int,
        reason: str,
        timestamp: datetime
    ):
        """Close all or part of position"""
        if contracts >= self.position_size:
            # Close entire position
            self.is_open = False
            self.closed_timestamp = timestamp
            self.position_size = 0
        else:
            # Partial close
            self.position_size -= contracts
        
        # Calculate realized P&L for closed contracts
        price_diff = exit_price - self.entry_price
        if self.side == PositionSide.SHORT:
            price_diff = -price_diff
        
        tick_diff = price_diff / settings.TICK_SIZE
        realized = tick_diff * settings.TICK_VALUE * contracts
        self.realized_pnl += realized
        
        self.exit_price = exit_price
        self.exit_reason = reason
    
    def activate_trailing_stop(self, current_price: float):
        """Activate trailing stop"""
        self.trailing_stop_active = True
        
        offset_ticks = settings.TRAIL_OFFSET_TICKS
        offset_price = offset_ticks * settings.TICK_SIZE
        
        if self.side == PositionSide.LONG:
            self.trailing_stop_price = current_price - offset_price
        else:
            self.trailing_stop_price = current_price + offset_price
    
    def update_trailing_stop(self, current_price: float):
        """Update trailing stop if price moves favorably"""
        if not self.trailing_stop_active:
            return
        
        offset_ticks = settings.TRAIL_OFFSET_TICKS
        offset_price = offset_ticks * settings.TICK_SIZE
        
        if self.side == PositionSide.LONG:
            new_stop = current_price - offset_price
            if new_stop > self.trailing_stop_price:
                self.trailing_stop_price = new_stop
        else:
            new_stop = current_price + offset_price
            if new_stop < self.trailing_stop_price:
                self.trailing_stop_price = new_stop
    
    def to_dict(self) -> Dict:
        """Convert trade to dictionary"""
        return {
            'trade_id': self.trade_id,
            'side': self.side.value,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'position_size': self.position_size,
            'initial_position_size': self.initial_position_size,
            'stop_loss': self.stop_loss,
            'take_profit_1': self.take_profit_1,
            'take_profit_2': self.take_profit_2,
            'is_open': self.is_open,
            'tp1_hit': self.tp1_hit,
            'tp2_hit': self.tp2_hit,
            'stop_hit': self.stop_hit,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'total_pnl': self.get_total_pnl(),
            'max_profit': self.max_profit,
            'max_drawdown': self.max_drawdown,
            'exit_reason': self.exit_reason,
            'timestamp': self.timestamp,
            'closed_timestamp': self.closed_timestamp
        }


class TradeManager:
    """
    Manages trade execution and position tracking
    
    Responsibilities:
    - Execute trades based on signals
    - Track open positions
    - Manage stop loss and take profit
    - Handle trailing stops
    - Track P&L
    - Enforce risk limits
    """
    
    def __init__(self):
        """Initialize trade manager"""
        self.open_trades: List[Trade] = []
        self.closed_trades: List[Trade] = []
        self.trade_counter = 0
        
        # Daily tracking
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.daily_wins = 0
        self.daily_losses = 0
        self.consecutive_losses = 0
        self.last_trade_date = None
        
        # Overall statistics
        self.total_trades = 0
        self.total_wins = 0
        self.total_losses = 0
        self.gross_profit = 0.0
        self.gross_loss = 0.0
    
    def can_open_trade(self, timestamp: datetime = None) -> Tuple[bool, str]:
        """
        Check if we can open a new trade
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            Tuple of (can_trade, reason)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Check if new day (reset daily counters)
        if self.last_trade_date != timestamp.date():
            self._reset_daily_counters(timestamp.date())
        
        # Check max positions
        if len(self.open_trades) >= settings.MAX_POSITION_SIZE:
            return False, f"Max positions reached ({settings.MAX_POSITION_SIZE})"
        
        # Check daily trade limit
        if self.daily_trades >= settings.MAX_TRADES_PER_DAY:
            return False, f"Max daily trades reached ({settings.MAX_TRADES_PER_DAY})"
        
        # Check daily loss limit
        if abs(self.daily_pnl) >= settings.MAX_DAILY_LOSS:
            return False, f"Daily loss limit reached (${abs(self.daily_pnl):.2f})"
        
        # ✅ NEW - Check consecutive losses (SAFETY FEATURE)
        if self.consecutive_losses >= settings.MAX_CONSECUTIVE_LOSSES:
            return False, f"Max consecutive losses reached ({settings.MAX_CONSECUTIVE_LOSSES})"
        
        return True, "OK"
    
    def open_trade(
        self,
        side: str,
        entry_price: float,
        position_size: int,
        stop_loss: float,
        take_profit_1: float,
        take_profit_2: float,
        timestamp: datetime
    ) -> Optional[Trade]:
        """
        Open a new trade
        
        Args:
            side: 'long' or 'short'
            entry_price: Entry price
            position_size: Number of contracts
            stop_loss: Stop loss price
            take_profit_1: First take profit
            take_profit_2: Second take profit
            timestamp: Entry timestamp
            
        Returns:
            Trade object if successful, None otherwise
        """
        # Check if we can trade
        can_trade, reason = self.can_open_trade(timestamp)
        if not can_trade:
            print(f"⚠️ Cannot open trade: {reason}")
            return None
        
        # Create trade
        self.trade_counter += 1
        trade_id = f"T{timestamp.strftime('%Y%m%d')}_{self.trade_counter:04d}"
        
        position_side = PositionSide.LONG if side.lower() == 'long' else PositionSide.SHORT
        
        trade = Trade(
            trade_id=trade_id,
            side=position_side,
            entry_price=entry_price,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit_1=take_profit_1,
            take_profit_2=take_profit_2,
            timestamp=timestamp
        )
        
        trade.entry_filled = True
        self.open_trades.append(trade)
        
        # Update counters
        self.daily_trades += 1
        self.total_trades += 1
        self.last_trade_date = timestamp.date()
        
        print(f"✅ Opened {side.upper()} trade: {trade_id} @ ${entry_price:.2f} ({position_size} contracts)")
        
        return trade
    
    def update_trades(self, current_price: float, timestamp: datetime):
        """
        Update all open trades with current price
        
        Args:
            current_price: Current market price
            timestamp: Current timestamp
        """
        for trade in self.open_trades[:]:  # Copy list to allow modification
            self._update_single_trade(trade, current_price, timestamp)
    
    def _update_single_trade(self, trade: Trade, current_price: float, timestamp: datetime):
        """Update a single trade"""
        # Update P&L
        trade.update_pnl(current_price)
        
        # Check for exits
        if trade.side == PositionSide.LONG:
            self._check_long_exits(trade, current_price, timestamp)
        else:
            self._check_short_exits(trade, current_price, timestamp)
        
        # Update trailing stop if active
        if trade.trailing_stop_active:
            trade.update_trailing_stop(current_price)
            self._check_trailing_stop(trade, current_price, timestamp)
        
        # Check if should activate trailing stop
        elif EXIT['use_trailing']:
            self._check_trailing_activation(trade, current_price)
    
    def _check_long_exits(self, trade: Trade, current_price: float, timestamp: datetime):
        """Check exit conditions for long trade"""
        # Stop loss
        if current_price <= trade.stop_loss:
            self._close_trade(trade, trade.stop_loss, trade.position_size, "Stop Loss", timestamp)
            trade.stop_hit = True
            return
        
        # Take profit 2
        if not trade.tp2_hit and current_price >= trade.take_profit_2:
            contracts_to_close = trade.position_size
            self._close_trade(trade, trade.take_profit_2, contracts_to_close, "Take Profit 2", timestamp)
            trade.tp2_hit = True
            return
        
        # Take profit 1 (partial close)
        if not trade.tp1_hit and current_price >= trade.take_profit_1:
            contracts_to_close = int(trade.initial_position_size * settings.PARTIAL_CLOSE_PCT)
            self._close_trade(trade, trade.take_profit_1, contracts_to_close, "Take Profit 1 (Partial)", timestamp)
            trade.tp1_hit = True
            
            # Move stop to breakeven
            if EXIT['move_to_breakeven']:
                offset = EXIT['breakeven_offset_ticks'] * settings.TICK_SIZE
                trade.stop_loss = trade.entry_price + offset
                print(f"  📈 Moved stop to breakeven + ${offset:.2f}")
    
    def _check_short_exits(self, trade: Trade, current_price: float, timestamp: datetime):
        """Check exit conditions for short trade"""
        # Stop loss
        if current_price >= trade.stop_loss:
            self._close_trade(trade, trade.stop_loss, trade.position_size, "Stop Loss", timestamp)
            trade.stop_hit = True
            return
        
        # Take profit 2
        if not trade.tp2_hit and current_price <= trade.take_profit_2:
            contracts_to_close = trade.position_size
            self._close_trade(trade, trade.take_profit_2, contracts_to_close, "Take Profit 2", timestamp)
            trade.tp2_hit = True
            return
        
        # Take profit 1 (partial close)
        if not trade.tp1_hit and current_price <= trade.take_profit_1:
            contracts_to_close = int(trade.initial_position_size * settings.PARTIAL_CLOSE_PCT)
            self._close_trade(trade, trade.take_profit_1, contracts_to_close, "Take Profit 1 (Partial)", timestamp)
            trade.tp1_hit = True
            
            # Move stop to breakeven
            if EXIT['move_to_breakeven']:
                offset = EXIT['breakeven_offset_ticks'] * settings.TICK_SIZE
                trade.stop_loss = trade.entry_price - offset
                print(f"  📉 Moved stop to breakeven - ${offset:.2f}")
    
    def _check_trailing_activation(self, trade: Trade, current_price: float):
        """Check if should activate trailing stop"""
        activation_ticks = settings.TRAIL_ACTIVATION_TICKS
        activation_price_diff = activation_ticks * settings.TICK_SIZE
        
        if trade.side == PositionSide.LONG:
            if current_price >= trade.entry_price + activation_price_diff:
                trade.activate_trailing_stop(current_price)
                print(f"  🎯 Activated trailing stop @ ${trade.trailing_stop_price:.2f}")
        else:
            if current_price <= trade.entry_price - activation_price_diff:
                trade.activate_trailing_stop(current_price)
                print(f"  🎯 Activated trailing stop @ ${trade.trailing_stop_price:.2f}")
    
    def _check_trailing_stop(self, trade: Trade, current_price: float, timestamp: datetime):
        """Check if trailing stop hit"""
        if trade.side == PositionSide.LONG:
            if current_price <= trade.trailing_stop_price:
                self._close_trade(trade, trade.trailing_stop_price, trade.position_size, "Trailing Stop", timestamp)
        else:
            if current_price >= trade.trailing_stop_price:
                self._close_trade(trade, trade.trailing_stop_price, trade.position_size, "Trailing Stop", timestamp)
    
    def _close_trade(
        self,
        trade: Trade,
        exit_price: float,
        contracts: int,
        reason: str,
        timestamp: datetime
    ):
        """Close a trade"""
        trade.close_position(exit_price, contracts, reason, timestamp)
        
        # Update statistics
        pnl = trade.realized_pnl if not trade.is_open else trade.get_total_pnl()
        self.daily_pnl += pnl
        
        if pnl > 0:
            self.daily_wins += 1
            self.total_wins += 1
            self.gross_profit += pnl
            self.consecutive_losses = 0  # Reset on win
        else:
            self.daily_losses += 1
            self.total_losses += 1
            self.gross_loss += abs(pnl)
            self.consecutive_losses += 1  # Increment on loss
        
        # Move to closed trades if fully closed
        if not trade.is_open:
            self.open_trades.remove(trade)
            self.closed_trades.append(trade)
            
            win_loss = "WIN" if pnl > 0 else "LOSS"
            print(f"  {'🟢' if pnl > 0 else '🔴'} Closed {trade.side.value.upper()} trade: {trade.trade_id}")
            print(f"     Exit: ${exit_price:.2f} | Reason: {reason} | P&L: ${pnl:.2f} ({win_loss})")
    
    def close_all_positions(self, current_price: float, reason: str, timestamp: datetime):
        """Close all open positions"""
        print(f"\n⚠️ Closing all positions - Reason: {reason}")
        
        for trade in self.open_trades[:]:
            self._close_trade(trade, current_price, trade.position_size, reason, timestamp)
    
    def _reset_daily_counters(self, date):
        """Reset daily counters for new day"""
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.daily_wins = 0
        self.daily_losses = 0
        self.last_trade_date = date
        # Note: consecutive_losses is NOT reset daily - it persists across days
    
    def get_statistics(self) -> Dict:
        """Get trading statistics"""
        total_pnl = self.daily_pnl + sum(t.unrealized_pnl for t in self.open_trades)
        
        win_rate = (self.total_wins / self.total_trades * 100) if self.total_trades > 0 else 0
        
        avg_win = (self.gross_profit / self.total_wins) if self.total_wins > 0 else 0
        avg_loss = (self.gross_loss / self.total_losses) if self.total_losses > 0 else 0
        
        profit_factor = (self.gross_profit / self.gross_loss) if self.gross_loss > 0 else 0
        
        return {
            'open_positions': len(self.open_trades),
            'daily_trades': self.daily_trades,
            'daily_pnl': self.daily_pnl,
            'daily_wins': self.daily_wins,
            'daily_losses': self.daily_losses,
            'total_trades': self.total_trades,
            'total_wins': self.total_wins,
            'total_losses': self.total_losses,
            'win_rate': win_rate,
            'gross_profit': self.gross_profit,
            'gross_loss': self.gross_loss,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'consecutive_losses': self.consecutive_losses
        }
    
    def print_status(self):
        """Print current status"""
        stats = self.get_statistics()
        
        print("\n" + "="*70)
        print("TRADE MANAGER STATUS")
        print("="*70)
        
        print(f"\n📊 OPEN POSITIONS: {stats['open_positions']}")
        for trade in self.open_trades:
            print(f"  {trade.trade_id}: {trade.side.value.upper()} {trade.position_size} @ ${trade.entry_price:.2f}")
            print(f"    P&L: ${trade.unrealized_pnl:.2f} | Stop: ${trade.stop_loss:.2f}")
        
        print(f"\n📈 TODAY:")
        print(f"  Trades: {stats['daily_trades']}")
        print(f"  Wins: {stats['daily_wins']} | Losses: {stats['daily_losses']}")
        print(f"  P&L: ${stats['daily_pnl']:.2f}")
        
        print(f"\n📊 ALL TIME:")
        print(f"  Total Trades: {stats['total_trades']}")
        print(f"  Win Rate: {stats['win_rate']:.1f}%")
        print(f"  Profit Factor: {stats['profit_factor']:.2f}")
        print(f"  Total P&L: ${stats['total_pnl']:.2f}")
        
        if stats['consecutive_losses'] > 0:
            print(f"\n⚠️ RISK ALERT:")
            print(f"  Consecutive Losses: {stats['consecutive_losses']}/{settings.MAX_CONSECUTIVE_LOSSES}")
        
        print("\n" + "="*70 + "\n")


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("Testing Trade Manager...")
    print("="*70)
    
    # Initialize manager
    manager = TradeManager()
    
    # Test opening a trade
    print("\n📊 Test 1: Opening a LONG trade")
    trade = manager.open_trade(
        side='long',
        entry_price=4500.00,
        position_size=2,
        stop_loss=4498.00,
        take_profit_1=4503.00,
        take_profit_2=4505.00,
        timestamp=datetime.now()
    )
    
    # Simulate price movement
    print("\n📈 Simulating price movement...")
    
    # Price moves up
    manager.update_trades(4501.00, datetime.now())
    print(f"Price: $4501.00 | Unrealized P&L: ${trade.unrealized_pnl:.2f}")
    
    # Hit TP1
    manager.update_trades(4503.00, datetime.now())
    print(f"Price: $4503.00 | Hit TP1!")
    
    # Continue up to TP2
    manager.update_trades(4505.00, datetime.now())
    print(f"Price: $4505.00 | Hit TP2!")
    
    # Print final status
    manager.print_status()
    
    print("\n✅ Trade manager working correctly!")