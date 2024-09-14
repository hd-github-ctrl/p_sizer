import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, List
from exchange_connector import ExchangeConnector
from order_execution_engine import OrderExecutionEngine
from position_sizing import PositionSizing
from risk_management import RiskManagement
from datetime import datetime

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler('position_entry_and_signal.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class TradeSignalGenerator:
    """
    Optimized class using ICT concepts for generating trade signals, identifying entry points, 
    and managing stop-loss, take-profit, and trade execution.
    """
    
    def __init__(self, exchange_connector: ExchangeConnector, position_sizing: PositionSizing,
                 risk_management: RiskManagement, order_execution_engine: OrderExecutionEngine, 
                 strategy_parameters: Dict[str, float]):
        """
        Initialize with necessary data and strategy parameters.
        """
        self.exchange_connector = exchange_connector
        self.position_sizing = position_sizing
        self.risk_management = risk_management
        self.order_execution_engine = order_execution_engine
        self.strategy_parameters = strategy_parameters
        logger.info("TradeSignalGenerator initialized.")
        
    def calculate_atr(self, price_data: pd.DataFrame) -> float:
        """
        Calculate the Average True Range (ATR) for volatility assessment.
        """
        atr_period = self.strategy_parameters.get('atr_period', 14)
        high_low = price_data['High'] - price_data['Low']
        high_close = np.abs(price_data['High'] - price_data['Close'].shift())
        low_close = np.abs(price_data['Low'] - price_data['Close'].shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=atr_period).mean().iloc[-1]
        return atr

    def identify_fvg(self, price_data: pd.DataFrame) -> list:
        """
        Identify potential Fair Value Gaps (FVG) from historical price data.
        """
        fvgs = []
        for i in range(2, len(price_data)):
            current_candle = price_data.iloc[i]
            prev_candle = price_data.iloc[i - 1]
            prev_prev_candle = price_data.iloc[i - 2]

            if current_candle['Low'] > prev_candle['High'] and prev_prev_candle['Low'] > prev_candle['High']:
                # Bullish FVG
                fvgs.append({
                    'type': 'bullish',
                    'high': current_candle['Low'],
                    'low': prev_candle['High'],
                    'timestamp': current_candle.name
                })
            elif current_candle['High'] < prev_candle['Low'] and prev_prev_candle['High'] < prev_candle['Low']:
                # Bearish FVG
                fvgs.append({
                    'type': 'bearish',
                    'high': prev_candle['Low'],
                    'low': current_candle['High'],
                    'timestamp': current_candle.name
                })
        return fvgs

    def identify_order_blocks(self, price_data: pd.DataFrame) -> list:
        """
        Identify potential Order Blocks (OBs) from historical price data.
        """
        order_blocks = []
        for i in range(1, len(price_data)):
            current_candle = price_data.iloc[i]
            prev_candle = price_data.iloc[i - 1]

            if current_candle['Close'] > current_candle['Open'] and prev_candle['Close'] < prev_candle['Open']:
                # Bullish Order Block
                order_blocks.append({
                    'type': 'bullish',
                    'high': max(prev_candle['High'], current_candle['High']),
                    'low': min(prev_candle['Low'], current_candle['Low']),
                    'timestamp': current_candle.name
                })
            elif current_candle['Close'] < current_candle['Open'] and prev_candle['Close'] > prev_candle['Open']:
                # Bearish Order Block
                order_blocks.append({
                    'type': 'bearish',
                    'high': max(prev_candle['High'], current_candle['High']),
                    'low': min(prev_candle['Low'], current_candle['Low']),
                    'timestamp': current_candle.name
                })
        return order_blocks

    def identify_liquidity_zones(self, price_data: pd.DataFrame) -> list:
        """
        Identify potential liquidity zones where stop-losses accumulate.
        """
        liquidity_zones = []
        for i in range(len(price_data)):
            current_candle = price_data.iloc[i]
            if current_candle['High'] == price_data['High'].rolling(window=5).max().iloc[i]:
                liquidity_zones.append({
                    'type': 'liquidity_zone_high',
                    'level': current_candle['High'],
                    'timestamp': current_candle.name
                })
            if current_candle['Low'] == price_data['Low'].rolling(window=5).min().iloc[i]:
                liquidity_zones.append({
                    'type': 'liquidity_zone_low',
                    'level': current_candle['Low'],
                    'timestamp': current_candle.name
                })
        return liquidity_zones

    def identify_kill_zones(self) -> str:
        """
        Identify current kill zone based on the time of day.
        """
        current_time = datetime.utcnow().time()
        london_open = datetime.strptime("08:00", "%H:%M").time()
        london_close = datetime.strptime("10:00", "%H:%M").time()
        ny_open = datetime.strptime("13:00", "%H:%M").time()
        ny_close = datetime.strptime("15:00", "%H:%M").time()

        if london_open <= current_time <= london_close:
            return "London Kill Zone"
        elif ny_open <= current_time <= ny_close:
            return "New York Kill Zone"
        else:
            return "Off Kill Zone"

    def confirm_entry(self, fvg: dict, ob: dict, liquidity_zones: list, price_data: pd.DataFrame, atr_value: float) -> Optional[dict]:
        """
        Confirm a trade entry signal using FVG, OB, and liquidity zones.
        """
        for i in range(len(price_data)):
            current_candle = price_data.iloc[i]
            if fvg['type'] == ob['type']:
                if fvg['type'] == 'bullish' and ob['low'] <= current_candle['Low'] <= fvg['high']:
                    entry_price = current_candle['Close']
                    stop_loss = ob['low'] - atr_value * self.strategy_parameters.get('atr_multiplier', 1.5)
                    take_profit = entry_price + 2 * (entry_price - stop_loss)
                    return {
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'type': 'bullish',
                        'timestamp': current_candle.name
                    }
                elif fvg['type'] == 'bearish' and ob['high'] >= current_candle['High'] >= fvg['low']:
                    entry_price = current_candle['Close']
                    stop_loss = ob['high'] + atr_value * self.strategy_parameters.get('atr_multiplier', 1.5)
                    take_profit = entry_price - 2 * (stop_loss - entry_price)
                    return {
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'type': 'bearish',
                        'timestamp': current_candle.name
                    }
        return None

    def execute_trade(self, symbol: str, entry_signal: dict):
        """
        Execute a trade based on the confirmed entry signal.
        """
        position_size = self.position_sizing.calculate_position_size(entry_signal['entry_price'], self.strategy_parameters)
        self.order_execution_engine.place_order(
            symbol=symbol,
            order_type=entry_signal['type'],
            entry_price=entry_signal['entry_price'],
            stop_loss=entry_signal['stop_loss'],
            take_profit=entry_signal['take_profit'],
            size=position_size
        )
        logger.info(f"Executed {entry_signal['type']} trade on {symbol} at price {entry_signal['entry_price']}")

    def generate_signals(self, symbol: str, timeframe: str, atr_value: float) -> list:
        """
        Generate buy/sell signals using ICT concepts (FVGs, OBs, and liquidity zones).
        """
        price_data = self.exchange_connector.get_historical_data(symbol, timeframe)
        fvgs = self.identify_fvg(price_data)
        order_blocks = self.identify_order_blocks(price_data)
        liquidity_zones = self.identify_liquidity_zones(price_data)

        signals = []
        for fvg in fvgs:
            for ob in order_blocks:
                # Confirm the entry signal based on FVGs, OBs, and liquidity zones
                entry_signal = self.confirm_entry(fvg, ob, liquidity_zones, price_data, atr_value)
                if entry_signal:
                    kill_zone = self.identify_kill_zones()
                    # Only trade during the Kill Zones (London or New York)
                    if kill_zone in ["London Kill Zone", "New York Kill Zone"]:
                        signals.append(entry_signal)
                        logger.info(f"Confirmed {entry_signal['type']} signal for {symbol} during {kill_zone}.")
        return signals

    def run_main_loop(self, symbols: List[str], portfolio_value: float, check_interval_minutes: int = 60):
        """
        Main loop for generating and executing trade signals based on the ICT strategy.
        
        :param symbols: List of symbols to monitor (e.g., ['EUR/USD', 'GBP/USD']).
        :param portfolio_value: Total portfolio value for position sizing.
        :param check_interval_minutes: Interval in minutes to check for new signals and trades.
        """
        while True:
            for symbol in symbols:
                # Fetch the latest data and calculate ATR
                price_data = self.exchange_connector.get_historical_data(symbol, '4h')  # Example timeframe
                atr_value = self.calculate_atr(price_data)

                # Generate trading signals based on ICT concepts
                signals = self.generate_signals(symbol, '4h', atr_value)

                # Execute trades based on generated signals
                for signal in signals:
                    self.execute_trade(symbol, signal)

            # Wait before the next iteration
            logger.info(f"Sleeping for {check_interval_minutes} minutes before next check.")
            time.sleep(check_interval_minutes * 60)

    def calculate_position_size(self, atr_value: float, portfolio_value: float) -> float:
        """
        Calculate the position size based on ATR value and the portfolio's risk settings.
        
        :param atr_value: The calculated ATR value.
        :param portfolio_value: The total value of the portfolio.
        :return: Calculated position size.
        """
        risk_percentage = self.strategy_parameters.get('risk_percentage', 0.01)
        risk_per_trade = portfolio_value * risk_percentage
        return risk_per_trade / atr_value

    def apply_stop_loss_and_take_profit(self, entry_price: float, position_type: str, atr_value: float) -> Dict[str, float]:
        """
        Apply stop-loss and take-profit levels based on ATR and position type.
        
        :param entry_price: The price at which the trade is entered.
        :param position_type: 'long' or 'short'.
        :param atr_value: ATR value for calculating risk levels.
        :return: A dictionary containing stop-loss and take-profit levels.
        """
        risk_reward_ratio = self.strategy_parameters.get('risk_reward_ratio', 2.0)
        stop_loss_pips = atr_value * self.strategy_parameters.get('atr_multiplier', 1.5)

        if position_type == 'long':
            stop_loss = entry_price - stop_loss_pips
            take_profit = entry_price + (stop_loss_pips * risk_reward_ratio)
        elif position_type == 'short':
            stop_loss = entry_price + stop_loss_pips
            take_profit = entry_price - (stop_loss_pips * risk_reward_ratio)
        else:
            raise ValueError("Invalid position type. Must be 'long' or 'short'.")
        
        return {'stop_loss': stop_loss, 'take_profit': take_profit}

    def monitor_position(self, trade_details: Dict[str, float]):
        """
        Monitor an open position and apply trailing stop-loss if necessary.
        
        :param trade_details: Dictionary containing trade information.
        :return: Updated trade details.
        """
        current_price = self.exchange_connector.get_current_price(trade_details['symbol'])
        trailing_stop_active = self.strategy_parameters.get('trailing_stop', False)
        trailing_stop_distance = self.strategy_parameters.get('trailing_stop_distance', 20)

        if trailing_stop_active:
            if trade_details['position_type'] == 'long':
                new_stop_loss = max(trade_details['stop_loss'], current_price - (trailing_stop_distance * 0.0001))
                if new_stop_loss != trade_details['stop_loss']:
                    trade_details['stop_loss'] = new_stop_loss
                    logger.info(f"Trailing stop-loss adjusted to {new_stop_loss} for long position.")
            elif trade_details['position_type'] == 'short':
                new_stop_loss = min(trade_details['stop_loss'], current_price + (trailing_stop_distance * 0.0001))
                if new_stop_loss != trade_details['stop_loss']:
                    trade_details['stop_loss'] = new_stop_loss
                    logger.info(f"Trailing stop-loss adjusted to {new_stop_loss} for short position.")
        
        if (trade_details['position_type'] == 'long' and current_price >= trade_details['take_profit']) or \
           (trade_details['position_type'] == 'short' and current_price <= trade_details['take_profit']):
            logger.info(f"Take-profit reached for {trade_details['position_type']} at {current_price}.")
            return self.close_trade(current_price, trade_details)
        elif (trade_details['position_type'] == 'long' and current_price <= trade_details['stop_loss']) or \
             (trade_details['position_type'] == 'short' and current_price >= trade_details['stop_loss']):
            logger.info(f"Stop-loss triggered for {trade_details['position_type']} at {current_price}.")
            return self.close_trade(current_price, trade_details)

        return trade_details

    def close_trade(self, closing_price: float, trade_details: Dict[str, float]) -> Dict[str, float]:
        """
        Close an open trade and calculate realized profit or loss.
        
        :param closing_price: The price at which the trade is closed.
        :param trade_details: The details of the open trade.
        :return: Updated trade details with profit/loss information.
        """
        realized_profit = (closing_price - trade_details['entry_price']) * trade_details['position_size']
        if trade_details['position_type'] == 'short':
            realized_profit = -realized_profit

        trade_details['status'] = 'closed'
        trade_details['close_price'] = closing_price
        trade_details['realized_profit'] = realized_profit

        logger.info(f"Trade closed at {closing_price} with a realized profit/loss of {realized_profit}.")
        return trade_details
