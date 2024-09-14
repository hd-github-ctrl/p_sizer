import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler('position_exit.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class StopLossExit:
    """
    Handles the application and monitoring of stop-loss levels based on ATR or static settings.
    """

    def __init__(self, strategy_parameters: dict):
        """
        Initializes the StopLossExit class.

        :param strategy_parameters: Dictionary containing settings for stop-loss and ATR parameters.
        """
        self.strategy_parameters = strategy_parameters
        self.atr_period = strategy_parameters.get('atr_period', 14)
        self.atr_multiplier = strategy_parameters.get('atr_multiplier', 1.5)
        self.trailing_stop_active = strategy_parameters.get('trailing_stop', True)
        self.trailing_stop_distance = strategy_parameters.get('trailing_stop_distance', 20)
        logger.info("StopLossExit initialized with parameters: %s", strategy_parameters)

    def calculate_atr(self, price_data: pd.DataFrame) -> float:
        """
        Calculates the Average True Range (ATR) for volatility adjustment in stop-loss levels.

        :param price_data: Historical price data (OHLC).
        :return: The current ATR value.
        """
        high_low = price_data['High'] - price_data['Low']
        high_close = np.abs(price_data['High'] - price_data['Close'].shift())
        low_close = np.abs(price_data['Low'] - price_data['Close'].shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=self.atr_period).mean().iloc[-1]
        logger.debug(f"Calculated ATR: {atr}")
        return atr

    def calculate_initial_stop_loss(self, entry_price: float, price_data: pd.DataFrame, position_type: str) -> float:
        """
        Calculate the initial stop-loss level based on ATR.

        :param entry_price: The price at which the trade was entered.
        :param price_data: DataFrame containing historical price data (OHLC).
        :param position_type: 'long' for buying and 'short' for selling.
        :return: The calculated stop-loss price.
        """
        atr = self.calculate_atr(price_data)
        stop_loss_price = entry_price - atr * self.atr_multiplier if position_type == 'long' else entry_price + atr * self.atr_multiplier
        logger.info(f"Calculated initial stop-loss at {stop_loss_price} for {position_type} position.")
        return stop_loss_price

    def adjust_stop_loss(self, current_price: float, stop_loss_price: float, position_type: str) -> float:
        """
        Adjust the stop-loss based on trailing stop settings or other conditions.

        :param current_price: The current market price.
        :param stop_loss_price: The original stop-loss price.
        :param position_type: 'long' or 'short'.
        :return: The adjusted stop-loss price.
        """
        if not self.trailing_stop_active:
            logger.debug("Trailing stop not active. Returning original stop-loss.")
            return stop_loss_price

        if position_type == 'long':
            new_stop_loss = max(stop_loss_price, current_price - self.trailing_stop_distance)
        elif position_type == 'short':
            new_stop_loss = min(stop_loss_price, current_price + self.trailing_stop_distance)
        else:
            logger.error(f"Invalid position type: {position_type}. Must be 'long' or 'short'.")
            raise ValueError(f"Invalid position type: {position_type}")

        if new_stop_loss != stop_loss_price:
            logger.info(f"Trailing stop adjusted stop-loss to {new_stop_loss} from {stop_loss_price}.")
        return new_stop_loss

    def execute_stop_loss_exit(self, current_price: float, stop_loss_price: float, trade_details: dict) -> dict:
        """
        Execute the trade exit if the stop-loss level is hit.

        :param current_price: The current market price.
        :param stop_loss_price: The stop-loss price.
        :param trade_details: A dictionary containing details of the executed trade including entry price and position type.
        :return: Updated trade details with exit status and performance metrics.
        """
        if (trade_details['position_type'] == 'long' and current_price <= stop_loss_price) or \
           (trade_details['position_type'] == 'short' and current_price >= stop_loss_price):
            logger.info(f"Stop-loss triggered for {trade_details['position_type']} position at {current_price}.")
            return {'status': 'closed', 'close_price': current_price}

        logger.debug(f"Stop-loss not triggered. Trade remains open.")
        return {'status': 'open', 'current_price': current_price}


class FVGExitStrategy:
    """
    Handles exits based on Fair Value Gaps (FVGs). Trades will close at the completion of the FVG.
    """

    def __init__(self, strategy_parameters: dict):
        """
        Initializes FVGExitStrategy.

        :param strategy_parameters: Strategy-specific parameters for FVG exits.
        """
        self.strategy_parameters = strategy_parameters
        logger.info("FVGExitStrategy initialized with parameters: %s", strategy_parameters)

    def check_fvg_completion(self, fvg: dict, current_price: float, trade_details: dict) -> bool:
        """
        Check if the Fair Value Gap (FVG) has completed, signaling an exit.

        :param fvg: The Fair Value Gap details.
        :param current_price: Current market price.
        :param trade_details: The details of the trade being monitored.
        :return: True if the FVG is completed and trade should be exited.
        """
        if trade_details['position_type'] == 'long' and current_price >= fvg['high']:
            logger.info(f"FVG completed for long position. Current price: {current_price}, FVG high: {fvg['high']}")
            return True
        elif trade_details['position_type'] == 'short' and current_price <= fvg['low']:
            logger.info(f"FVG completed for short position. Current price: {current_price}, FVG low: {fvg['low']}")
            return True

        logger.debug(f"FVG not completed. Trade remains open.")
        return False


class OrderBlockExitStrategy:
    """
    Exit strategy based on Order Blocks (OBs), where trades close after the price reaches the opposite OB.
    """

    def __init__(self, strategy_parameters: dict):
        """
        Initializes OrderBlockExitStrategy.

        :param strategy_parameters: Strategy-specific parameters for OB exits.
        """
        self.strategy_parameters = strategy_parameters
        logger.info("OrderBlockExitStrategy initialized with parameters: %s", strategy_parameters)

    def check_order_block_exit(self, order_block: dict, current_price: float, trade_details: dict) -> bool:
        """
        Check if the price has reached the opposite Order Block (OB) for an exit.

        :param order_block: The Order Block details.
        :param current_price: Current market price.
        :param trade_details: The details of the trade being monitored.
        :return: True if the opposite OB has been reached and trade should be exited.
        """
        if trade_details['position_type'] == 'long' and current_price >= order_block['high']:
            logger.info(f"Order Block exit for long position at {current_price}. OB high: {order_block['high']}")
            return True
        elif trade_details['position_type'] == 'short' and current_price <= order_block['low']:
            logger.info(f"Order Block exit for short position at {current_price}. OB low: {order_block['low']}")
            return True

        logger.debug(f"Order Block not reached. Trade remains open.")
        return False


class EmergencyExitStrategy:
    """
    Emergency exit strategy triggered by unexpected market conditions such as high volatility or significant news events.
    """

    def __init__(self, strategy_parameters: dict, news_feed):
        """
        Initializes the EmergencyExitStrategy with required components.

        :param strategy_parameters: Strategy-specific parameters for emergency exits.
        :param news_feed: A source for high-impact news events that could trigger an exit.
        """
        self.strategy_parameters = strategy_parameters
        self.news_feed = news_feed
        logger.info("EmergencyExitStrategy initialized with parameters: %s", strategy_parameters)

    def check_for_emergency_exit(self, trade_details: dict, current_price: float) -> bool:
        """
        Check if an emergency exit should be triggered due to high-impact news or extreme volatility.

        :param trade_details: The details of the trade being monitored.
        :param current_price: Current market price.
        :return: True if an emergency exit should be triggered.
        """
        # Check for high-impact news
        latest_news = self.news_feed.get_latest_news()
        for news_item in latest_news:
            if news_item['impact'] == 'high':
                logger.info(f"Emergency exit triggered due to high-impact news: {news_item['headline']}")
                return True

        # Check for extreme volatility (could be defined by ATR or standard deviation spikes)
        volatility = self.calculate_volatility(trade_details['symbol'])
        if volatility > self.strategy_parameters.get('volatility_threshold', 2.0):
            logger.info(f"Emergency exit triggered due to extreme volatility. Volatility: {volatility}")
            return True

        return False

    def calculate_volatility(self, symbol: str) -> float:
        """
        Calculate the volatility of the asset using standard deviation or ATR.

        :param symbol: The trading symbol.
        :return: The calculated volatility.
        """
        # Placeholder: use ATR or other volatility measures
        return np.random.uniform(1, 3)


class MultipleOfRiskExitStrategy:
    """
    Exit strategy based on achieving a predefined multiple of the initial risk.
    """

    def __init__(self, strategy_parameters: dict):
        """
        Initializes MultipleOfRiskExitStrategy.

        :param strategy_parameters: Strategy-specific parameters including risk-reward settings.
        """
        self.strategy_parameters = strategy_parameters
        logger.info("MultipleOfRiskExitStrategy initialized with parameters: %s", strategy_parameters)

    def calculate_exit_level(self, entry_price: float, risk_per_trade: float, position_type: str) -> float:
        """
        Calculate the exit level based on a multiple of the risk taken.

        :param entry_price: Entry price of the trade.
        :param risk_per_trade: Risk taken per trade (distance between entry and stop-loss).
        :param position_type: 'long' or 'short'.
        :return: Calculated exit price.
        """
        risk_reward_ratio = self.strategy_parameters.get('risk_reward_ratio', 2.0)
        exit_price = entry_price + (risk_per_trade * risk_reward_ratio) if position_type == 'long' else entry_price - (risk_per_trade * risk_reward_ratio)
        logger.info(f"Calculated exit level at {exit_price} for {position_type} position.")
        return exit_price

    def execute_exit(self, current_price: float, exit_level: float, trade_details: dict) -> dict:
        """
        Execute the exit if the current price reaches the calculated exit level.

        :param current_price: Current market price.
        :param exit_level: The target price at which to exit the trade.
        :param trade_details: Details of the executed trade including entry price and position type.
        :return: Updated trade details after exit.
        """
        if (trade_details['position_type'] == 'long' and current_price >= exit_level) or \
           (trade_details['position_type'] == 'short' and current_price <= exit_level):
            logger.info(f"Exit level reached at {current_price}. Closing position.")
            return {'status': 'closed', 'close_price': current_price}
        
        return {'status': 'open', 'current_price': current_price}


class BreakEvenExitStrategy:
    """
    Implements break-even exits to protect against losses after a position has moved in the desired direction.
    """

    def __init__(self, strategy_parameters: dict):
        """
        Initializes the BreakEvenExitStrategy class.

        :param strategy_parameters: Strategy-specific parameters for break-even exits.
        """
        self.strategy_parameters = strategy_parameters
        logger.info("BreakEvenExitStrategy initialized with parameters: %s", strategy_parameters)

    def check_break_even(self, current_price: float, entry_price: float, stop_loss_price: float, position_type: str) -> float:
        """
        Adjust the stop-loss to break-even once the price has moved in favor of the trade.

        :param current_price: The current market price.
        :param entry_price: The price at which the trade was entered.
        :param stop_loss_price: The current stop-loss price.
        :param position_type: 'long' or 'short'.
        :return: The updated stop-loss price if break-even is reached.
        """
        if position_type == 'long' and current_price >= entry_price + self.strategy_parameters.get('breakeven_buffer', 1.5) * (entry_price - stop_loss_price):
            logger.info(f"Break-even reached for long position. Adjusting stop-loss to {entry_price}.")
            return entry_price
        elif position_type == 'short' and current_price <= entry_price - self.strategy_parameters.get('breakeven_buffer', 1.5) * (stop_loss_price - entry_price):
            logger.info(f"Break-even reached for short position. Adjusting stop-loss to {entry_price}.")
            return entry_price

        logger.debug("Break-even not reached. Keeping stop-loss unchanged.")
        return stop_loss_price


class PositionExitManager:
    """
    Main class to manage the various exit strategies for positions, including stop-loss, take-profit, trailing stop, and ICT concepts.
    """

    def __init__(self, strategy_parameters, market_data_fetcher, order_execution_engine, news_feed):
        """
        Initializes the PositionExitManager with all exit strategies.

        :param strategy_parameters: Parameters for exit strategies.
        :param market_data_fetcher: Instance to fetch market data.
        :param order_execution_engine: Instance to execute orders.
        :param news_feed: Instance to fetch high-impact news events.
        """
        self.stop_loss_exit = StopLossExit(strategy_parameters)
        self.fvg_exit = FVGExitStrategy(strategy_parameters)
        self.ob_exit = OrderBlockExitStrategy(strategy_parameters)
        self.emergency_exit = EmergencyExitStrategy(strategy_parameters, news_feed)
        self.break_even_exit = BreakEvenExitStrategy(strategy_parameters)
        self.multiple_risk_exit = MultipleOfRiskExitStrategy(strategy_parameters)
        self.market_data_fetcher = market_data_fetcher
        self.order_execution_engine = order_execution_engine
        self.positions = {}

    def set_positions(self, positions):
        """
        Set the current open positions to be managed.
        
        :param positions: Dictionary of open positions.
        """
        self.positions = positions
        logger.info(f"Positions set for exit management: {positions}")

    def manage_exits(self):
        """
        Manage exits for all open positions, applying different strategies based on the market conditions.
        """
        for position_id, position in self.positions.items():
            market_data = self.market_data_fetcher.get_current_market_data(position['symbol'])
            current_price = market_data['price']
            logger.info(f"Managing exit for position {position_id} with current price {current_price}")

            # Stop-loss handling
            stop_loss_price = self.stop_loss_exit.adjust_stop_loss(current_price, position['stop_loss'], position['position_type'])
            self.positions[position_id]['stop_loss'] = stop_loss_price
            if self.stop_loss_exit.execute_stop_loss_exit(current_price, stop_loss_price, position)['status'] == 'closed':
                self.exit_position(position_id, 'stop_loss')
                continue

            # Break-even handling
            break_even_stop = self.break_even_exit.check_break_even(current_price, position['entry_price'], stop_loss_price, position['position_type'])
            self.positions[position_id]['stop_loss'] = break_even_stop

            # FVG completion check
            if self.fvg_exit.check_fvg_completion(position['fvg'], current_price, position):
                self.exit_position(position_id, 'fvg_completion')
                continue

            # Order Block handling
            if self.ob_exit.check_order_block_exit(position['order_block'], current_price, position):
                self.exit_position(position_id, 'order_block_completion')
                continue

            # Emergency exit handling
            if self.emergency_exit.check_for_emergency_exit(position, current_price):
                self.exit_position(position_id, 'emergency_exit')
                continue

    def exit_position(self, position_id, exit_reason):
        """
        Exit the position based on the given reason.

        :param position_id: The position identifier.
        :param exit_reason: Reason for exiting (e.g., stop_loss, fvg_completion).
        """
        logger.info(f"Exiting position {position_id} due to {exit_reason}")
        self.order_execution_engine.close_position(position_id)
        # Remove the position from the list
        del self.positions[position_id]
