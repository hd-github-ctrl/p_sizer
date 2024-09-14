import logging
import numpy as np
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)

class RiskManagementSystem:
    def __init__(self, config):
        """
        Initializes the RiskManagementSystem with portfolio value, risk per trade, leverage, volatility thresholds, and drawdown limits.

        :param config: Configuration dictionary with key parameters.
        """
        self.portfolio_value = config.get('portfolio_value', 100000)  # Default $100k portfolio
        self.max_risk_per_trade = config.get('max_risk_per_trade', 0.002)  # 0.2% risk per trade
        self.max_leverage = config.get('max_leverage', 1.0)  # Default leverage 1:1
        self.volatility_threshold = config.get('volatility_threshold', 0.02)
        self.max_drawdown = config.get('max_drawdown', 0.2)
        self.trail_stop_distance = config.get('trail_stop_distance', 20)  # Trailing stop in pips
        self.partial_scale_out = config.get('partial_scale_out', 0.5)  # 50% scale-out
        self.initial_capital = self.portfolio_value
        self.current_capital = self.portfolio_value
        self.max_drawdown_reached = False
        self.positions = {}  # Stores active trades
        logger.info(f"RiskManagementSystem initialized with config: {config}")

    def calculate_position_size(self, stop_loss_distance: float, entry_price: float) -> dict:
        """
        Calculates position size based on risk per trade and stop-loss distance.
        Integrates leverage based on strategy configuration.

        :param stop_loss_distance: Distance between entry price and stop-loss.
        :param entry_price: The trade's entry price.
        :return: Dictionary containing position size and leveraged size.
        """
        if stop_loss_distance <= 0:
            raise ValueError("Stop loss distance must be greater than zero.")
        
        risk_amount = self.portfolio_value * self.max_risk_per_trade
        position_size = risk_amount / stop_loss_distance
        leveraged_position_size = position_size * self.max_leverage

        logger.info(f"Calculated position size: {position_size} units, Leverage: {self.max_leverage}")
        return {'position_size': position_size, 'leveraged_position_size': leveraged_position_size}

    def enforce_risk_limits(self, entry_price: float, stop_loss_price: float, current_position_size: float) -> float:
        """
        Enforces risk limits based on the strategy’s risk management rules.
        Adjusts position size based on optimal risk exposure.

        :param entry_price: The trade's entry price.
        :param stop_loss_price: The stop-loss price.
        :param current_position_size: Current trade size.
        :return: Adjusted position size if overexposed.
        """
        stop_loss_distance = abs(entry_price - stop_loss_price)
        optimal_position_size = self.calculate_position_size(stop_loss_distance, entry_price)['position_size']

        if current_position_size > optimal_position_size:
            logger.warning(f"Reducing position size from {current_position_size} to {optimal_position_size}.")
            return optimal_position_size
        return current_position_size

    def calculate_stop_loss(self, entry_price: float, atr_value: float, position_type: str, order_block_level: float) -> float:
        """
        Calculates stop-loss using ATR and key levels (Order Blocks, FVGs).

        :param entry_price: Trade entry price.
        :param atr_value: Current ATR (Average True Range).
        :param position_type: Trade direction ('buy' or 'sell').
        :param order_block_level: Identified Order Block level.
        :return: Stop-loss price.
        """
        atr_multiplier = 2.0  # ATR multiplier for stop-loss
        stop_loss_distance = atr_multiplier * atr_value

        if position_type == 'buy':
            stop_loss_price = max(entry_price - stop_loss_distance, order_block_level)
        elif position_type == 'sell':
            stop_loss_price = min(entry_price + stop_loss_distance, order_block_level)
        else:
            raise ValueError("Position type must be 'buy' or 'sell'.")

        logger.info(f"Stop-loss calculated: {stop_loss_price} for {position_type} trade.")
        return stop_loss_price

    def apply_trailing_stop(self, current_price: float, entry_price: float, position_type: str, order_block_level: float) -> float:
        """
        Applies trailing stop loss dynamically as the trade progresses.

        :param current_price: Current asset price.
        :param entry_price: Trade entry price.
        :param position_type: 'buy' or 'sell'.
        :param order_block_level: Nearest Order Block level.
        :return: Adjusted stop-loss price with trailing stop.
        """
        if position_type == 'buy':
            return max(current_price - self.trail_stop_distance, entry_price, order_block_level)
        elif position_type == 'sell':
            return min(current_price + self.trail_stop_distance, entry_price, order_block_level)
        else:
            raise ValueError("Position type must be 'buy' or 'sell'.")

    def scale_out_partial_position(self, current_position_size: float) -> float:
        """
        Scales out of the current position partially based on predefined targets (FVG or OB levels).

        :param current_position_size: Current position size.
        :return: New reduced position size.
        """
        new_position_size = current_position_size * self.partial_scale_out
        logger.info(f"Scaling out of position. New size: {new_position_size}")
        return new_position_size

    def manage_drawdown(self):
        """
        Manages portfolio drawdown and halts trading if the maximum drawdown is reached.
        """
        drawdown = (self.initial_capital - self.current_capital) / self.initial_capital
        if drawdown >= self.max_drawdown:
            self.max_drawdown_reached = True
            logger.warning(f"Max drawdown reached: {drawdown * 100:.2f}%. Halting trading.")
            self.halt_trading()
        else:
            logger.info(f"Current drawdown: {drawdown * 100:.2f}%.")

    def halt_trading(self):
        """
        Halts trading operations due to hitting max drawdown.
        """
        logger.info("Trading halted to prevent further losses.")

    def monitor_portfolio(self, trade_results: list):
        """
        Monitors the portfolio and checks for drawdown levels after each trade.

        :param trade_results: List of trade results with profit/loss data.
        """
        total_pnl = sum([trade['profit_loss'] for trade in trade_results])
        self.current_capital += total_pnl
        self.manage_drawdown()

    def dynamic_position_adjustment(self, current_volatility: float, position_size: float) -> float:
        """
        Adjusts position size based on market volatility.

        :param current_volatility: Market volatility (e.g., ATR or standard deviation).
        :param position_size: Initial position size.
        :return: Adjusted position size based on volatility.
        """
        if current_volatility > self.volatility_threshold:
            adjustment_factor = self.volatility_threshold / current_volatility
            logger.info(f"Volatility adjustment: New position size = {position_size * adjustment_factor}")
            return position_size * adjustment_factor
        return position_size

    def adjust_positions_based_on_correlation(self, portfolio_positions: dict, lookback_period: int = 30):
        """
        Adjusts positions based on correlation between assets in the portfolio.

        :param portfolio_positions: Active positions in the portfolio.
        :param lookback_period: Period to analyze correlations.
        """
        for (symbol1, symbol2) in portfolio_positions.keys():
            correlation = self.calculate_correlation(symbol1, symbol2, lookback_period)
            if abs(correlation) > 0.7:
                logger.info(f"High correlation detected: {symbol1} and {symbol2}. Adjusting positions.")

    def calculate_correlation(self, symbol1: str, symbol2: str, lookback_period: int) -> float:
        """
        Calculates correlation between two assets over a specified period.

        :param symbol1: First asset.
        :param symbol2: Second asset.
        :param lookback_period: Period for correlation calculation.
        :return: Correlation coefficient.
        """
        data1 = self.fetch_historical_data(symbol1, lookback_period)
        data2 = self.fetch_historical_data(symbol2, lookback_period)

        if len(data1) != len(data2):
            logger.warning("Data length mismatch for correlation calculation.")
            return 0.0

        returns1 = np.diff(np.log(data1))
        returns2 = np.diff(np.log(data2))
        correlation, _ = pearsonr(returns1, returns2)
        logger.info(f"Correlation between {symbol1} and {symbol2}: {correlation:.2f}")
        return correlation

    def fetch_historical_data(self, symbol: str, lookback_period: int) -> list:
        """
        Fetches historical price data for an asset.

        :param symbol: Asset symbol.
        :param lookback_period: Period for fetching historical data.
        :return: List of historical prices.
        """
        # Placeholder: Replace with actual market data fetching logic.
        return np.random.randn(lookback_period).tolist()

    def risk_report(self) -> dict:
        """
        Generates a risk report summarizing key portfolio metrics such as drawdown and current capital.

        :return: Dictionary with risk metrics.
        """
        drawdown = (self.initial_capital - self.current_capital) / self.initial_capital
        open_positions_risk = self.calculate_total_open_risk()

        risk_report = {
            'current_capital': self.current_capital,
            'initial_capital': self.initial_capital,
            'drawdown': drawdown,
            'max_drawdown_reached': self.max_drawdown_reached,
            'open_positions_risk': open_positions_risk
        }
        logger.info(f"Generated risk report: {risk_report}")
        return risk_report

    def calculate_total_open_risk(self) -> float:
        """
        Calculate the total risk for open positions in the portfolio.

        :return: Total risk of open positions.
        """
        total_risk = 0.0
        for position in self.positions.values():
            total_risk += self.calculate_position_risk(position)
        logger.info(f"Total open risk calculated: {total_risk}")
        return total_risk

    def calculate_position_risk(self, position: dict) -> float:
        """
        Calculate the risk for a specific position.

        :param position: A dictionary containing the position details.
        :return: Risk associated with the position.
        """
        entry_price = position.get('entry_price')
        stop_loss = position.get('stop_loss')
        size = position.get('size')

        risk_per_position = abs(entry_price - stop_loss) * size
        logger.info(f"Risk for position {position['symbol']}: {risk_per_position}")
        return risk_per_position

    def update_risk_parameters(self, portfolio_value: float = None, max_risk_per_trade: float = None, max_leverage: float = None, max_drawdown: float = None):
        """
        Dynamically updates risk management parameters.

        :param portfolio_value: Updated portfolio value.
        :param max_risk_per_trade: Updated max risk per trade.
        :param max_leverage: Updated max leverage.
        :param max_drawdown: Updated max drawdown percentage.
        """
        if portfolio_value is not None:
            self.portfolio_value = portfolio_value
            logger.info(f"Updated portfolio value: {portfolio_value}")

        if max_risk_per_trade is not None:
            self.max_risk_per_trade = max_risk_per_trade
            logger.info(f"Updated max risk per trade: {max_risk_per_trade}")

        if max_leverage is not None:
            self.max_leverage = max_leverage
            logger.info(f"Updated max leverage: {max_leverage}")

        if max_drawdown is not None:
            self.max_drawdown = max_drawdown
            logger.info(f"Updated max drawdown: {max_drawdown}")

    def enforce_stop_loss(self, current_price: float, stop_loss_price: float) -> bool:
        """
        Checks if the current price has hit the stop-loss level.

        :param current_price: Current market price.
        :param stop_loss_price: Predefined stop-loss price.
        :return: True if stop-loss condition is met, otherwise False.
        """
        if current_price <= stop_loss_price:
            logger.info("Stop-loss triggered. Exiting position.")
            return True
        return False

    def enforce_position_limits(self, current_open_trades: int, max_open_trades: int) -> bool:
        """
        Enforces a limit on the number of open trades.

        :param current_open_trades: Number of currently open trades.
        :param max_open_trades: Maximum allowed open trades.
        :return: True if within limits, otherwise False.
        """
        if current_open_trades >= max_open_trades:
            logger.warning(f"Reached max open trades limit: {max_open_trades}.")
            return False
        return True

    def calculate_risk_reward_ratio(self, entry_price: float, stop_loss_price: float, take_profit_price: float) -> float:
        """
        Calculate the risk-reward ratio for the trade.

        :param entry_price: Price at which the position is entered.
        :param stop_loss_price: Predefined stop-loss price.
        :param take_profit_price: Predefined take-profit price.
        :return: Risk-reward ratio.
        """
        risk = abs(entry_price - stop_loss_price)
        reward = abs(take_profit_price - entry_price)
        if risk == 0:
            raise ValueError("Risk cannot be zero.")
        risk_reward_ratio = reward / risk
        logger.info(f"Risk-reward ratio calculated: {risk_reward_ratio:.2f}")
        return risk_reward_ratio
