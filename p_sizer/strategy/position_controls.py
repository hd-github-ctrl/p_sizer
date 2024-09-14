import logging
import time
import numpy as np
from exchange_connector import ExchangeConnector  # Assuming this is the connector class for the exchange
from risk_management import RiskManagement  # Assuming this handles risk calculations and checks

class LeverageControl:
    """
    Manages the leverage of trading positions dynamically based on portfolio risk and market volatility.
    It ensures that leverage stays within the allowed limits and adjusts leverage as needed.
    """
    
    def __init__(self, exchange_connector: ExchangeConnector, risk_management: RiskManagement, max_leverage: int = 500, min_leverage: int = 1):
        """
        Initializes LeverageControl for managing dynamic leverage.

        :param exchange_connector: Instance of ExchangeConnector to interact with the exchange.
        :param risk_management: Instance of RiskManagement to enforce risk-based leverage adjustments.
        :param max_leverage: Maximum allowable leverage.
        :param min_leverage: Minimum allowable leverage.
        """
        self.exchange_connector = exchange_connector
        self.risk_management = risk_management
        self.max_leverage = max_leverage
        self.min_leverage = min_leverage
        self.current_leverage = min_leverage
        self.logger = self.setup_logging()

        self.logger.info("LeverageControl initialized.")

    def setup_logging(self) -> logging.Logger:
        """Sets up the logging configuration for LeverageControl."""
        logging.basicConfig(filename='leverage_control.log',
                            level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        return logging.getLogger('LeverageControl')

    def adjust_leverage(self):
        """
        Adjusts the leverage dynamically based on portfolio risk and market volatility.
        """
        portfolio_risk = self.risk_management.calculate_portfolio_risk()
        market_volatility = self.exchange_connector.get_market_volatility()

        new_leverage = self.calculate_leverage(portfolio_risk, market_volatility)

        if new_leverage != self.current_leverage:
            self.logger.info(f"Adjusting leverage from {self.current_leverage} to {new_leverage}.")
            self.current_leverage = new_leverage
            self.exchange_connector.set_leverage(new_leverage)

    def calculate_leverage(self, portfolio_risk: float, market_volatility: float) -> int:
        """
        Calculates the appropriate leverage based on current portfolio risk and market volatility.

        :param portfolio_risk: Current risk level of the portfolio as a percentage of the total portfolio value.
        :param market_volatility: Current market volatility.
        :return: The calculated leverage value.
        """
        self.logger.info(f"Calculating leverage with portfolio risk: {portfolio_risk}% and market volatility: {market_volatility}.")

        if portfolio_risk > 0.2:  # High risk
            leverage = max(self.min_leverage, int(self.max_leverage * 0.1))
        elif market_volatility > 0.05:  # High volatility
            leverage = max(self.min_leverage, int(self.max_leverage * 0.5))
        else:
            leverage = self.max_leverage

        self.logger.info(f"Calculated leverage: {leverage}.")
        return leverage

    def enforce_leverage_limits(self):
        """
        Ensures that leverage is within the maximum and minimum allowed limits.
        """
        if self.current_leverage > self.max_leverage:
            self.logger.warning(f"Leverage {self.current_leverage} exceeds max leverage {self.max_leverage}, adjusting to max leverage.")
            self.current_leverage = self.max_leverage
        elif self.current_leverage < self.min_leverage:
            self.logger.warning(f"Leverage {self.current_leverage} is below min leverage {self.min_leverage}, adjusting to min leverage.")
            self.current_leverage = self.min_leverage

        self.exchange_connector.set_leverage(self.current_leverage)
        self.logger.info(f"Leverage limits enforced, current leverage: {self.current_leverage}.")

    def monitor_and_adjust(self, interval: int = 60):
        """
        Monitors and adjusts leverage at specified intervals. This function is intended to be run in a loop.

        :param interval: Time interval in seconds between each leverage adjustment check.
        """
        self.logger.info(f"Starting leverage monitoring every {interval} seconds.")
        while True:
            try:
                self.adjust_leverage()
                self.enforce_leverage_limits()
            except Exception as e:
                self.logger.error(f"Error during leverage adjustment: {e}")
            time.sleep(interval)

    def on_order_submission(self, order_value: float, *args, **kwargs):
        """
        Adjusts leverage before submitting large orders that may affect portfolio risk.

        :param order_value: The value of the order being placed.
        """
        portfolio_value = self.risk_management.get_portfolio_value()
        if order_value > 0.05 * portfolio_value:  # If the order is large, adjust leverage.
            self.logger.info(f"Significant order value detected: {order_value}, adjusting leverage.")
            self.adjust_leverage()

        order_submission_function = kwargs.get('order_submission_function')
        if callable(order_submission_function):
            order_submission_function(*args, **kwargs)
            self.logger.info("Order submitted with adjusted leverage.")

    def override_leverage(self, leverage: int):
        """
        Manually overrides the leverage to handle extreme market conditions.

        :param leverage: The manually set leverage value.
        """
        self.logger.warning(f"Manual leverage override to {leverage}.")
        self.current_leverage = max(self.min_leverage, min(leverage, self.max_leverage))
        self.exchange_connector.set_leverage(self.current_leverage)
        self.logger.info(f"Leverage manually overridden to {self.current_leverage}.")


class VolatilityControl:
    """
    Manages position sizing based on market volatility, adjusting position sizes dynamically.
    """

    def __init__(self, config: Dict[str, float]):
        """
        Initializes VolatilityControl with configuration parameters.

        :param config: Dictionary containing volatility control parameters.
        """
        self.max_volatility = config.get('max_volatility', 0.02)
        self.min_volatility = config.get('min_volatility', 0.005)
        self.volatility_window = config.get('volatility_window', 20)
        self.position_size_adjustment_factor = config.get('position_size_adjustment_factor', 0.5)
        self.volatility = None
        logging.info("VolatilityControl initialized.")

    def calculate_volatility(self, price_data: np.ndarray) -> float:
        """
        Calculates the market volatility based on historical price data.

        :param price_data: Numpy array of historical price data.
        :return: The calculated annualized volatility.
        """
        if len(price_data) < self.volatility_window:
            raise ValueError("Not enough data to calculate volatility.")

        log_returns = np.diff(np.log(price_data))
        volatility = np.std(log_returns) * np.sqrt(252)  # Annualized volatility
        self.volatility = volatility
        logging.info(f"Calculated volatility: {volatility:.5f}")
        return volatility

    def adjust_position_size(self, base_position_size: float) -> float:
        """
        Adjusts position size based on current market volatility.

        :param base_position_size: The initial position size before adjustment.
        :return: The adjusted position size.
        """
        if self.volatility is None:
            raise RuntimeError("Volatility has not been calculated. Run calculate_volatility() first.")

        if self.volatility > self.max_volatility:
            adjusted_position_size = base_position_size * self.position_size_adjustment_factor
            logging.info(f"High volatility detected. Reducing position size to {adjusted_position_size:.5f}.")
        elif self.volatility < self.min_volatility:
            adjusted_position_size = base_position_size / self.position_size_adjustment_factor
            logging.info(f"Low volatility detected. Increasing position size to {adjusted_position_size:.5f}.")
        else:
            adjusted_position_size = base_position_size
            logging.info(f"Volatility within range. Position size remains {adjusted_position_size:.5f}.")

        return adjusted_position_size

    def enforce_volatility_limits(self, price_data: np.ndarray, base_position_size: float) -> float:
        """
        Enforces volatility limits and adjusts position size accordingly.

        :param price_data: Historical price data.
        :param base_position_size: Initial position size before adjustment.
        :return: Final adjusted position size based on volatility.
        """
        calculated_volatility = self.calculate_volatility(price_data)
        adjusted_position_size = self.adjust_position_size(base_position_size)

        logging.info(f"Volatility: {calculated_volatility:.5f} enforced with adjusted position size: {adjusted_position_size:.5f}.")
        return adjusted_position_size
