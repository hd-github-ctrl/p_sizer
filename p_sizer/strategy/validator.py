import logging
from typing import Any, Dict, List


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class Validator:
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the Validator with the given configuration.

        :param config: A dictionary containing the strategy's configuration parameters.
        """
        self.config = config
        logging.info("Validator initialized with configuration.")

    def validate(self):
        """
        Runs all validation checks.
        """
        self._validate_risk_settings()
        self._validate_spread_thresholds()
        self._validate_indicator_settings()
        self._validate_trade_management()
        self._validate_backtesting_requirements()
        logging.info("All validation checks passed successfully.")

    def _validate_risk_settings(self):
        """
        Validates the risk settings to ensure they are within acceptable bounds.
        """
        risk_percentage = self.config.get('risk_percentage', None)
        leverage = self.config.get('leverage', None)

        if risk_percentage is None or not (0.01 <= risk_percentage <= 0.02):
            raise ValidationError(f"Invalid risk percentage: {risk_percentage}. Must be between 1% and 2%.")

        if leverage is None or leverage < 1 or leverage > 500:
            raise ValidationError(f"Invalid leverage: {leverage}. Must be between 1 and 500.")

        logging.info(f"Risk settings validated: risk_percentage={risk_percentage}, leverage={leverage}")

    def _validate_spread_thresholds(self):
        """
        Validates the spread thresholds for placing orders.
        """
        min_spread = self.config.get('min_spread', None)
        max_spread = self.config.get('max_spread', None)

        if min_spread is None or min_spread <= 0:
            raise ValidationError(f"Invalid minimum spread: {min_spread}. Must be greater than 0.")

        if max_spread is None or max_spread <= min_spread:
            raise ValidationError(f"Invalid maximum spread: {max_spread}. Must be greater than minimum spread.")

        logging.info(f"Spread thresholds validated: min_spread={min_spread}, max_spread={max_spread}")

    def _validate_indicator_settings(self):
        """
        Validates the settings for indicators used in the strategy.
        """
        atr_multiplier = self.config.get('atr_multiplier', None)
        if atr_multiplier is None or atr_multiplier <= 0:
            raise ValidationError(f"Invalid ATR multiplier: {atr_multiplier}. Must be greater than 0.")

        liquidity_filters = self.config.get('liquidity_filters', {})
        min_depth = liquidity_filters.get('min_depth', None)
        max_volatility = liquidity_filters.get('max_volatility', None)

        if min_depth is None or min_depth <= 0:
            raise ValidationError(f"Invalid minimum order book depth: {min_depth}. Must be greater than 0.")

        if max_volatility is None or max_volatility <= 0:
            raise ValidationError(f"Invalid maximum volatility threshold: {max_volatility}. Must be greater than 0.")

        logging.info(f"Indicator settings validated: ATR multiplier={atr_multiplier}, "
                     f"min_depth={min_depth}, max_volatility={max_volatility}")

    def _validate_trade_management(self):
        """
        Validates the trade management settings.
        """
        scaling_options = self.config.get('scaling_options', {})
        hedging_settings = self.config.get('hedging_settings', {})

        scale_in_conditions = scaling_options.get('scale_in_conditions', None)
        scale_out_conditions = scaling_options.get('scale_out_conditions', None)

        if scale_in_conditions is None or not isinstance(scale_in_conditions, list):
            raise ValidationError(f"Invalid scale-in conditions: {scale_in_conditions}. Must be a list of conditions.")

        if scale_out_conditions is None or not isinstance(scale_out_conditions, list):
            raise ValidationError(f"Invalid scale-out conditions: {scale_out_conditions}. Must be a list of conditions.")

        correlation_threshold = hedging_settings.get('correlation_threshold', None)
        if correlation_threshold is None or not (0 <= correlation_threshold <= 1):
            raise ValidationError(f"Invalid correlation threshold for hedging: {correlation_threshold}. "
                                  f"Must be between 0 and 1.")

        logging.info("Trade management settings validated successfully.")

    def _validate_backtesting_requirements(self):
        """
        Ensures that backtesting requirements are met, if backtesting is to be performed.
        """
        backtesting_required = self.config.get('backtesting', False)
        if backtesting_required:
            historical_data = self.config.get('historical_data', None)
            if historical_data is None or not isinstance(historical_data, list) or len(historical_data) == 0:
                raise ValidationError("Backtesting is enabled, but no historical data provided.")
            logging.info("Backtesting requirements validated successfully.")

    def validate_inputs(self, inputs: Dict[str, Any]):
        """
        Validates that the strategy inputs are correct.

        :param inputs: Dictionary of inputs to validate.
        """
        for key, value in inputs.items():
            if value is None:
                raise ValidationError(f"Missing required input: {key}")
            if isinstance(value, (int, float)) and value < 0:
                raise ValidationError(f"Invalid input value for {key}: {value}. Must be non-negative.")
        logging.info("All strategy inputs validated successfully.")

    def validate_order_execution(self, order_details: Dict[str, Any]):
        """
        Validates the order execution details.

        :param order_details: A dictionary containing order details like price, volume, etc.
        """
        order_size = order_details.get('order_size', None)
        price = order_details.get('price', None)

        if order_size is None or order_size <= 0:
            raise ValidationError(f"Invalid order size: {order_size}. Must be greater than 0.")

        if price is None or price <= 0:
            raise ValidationError(f"Invalid price: {price}. Must be greater than 0.")

        logging.info(f"Order execution validated: order_size={order_size}, price={price}")

    def validate_strategy_template(self, template: Dict[str, Any]):
        """
        Validates that the strategy template is complete and correct.

        :param template: A dictionary representing a strategy template.
        """
        required_keys = ['risk_settings', 'spread_thresholds', 'indicator_settings', 'trade_management']

        for key in required_keys:
            if key not in template:
                raise ValidationError(f"Missing required template key: {key}")

        logging.info("Strategy template validated successfully.")
