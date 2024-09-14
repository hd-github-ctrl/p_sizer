import json
import os
import threading
import logging
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime, timedelta

# Logger configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler('fvg_order_block_strategy_parameters.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class ParameterManager:
    """A base class to manage saving, loading, and validation of settings for the strategy, now including ICT, FVG, and OB parameters."""

    def __init__(self, config_path: str = 'config/'):
        self.config_path = config_path
        os.makedirs(self.config_path, exist_ok=True)
        self.settings = {}

    def load_settings(self, filename: str) -> Dict[str, Any]:
        """Loads settings from a specified file."""
        template_path = os.path.join(self.config_path, filename)
        try:
            with open(template_path, 'r') as f:
                self.settings = json.load(f)
            logger.info(f"Settings loaded from {template_path}")
        except FileNotFoundError:
            logger.warning(f"Settings file {filename} not found. Using default settings.")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding settings file {filename}: {e}")
        return self.settings

    def save_settings(self, filename: str, settings: Optional[Dict[str, Any]] = None) -> None:
        """Saves the provided settings to a specified file."""
        if settings:
            self.settings = settings
        template_path = os.path.join(self.config_path, filename)
        try:
            with open(template_path, 'w') as f:
                json.dump(self.settings, f, indent=4)
            logger.info(f"Settings successfully saved to {template_path}")
        except IOError as e:
            logger.error(f"Failed to save settings to {template_path}: {e}")

    def validate_settings(self, settings: Dict[str, Any]) -> bool:
        """Validates settings based on predefined criteria."""
        raise NotImplementedError("This method should be implemented by subclasses.")


class CustomizableInputs(ParameterManager):
    """Handles general customizable inputs for the strategy configuration, including FVGs, OBs, and ICT concepts."""

    def __init__(self, config_file: str = 'default_config.json'):
        super().__init__()
        self.config_file = config_file
        self.settings = self.default_settings()
        self.load_settings(self.config_file)

    def default_settings(self) -> Dict[str, Any]:
        """Default settings for the strategy."""
        return {
            "risk_percentage": 1.0,  # Risk per trade
            "leverage": 1.0,  # Leverage used in the strategy
            "stop_loss_atr_multiplier": 1.5,  # Multiplier for ATR-based stop-loss
            "take_profit_atr_multiplier": 2.0,  # Multiplier for ATR-based take-profit
            "fvg_min_pip_size": 10,  # Minimum pip size to identify FVG
            "order_block_min_candle_count": 3,  # Minimum candles to form OB
            "atr_period": 14,  # Period for ATR calculation
            "dca_max_entries": 3,  # Max DCA entries
            "dca_spacing_atr": 1.0  # ATR spacing for DCA entries
        }

    def validate_settings(self, settings: Dict[str, Any]) -> bool:
        """Validates the provided settings for the strategy."""
        valid = True
        error_messages = []

        # Validation rules for FVG and OB concepts
        if not (0.01 <= settings.get('risk_percentage', 0) <= 5.0):
            error_messages.append("Risk percentage must be between 0.01 and 5.0.")
        if not (1 <= settings.get('leverage', 0) <= 500):
            error_messages.append("Leverage must be between 1 and 500.")
        if settings.get('stop_loss_atr_multiplier', 0) <= 0:
            error_messages.append("Stop-loss ATR multiplier must be greater than 0.")
        if settings.get('take_profit_atr_multiplier', 0) <= 0:
            error_messages.append("Take-profit ATR multiplier must be greater than 0.")
        if settings.get('fvg_min_pip_size', 0) < 1:
            error_messages.append("Fair Value Gap minimum pip size must be at least 1.")
        if settings.get('order_block_min_candle_count', 0) < 1:
            error_messages.append("Order Block minimum candle count must be at least 1.")
        if not (5 <= settings.get('atr_period', 0) <= 50):
            error_messages.append("ATR period must be between 5 and 50.")
        if not (1 <= settings.get('dca_max_entries', 0) <= 10):
            error_messages.append("DCA max entries must be between 1 and 10.")
        if settings.get('dca_spacing_atr', 0) <= 0:
            error_messages.append("DCA spacing ATR must be greater than 0.")

        if error_messages:
            valid = False
            for msg in error_messages:
                logger.error(msg)

        return valid


class DCAParameters(ParameterManager):
    """Manages the settings and calculations related to Dollar-Cost Averaging (DCA), focusing on optimal spacing in FVG retracements."""

    def __init__(self, config_path: str = 'config/'):
        super().__init__(config_path)
        self.settings = {
            "dca_enabled": True,  # Enable DCA
            "dca_interval": 1.0,  # Interval for DCA entries in terms of ATR
            "max_dca_steps": 5,  # Maximum DCA steps allowed
            "dca_multiplier": 1.5,  # Multiplier for each DCA entry size
            "dca_threshold": 0.5  # Threshold before triggering DCA entries
        }

    def calculate_dca_position_size(self, base_position_size: float, step: int) -> float:
        """Calculates the position size for the given DCA step."""
        if step > self.settings["max_dca_steps"]:
            raise ValueError(f"Step {step} exceeds max DCA steps of {self.settings['max_dca_steps']}")
        return base_position_size * (self.settings["dca_multiplier"] ** (step - 1))


class DynamicFieldsManager:
    """Manages dynamic adjustments of strategy parameters based on market conditions, especially volatility, FVG, and OB insights."""

    def __init__(self, initial_params: Dict[str, Any], update_interval: float = 1.0):
        self.params = initial_params
        self.update_interval = update_interval
        self.update_functions = {}
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.start()

    def register_update_function(self, param_name: str, update_func: Callable[[Any], Any]):
        """Registers a dynamic update function for a specific parameter."""
        with self.lock:
            self.update_functions[param_name] = update_func

    def _update_loop(self):
        """Loop for updating parameters dynamically based on registered functions."""
        while not self.stop_event.is_set():
            with self.lock:
                for param_name, update_func in self.update_functions.items():
                    try:
                        new_value = update_func(self.params.get(param_name))
                        if new_value is not None:
                            self.params[param_name] = new_value
                            logger.info(f"Updated {param_name} to {new_value}")
                    except Exception as e:
                        logger.error(f"Error updating {param_name}: {e}")
            threading.Event().wait(self.update_interval)

    def stop(self):
        """Stops the dynamic update loop."""
        self.stop_event.set()
        self.update_thread.join()


class MaxOpenTrades:
    """Manages the number of open trades and ensures that the maximum limit is enforced."""

    def __init__(self, max_trades: int, exchange_connector):
        self.max_trades = max_trades
        self.exchange_connector = exchange_connector

    def get_open_trades(self) -> List[str]:
        """Gets a list of currently open trades."""
        try:
            return self.exchange_connector.get_open_trades()
        except Exception as e:
            logger.error(f"Error retrieving open trades: {e}")
            return []

    def enforce_limit(self) -> bool:
        """Enforces the maximum open trades limit."""
        open_trades = self.get_open_trades()
        if len(open_trades) >= self.max_trades:
            logger.warning(f"Max open trades limit reached: {len(open_trades)}/{self.max_trades}")
            return False
        return True


class StrategyParameters:
    """Manages the main strategy parameters and dynamically adjusts them based on market conditions like FVGs and OBs."""

    def __init__(self, initial_parameters: Dict[str, Any]):
        self.parameters = initial_parameters
        self.default_parameters = initial_parameters.copy()

    def get_parameter(self, param_name: str) -> Any:
        """Gets the value of a specific parameter."""
        return self.parameters.get(param_name)

    def set_parameter(self, param_name: str, value: Any):
        """Sets the value of a specific parameter."""
        self.parameters[param_name] = value

    def reset_parameters(self):
        """Resets all parameters to their default values."""
        self.parameters = self.default_parameters.copy()

    def apply_market_conditions(self, market_conditions: Dict[str, Any]):
        """Applies dynamic adjustments to parameters based on market conditions, such as FVG retracements and OB reactions."""
        if 'volatility' in market_conditions:
            if market_conditions['volatility'] > self.get_parameter('volatility_threshold'):
                self.set_parameter('spread_threshold', self.get_parameter('spread_threshold') * 1.2)
                self.set_parameter('stop_loss_atr_multiplier', self.get_parameter('stop_loss_atr_multiplier') * 1.5)
        if 'fvg' in market_conditions:
            self.set_parameter('fvg_min_pip_size', market_conditions['fvg']['min_pip_size'])


class TemplateManager(ParameterManager):
    """Manages strategy templates for saving, loading, and applying specific parameter configurations."""

    def __init__(self, template_directory: str = './templates'):
        super().__init__(template_directory)

    def save_template(self, template_name: str, strategy_parameters: Dict[str, Any]) -> None:
        """Saves strategy parameters as a template."""
        self.save_settings(f"{template_name}.json", strategy_parameters)

    def load_template(self, template_name: str) -> Optional[Dict[str, Any]]:
        """Loads a strategy template."""
        return self.load_settings(f"{template_name}.json")

    def apply_template(self, template_name: str, strategy_manager) -> bool:
        """Applies a strategy template to the strategy manager."""
        strategy_parameters = self.load_template(template_name)
        if strategy_parameters:
            strategy_manager.update_parameters(strategy_parameters)
            return True
        return False
