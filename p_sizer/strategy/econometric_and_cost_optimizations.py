import numpy as np
import pandas as pd
from scipy.optimize import minimize
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransactionCostControl:
    """
    Handles transaction cost control and optimization.
    This ensures that transaction costs are minimized and factored into decision-making.
    """

    def __init__(self, fixed_cost=0, variable_cost_rate=0.001):
        """
        :param fixed_cost: Fixed transaction cost, e.g., exchange fees.
        :param variable_cost_rate: Variable cost rate, e.g., percentage-based fees on trade size.
        """
        self.fixed_cost = fixed_cost
        self.variable_cost_rate = variable_cost_rate

    def calculate_transaction_cost(self, trade_value):
        """
        Calculate the total transaction cost for a given trade.
        
        :param trade_value: The total value of the trade (quantity * price).
        :return: Total transaction cost.
        """
        total_cost = self.fixed_cost + (self.variable_cost_rate * trade_value)
        logger.info(f"Transaction cost calculated: {total_cost}")
        return total_cost

    def optimize_transaction_cost(self, trade_values):
        """
        Optimize transaction sizes to minimize total costs.
        
        :param trade_values: A list of trade values (price * quantity).
        :return: The trade size that minimizes the transaction cost.
        """
        def cost_function(x):
            return self.calculate_transaction_cost(x)

        optimized_trade_size = minimize(cost_function, x0=np.mean(trade_values)).x[0]
        logger.info(f"Optimized trade size for cost minimization: {optimized_trade_size}")
        return optimized_trade_size


class MarginControl:
    """
    Controls margin requirements and ensures proper margin utilization, including preventing margin calls.
    """

    def __init__(self, leverage=2, maintenance_margin_rate=0.25, margin_call_limit=0.20):
        """
        :param leverage: Leverage applied to the portfolio (default 2x).
        :param maintenance_margin_rate: The minimum maintenance margin rate.
        :param margin_call_limit: The minimum threshold to prevent a margin call (percentage).
        """
        self.leverage = leverage
        self.maintenance_margin_rate = maintenance_margin_rate
        self.margin_call_limit = margin_call_limit

    def calculate_margin_requirements(self, portfolio_value):
        """
        Calculate margin requirements based on portfolio value and leverage.
        
        :param portfolio_value: The total value of the portfolio.
        :return: Margin requirement (amount of capital needed to maintain positions).
        """
        margin_requirement = portfolio_value / self.leverage
        logger.info(f"Margin requirement calculated: {margin_requirement}")
        return margin_requirement

    def check_margin_safety(self, portfolio_value, account_balance):
        """
        Ensure margin safety by comparing the account balance to the margin requirement.
        Includes a limiter to prevent margin calls.
        
        :param portfolio_value: Total value of the portfolio.
        :param account_balance: Current available balance in the account.
        :return: Boolean indicating whether the margin is safe.
        """
        margin_requirement = self.calculate_margin_requirements(portfolio_value)
        safety_threshold = margin_requirement * (1 + self.maintenance_margin_rate)

        if account_balance < safety_threshold:
            logger.warning("Margin warning: Account balance is close to or below maintenance margin.")
            if account_balance < margin_requirement * (1 + self.margin_call_limit):
                logger.error("Critical: Margin call imminent. Trading activity should be stopped.")
                return False
        else:
            logger.info("Margin is safe.")

        return True

    def apply_margin_limiter(self, portfolio_value, account_balance):
        """
        Apply a limiter to prevent margin calls by reducing exposure or liquidating assets.
        
        :param portfolio_value: Total value of the portfolio.
        :param account_balance: Current account balance.
        :return: New portfolio value after applying the limiter.
        """
        if not self.check_margin_safety(portfolio_value, account_balance):
            new_portfolio_value = account_balance * self.leverage
            logger.info(f"Portfolio value adjusted to {new_portfolio_value} to prevent margin call.")
            return new_portfolio_value
        return portfolio_value


class EconometricsOptimization:
    """
    Apply econometric models to optimize trading strategies based on time-series forecasting and statistical methods.
    """

    def __init__(self, time_series_data):
        """
        :param time_series_data: Historical price data or other financial time series.
        """
        self.time_series_data = time_series_data

    def apply_arima_model(self, order=(5, 1, 0)):
        """
        Apply ARIMA (AutoRegressive Integrated Moving Average) model for time series forecasting.
        
        :param order: The (p,d,q) order of the ARIMA model.
        :return: Fitted ARIMA model and forecast.
        """
        model = ARIMA(self.time_series_data, order=order)
        arima_result = model.fit()
        forecast = arima_result.forecast(steps=5)  # Forecasting next 5 steps
        logger.info(f"ARIMA model applied. Forecast: {forecast}")
        return arima_result, forecast

    def apply_regression_model(self, features_data):
        """
        Apply linear regression model to forecast and explain relationships between multiple features.
        
        :param features_data: Additional data (e.g., macroeconomic factors, other assets).
        :return: Regression model and residuals.
        """
        reg_model = LinearRegression()
        reg_model.fit(features_data, self.time_series_data)
        residuals = self.time_series_data - reg_model.predict(features_data)
        logger.info(f"Linear regression model coefficients: {reg_model.coef_}")
        return reg_model, residuals

    def optimize_portfolio_weights(self, returns, risk_free_rate=0.01):
        """
        Use econometric principles to optimize portfolio weights using CAPM or Sharpe Ratio maximization.
        
        :param returns: Expected returns of the assets.
        :param risk_free_rate: The risk-free rate for Sharpe ratio calculation.
        :return: Optimized portfolio weights.
        """
        def sharpe_ratio(weights):
            portfolio_return = np.sum(returns.mean() * weights) * 252  # Annualized return
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))  # Annualized volatility
            return -(portfolio_return - risk_free_rate) / portfolio_volatility  # Negative because we minimize

        num_assets = len(returns.columns)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Weights sum to 1
        bounds = tuple((0, 1) for asset in range(num_assets))  # No short selling

        optimized_result = minimize(sharpe_ratio, num_assets * [1. / num_assets], bounds=bounds, constraints=constraints)
        optimized_weights = optimized_result.x
        logger.info(f"Optimized portfolio weights: {optimized_weights}")
        return optimized_weights


