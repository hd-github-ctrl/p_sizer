import logging
import numpy as np
import pandas as pd

class StrategyMetrics:
    def __init__(self, config):
        """
        Initializes the StrategyMetrics with configurable parameters and default settings.
        
        :param config: A configuration dictionary containing relevant parameters like risk-free rate, win rate, etc.
        """
        self.config = config
        self.trade_records = []
        self.risk_free_rate = self.config.get('risk_free_rate', 0.01)
        self.trading_days_per_year = self.config.get('trading_days_per_year', 252)
        self.target_return = self.config.get('target_return', 0.0)
        self.initial_capital = self.config.get('initial_capital', 100000)  # Default to $100k
        self.win_rate = self.config.get('win_rate', 0.6)
        self.loss_rate = self.config.get('loss_rate', 0.4)
        self.risk_per_trade = self.config.get('risk_per_trade', 0.01)
        self.reward_to_risk_ratio = self.config.get('reward_to_risk_ratio', 2.0)
        logging.info("StrategyMetrics initialized with config: %s", self.config)

    def record_trade(self, trade):
        """
        Records a trade for performance tracking.
        
        :param trade: A dictionary containing trade details such as entry and exit prices, trade size, etc.
        """
        self.trade_records.append(trade)

    def calculate_pnl(self):
        """
        Calculates total Profit and Loss (PnL) from recorded trades.
        
        :return: Total PnL value.
        """
        pnl = sum((trade['exit_price'] - trade['entry_price']) * trade['size'] for trade in self.trade_records)
        logging.info(f"Total PnL calculated: {pnl}")
        return pnl

    def _get_trade_returns(self):
        """
        Helper method to calculate the returns from recorded trades.
        
        :return: A NumPy array of trade returns.
        """
        returns = np.array([(trade['exit_price'] - trade['entry_price']) / trade['entry_price'] for trade in self.trade_records])
        logging.debug(f"Trade returns calculated: {returns}")
        return returns

    def _get_equity_curve(self):
        """
        Helper method to calculate the equity curve from trade records.
        
        :return: A NumPy array representing the equity curve.
        """
        equity_curve = np.cumsum([(trade['exit_price'] - trade['entry_price']) * trade['size'] for trade in self.trade_records])
        logging.debug(f"Equity curve calculated: {equity_curve}")
        return equity_curve

    def calculate_sharpe_ratio(self):
        """
        Calculates the Sharpe Ratio based on recorded trade returns and the risk-free rate.
        
        :return: Sharpe Ratio.
        """
        returns = self._get_trade_returns()
        excess_returns = returns - self.risk_free_rate / self.trading_days_per_year
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(self.trading_days_per_year)
        logging.info(f"Sharpe Ratio calculated: {sharpe_ratio}")
        return sharpe_ratio

    def calculate_sortino_ratio(self):
        """
        Calculates the Sortino Ratio considering downside risk only.
        
        :return: Sortino Ratio.
        """
        returns = self._get_trade_returns()
        downside_risk = np.std([r for r in returns if r < 0])
        excess_return = np.mean(returns) - self.target_return
        sortino_ratio = excess_return / downside_risk * np.sqrt(self.trading_days_per_year)
        logging.info(f"Sortino Ratio calculated: {sortino_ratio}")
        return sortino_ratio

    def calculate_max_drawdown(self):
        """
        Calculates the Maximum Drawdown from the equity curve.
        
        :return: Maximum Drawdown.
        """
        equity_curve = self._get_equity_curve()
        drawdown = equity_curve - np.maximum.accumulate(equity_curve)
        max_drawdown = drawdown.min()
        logging.info(f"Max Drawdown calculated: {max_drawdown}")
        return max_drawdown

    def calculate_calmar_ratio(self):
        """
        Calculates the Calmar Ratio, which is the annualized return divided by the maximum drawdown.
        
        :return: Calmar Ratio.
        """
        annual_return = self.calculate_annualized_return(self._get_trade_returns())
        max_drawdown = abs(self.calculate_max_drawdown())
        calmar_ratio = annual_return / max_drawdown if max_drawdown != 0 else np.nan
        logging.info(f"Calmar Ratio calculated: {calmar_ratio}")
        return calmar_ratio

    def calculate_annualized_return(self, returns):
        """
        Calculates annualized return from daily returns.
        
        :return: Annualized return.
        """
        average_daily_return = np.mean(returns)
        annualized_return = (1 + average_daily_return) ** self.trading_days_per_year - 1
        logging.debug(f"Annualized return calculated: {annualized_return}")
        return annualized_return

    def generate_performance_report(self):
        """
        Generates a full performance report in a DataFrame format.
        
        :return: A Pandas DataFrame containing performance metrics like PnL, Sharpe Ratio, etc.
        """
        pnl = self.calculate_pnl()
        sharpe_ratio = self.calculate_sharpe_ratio()
        sortino_ratio = self.calculate_sortino_ratio()
        max_drawdown = self.calculate_max_drawdown()
        calmar_ratio = self.calculate_calmar_ratio()

        report_data = {
            'PnL': [pnl],
            'Sharpe Ratio': [sharpe_ratio],
            'Sortino Ratio': [sortino_ratio],
            'Max Drawdown': [max_drawdown],
            'Calmar Ratio': [calmar_ratio]
        }

        report_df = pd.DataFrame(report_data)
        logging.info("Performance report generated")
        return report_df

    def reset(self):
        """
        Resets the performance tracker for a new session.
        """
        self.trade_records = []
        logging.info("Performance tracker reset; trade records cleared")

    def calculate_risk_of_ruin(self):
        """
        Calculates the risk of ruin based on current strategy parameters.
        
        :return: Risk of Ruin value.
        """
        loss_to_win_ratio = self.loss_rate / self.win_rate
        kelly_fraction = self.win_rate - (self.loss_rate / self.reward_to_risk_ratio)
        
        if kelly_fraction <= 0:
            logging.warning("The strategy is not viable in the long term, high risk of ruin.")
            return 1.0

        risk_of_ruin = (loss_to_win_ratio ** (kelly_fraction / self.risk_per_trade))
        logging.info(f"Risk of ruin calculated: {risk_of_ruin}")
        return risk_of_ruin

    def simulate_risk_of_ruin(self, num_trades, simulations=10000):
        """
        Simulates the risk of ruin using a Monte Carlo approach.
        
        :param num_trades: Number of trades to simulate.
        :param simulations: Number of simulation runs.
        :return: Simulated risk of ruin.
        """
        import random

        ruins = 0
        for _ in range(simulations):
            capital = self.initial_capital
            for _ in range(num_trades):
                if random.random() < self.win_rate:
                    capital += self.reward_to_risk_ratio * self.risk_per_trade * capital
                else:
                    capital -= self.risk_per_trade * capital
                if capital <= 0:
                    ruins += 1
                    break

        simulated_risk_of_ruin = ruins / simulations
        logging.info(f"Simulated risk of ruin: {simulated_risk_of_ruin}")
        return simulated_risk_of_ruin

