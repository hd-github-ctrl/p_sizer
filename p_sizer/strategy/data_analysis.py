import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Dict, Any
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from datetime import datetime
from threading import Thread, Lock
import time
from bloomberg_api import BloombergAPI
from strategy_parameters import StrategyParameters
from alert_system import AlertSystem
from exchange_connector import ExchangeConnector
from order_execution_engine import OrderExecutionEngine
from performance_tracker import PerformanceTracker
from liquidity_detection import LiquidityDetection
from order_routing import OrderRouting
from market_regime_detection import MarketRegimeDetection

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BetaCalculator:
    def __init__(self, benchmark_returns: pd.Series, strategy_returns: pd.Series):
        self.benchmark_returns = benchmark_returns
        self.strategy_returns = strategy_returns
        logger.info("Initialized BetaCalculator with %d data points.", len(self.benchmark_returns))

    def calculate_beta(self) -> float:
        covariance_matrix = np.cov(self.strategy_returns, self.benchmark_returns)
        covariance = covariance_matrix[0, 1]
        variance = covariance_matrix[1, 1]
        beta = covariance / variance
        logger.info("Calculated beta: %f", beta)
        return beta

    def calculate_alpha(self, risk_free_rate: float) -> float:
        beta = self.calculate_beta()
        avg_strategy_return = self.strategy_returns.mean()
        avg_benchmark_return = self.benchmark_returns.mean()
        alpha = avg_strategy_return - risk_free_rate - beta * (avg_benchmark_return - risk_free_rate)
        logger.info("Calculated alpha: %f", alpha)
        return alpha

    def rolling_beta(self, window: int = 30) -> pd.Series:
        rolling_covariance = self.strategy_returns.rolling(window).cov(self.benchmark_returns)
        rolling_variance = self.benchmark_returns.rolling(window).var()
        rolling_beta = rolling_covariance / rolling_variance
        logger.info("Calculated rolling beta with window size %d.", window)
        return rolling_beta


class CalmarRatioCalculator:
    def __init__(self, returns: pd.Series, risk_free_rate: float = 0.0):
        self.returns = returns
        self.risk_free_rate = risk_free_rate
        logger.info("Initialized CalmarRatioCalculator with %d data points.", len(self.returns))

    def calculate_max_drawdown(self) -> float:
        cumulative_returns = (1 + self.returns).cumprod()
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()
        logger.info("Calculated maximum drawdown: %f%%", max_drawdown * 100)
        return max_drawdown

    def calculate_annualized_return(self) -> float:
        total_return = (1 + self.returns).prod() - 1
        n_years = len(self.returns) / 252
        annualized_return = (1 + total_return) ** (1 / n_years) - 1
        logger.info("Calculated annualized return: %f%%", annualized_return * 100)
        return annualized_return

    def calculate_calmar_ratio(self) -> float:
        max_drawdown = self.calculate_max_drawdown()
        annualized_return = self.calculate_annualized_return()
        if max_drawdown == 0:
            logger.warning("Max drawdown is zero, Calmar Ratio is set to infinity.")
            return np.inf
        calmar_ratio = (annualized_return - self.risk_free_rate) / abs(max_drawdown)
        logger.info("Calculated Calmar Ratio: %f", calmar_ratio)
        return calmar_ratio


class CointegrationAnalysis:
    def __init__(self, confidence_level: float = 0.05, max_p_value: float = 0.05):
        self.confidence_level = confidence_level
        self.max_p_value = max_p_value

    def test_cointegration(self, asset1: pd.Series, asset2: pd.Series) -> Tuple[bool, float, float]:
        coint_t, p_value, crit_value = coint(asset1, asset2)
        is_cointegrated = p_value < self.max_p_value
        return is_cointegrated, p_value, coint_t

    def find_cointegrated_pairs(self, data: pd.DataFrame) -> List[Tuple[str, str, float]]:
        n = data.shape[1]
        keys = data.columns
        pairs = []
        for i in range(n):
            for j in range(i+1, n):
                asset1 = data[keys[i]]
                asset2 = data[keys[j]]
                is_cointegrated, p_value, _ = self.test_cointegration(asset1, asset2)
                if is_cointegrated:
                    pairs.append((keys[i], keys[j], p_value))
        return pairs

    def get_spread(self, asset1: pd.Series, asset2: pd.Series) -> pd.Series:
        model = sm.OLS(asset1, sm.add_constant(asset2))
        results = model.fit()
        spread = asset1 - results.params[1] * asset2
        return spread

    def calculate_half_life(self, spread: pd.Series) -> float:
        spread_lag = spread.shift(1)
        spread_lag.iloc[0] = spread_lag.iloc[1]
        delta_spread = spread - spread_lag
        spread_lag = sm.add_constant(spread_lag)
        model = sm.OLS(delta_spread, spread_lag)
        results = model.fit()
        half_life = -np.log(2) / results.params[1]
        return half_life

    def calculate_z_score(self, spread: pd.Series) -> pd.Series:
        mean = spread.mean()
        std = spread.std()
        z_score = (spread - mean) / std
        return z_score

    def generate_trade_signals(self, z_score: pd.Series, entry_threshold: float = 2.0, exit_threshold: float = 0.5) -> pd.Series:
        signals = pd.Series(index=z_score.index)
        signals[z_score > entry_threshold] = -1
        signals[z_score < -entry_threshold] = 1
        signals[(z_score < exit_threshold) & (z_score > -exit_threshold)] = 0
        return signals.ffill().fillna(0)

    def monitor_cointegration(self, data: pd.DataFrame) -> Dict[Tuple[str, str], Dict[str, float]]:
        pairs_info = {}
        for pair in self.find_cointegrated_pairs(data):
            asset1, asset2, p_value = pair
            spread = self.get_spread(data[asset1], data[asset2])
            half_life = self.calculate_half_life(spread)
            z_score = self.calculate_z_score(spread)
            pairs_info[(asset1, asset2)] = {
                'p_value': p_value,
                'half_life': half_life,
                'latest_z_score': z_score.iloc[-1],
            }
        return pairs_info


class MacroFactorModel:
    def __init__(self, bloomberg_api: BloombergAPI, strategy_parameters: StrategyParameters, 
                 alert_system: AlertSystem, market_regime_detection: MarketRegimeDetection):
        self.bloomberg_api = bloomberg_api
        self.strategy_parameters = strategy_parameters
        self.alert_system = alert_system
        self.market_regime_detection = market_regime_detection
        self.logger = self.setup_logging()

    def setup_logging(self) -> logging.Logger:
        logging.basicConfig(filename='macrofactor_model.log',
                            level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger('MacroFactorModel')
        return logger

    def fetch_macroeconomic_data(self, indicators: list) -> dict:
        macro_data = {}
        for indicator in indicators:
            try:
                data = self.bloomberg_api.get_indicator_data(indicator)
                macro_data[indicator] = data
                self.logger.info(f"Fetched data for {indicator}: {data}")
            except Exception as e:
                self.logger.error(f"Error fetching data for {indicator}: {e}")
        return macro_data

    def analyze_macroeconomic_data(self, macro_data: dict) -> dict:
        analysis_results = {}
        if 'CPI' in macro_data:
            cpi = macro_data['CPI']['value']
            if cpi > 2.0:
                analysis_results['risk_level'] = 'high'
                analysis_results['leverage_adjustment'] = -0.1
                self.logger.info(f"High CPI detected ({cpi}%). Risk level set to high, reducing leverage.")
            else:
                analysis_results['risk_level'] = 'normal'
        if 'Unemployment Rate' in macro_data:
            unemployment_rate = macro_data['Unemployment Rate']['value']
            if unemployment_rate > 6.0:
                analysis_results['trade_frequency'] = 'low'
                self.logger.info(f"High Unemployment Rate detected ({unemployment_rate}%). Reducing trade frequency.")
            else:
                analysis_results['trade_frequency'] = 'normal'
        return analysis_results

    def adjust_strategy(self, analysis_results: dict):
        if 'risk_level' in analysis_results:
            if analysis_results['risk_level'] == 'high':
                self.strategy_parameters.adjust_leverage(-0.1)
                self.alert_system.send_alert("Risk level high due to macroeconomic factors. Leverage reduced.")
        if 'trade_frequency' in analysis_results:
            if analysis_results['trade_frequency'] == 'low':
                self.strategy_parameters.adjust_trade_frequency(-0.2)
                self.alert_system.send_alert("Trade frequency reduced due to high unemployment rate.")

    def execute_macro_analysis(self, indicators: list):
        self.logger.info("Starting macroeconomic analysis.")
        macro_data = self.fetch_macroeconomic_data(indicators)
        analysis_results = self.analyze_macroeconomic_data(macro_data)
        self.adjust_strategy(analysis_results)
        self.market_regime_detection.detect_and_adjust(macro_data)


class MarketDataAcquisition:
    def __init__(self, exchange_connector: ExchangeConnector, symbols: List[str], 
                 data_types: List[str], interval: int = 1):
        self.exchange_connector = exchange_connector
        self.symbols = symbols
        self.data_types = data_types
        self.interval = interval
        self.real_time_data = {symbol: Queue() for symbol in symbols}
        self.historical_data = {}
        self.lock = Lock()
        self.logger = self.setup_logging()
        self.fetch_threads = []
        self.stop_event = False

    def setup_logging(self) -> logging.Logger:
        logging.basicConfig(filename='market_data_acquisition.log',
                            level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger('MarketDataAcquisition')
        return logger

    def start_real_time_data_fetch(self):
        self.logger.info("Starting real-time data acquisition.")
        self.stop_event = False
        for symbol in self.symbols:
            thread = Thread(target=self._fetch_real_time_data, args=(symbol,))
            thread.start()
            self.fetch_threads.append(thread)

    def stop_real_time_data_fetch(self):
        self.logger.info("Stopping real-time data acquisition.")
        self.stop_event = True
        for thread in self.fetch_threads:
            thread.join()
        self.fetch_threads = []

    def _fetch_real_time_data(self, symbol: str):
        while not self.stop_event:
            try:
                with self.lock:
                    data = {}
                    if 'order_book' in self.data_types:
                        data['order_book'] = self.exchange_connector.fetch_order_book(symbol)
                    if 'trades' in self.data_types:
                        data['trades'] = self.exchange_connector.fetch_trades(symbol)
                    if 'ohlc' in self.data_types:
                        data['ohlc'] = self.exchange_connector.fetch_ohlc(symbol)
                    self.real_time_data[symbol].put(data)
                    self.logger.info(f"Fetched real-time data for {symbol}: {data}")
            except Exception as e:
                self.logger.error(f"Error fetching real-time data for {symbol}: {e}")
            time.sleep(self.interval)

    def fetch_historical_data(self, symbol: str, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        self.logger.info(f"Fetching historical data for {symbol} from {start_time} to {end_time}.")
        try:
            historical_data = {}
            if 'order_book' in self.data_types:
                historical_data['order_book'] = self.exchange_connector.fetch_historical_order_book(symbol, start_time, end_time)
            if 'trades' in self.data_types:
                historical_data['trades'] = self.exchange_connector.fetch_historical_trades(symbol, start_time, end_time)
            if 'ohlc' in self.data_types:
                historical_data['ohlc'] = self.exchange_connector.fetch_historical_ohlc(symbol, start_time, end_time)
            self.historical_data[symbol] = historical_data
            return historical_data
        except Exception as e:
            self.logger.error(f"Error fetching historical data for {symbol}: {e}")
            return {}

    def get_latest_data(self, symbol: str) -> Dict[str, Any]:
        if symbol in self.real_time_data and not self.real_time_data[symbol].empty():
            latest_data = self.real_time_data[symbol].get()
            return latest_data
        else:
            self.logger.warning(f"No data available for {symbol}.")
            return {}


class MarketRegimeDetection:
    def __init__(self, exchange_connector: ExchangeConnector, bloomberg_api: BloombergAPI, symbols: List[str], lookback_period: int = 20):
        self.exchange_connector = exchange_connector
        self.bloomberg_api = bloomberg_api
        self.symbols = symbols
        self.lookback_period = lookback_period
        self.current_regime = {symbol: MarketRegime.UNKNOWN for symbol in symbols}
        self.logger = self.setup_logging()

    def setup_logging(self) -> logging.Logger:
        logging.basicConfig(filename='market_regime_detection.log',
                            level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger('MarketRegimeDetection')
        return logger

    def detect_market_regime(self, symbol: str) -> MarketRegime:
        self.logger.info(f"Detecting market regime for {symbol}.")
        try:
            ohlc_data = self.exchange_connector.fetch_ohlc(symbol, self.lookback_period)
            close_prices = ohlc_data['close'].values
            macro_data = self.bloomberg_api.get_market_data(symbol)
            atr = self._calculate_atr(ohlc_data)
            rsi = self._calculate_rsi(close_prices)
            volatility = self._calculate_volatility(close_prices, macro_data)
            regime = self._classify_market_regime(rsi, atr, volatility)
            self.current_regime[symbol] = regime
            return regime
        except Exception as e:
            self.logger.error(f"Error detecting market regime for {symbol}: {e}")
            return MarketRegime.UNKNOWN

    def _calculate_atr(self, ohlc_data: pd.DataFrame) -> float:
        high_low = ohlc_data['high'] - ohlc_data['low']
        high_close = np.abs(ohlc_data['high'] - ohlc_data['close'].shift())
        low_close = np.abs(ohlc_data['low'] - ohlc_data['close'].shift())
        true_range = np.maximum.reduce([high_low, high_close, low_close])
        atr = true_range.rolling(window=self.lookback_period).mean().iloc[-1]
        return atr

    def _calculate_rsi(self, close_prices: np.array, period: int = 14) -> float:
        delta = np.diff(close_prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.mean(gain[-period:])
        avg_loss = np.mean(loss[-period:])
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_volatility(self, close_prices: np.array, macro_data: Dict[str, Any]) -> float:
        log_returns = np.diff(np.log(close_prices))
        basic_volatility = np.std(log_returns)
        if "VIX" in macro_data:
            vix_adjustment = macro_data["VIX"] / 100.0
            adjusted_volatility = basic_volatility * (1 + vix_adjustment)
        else:
            adjusted_volatility = basic_volatility
        return adjusted_volatility

    def _classify_market_regime(self, rsi: float, atr: float, volatility: float) -> MarketRegime:
        if rsi > 70:
            return MarketRegime.TRENDING_UP
        elif rsi < 30:
            return MarketRegime.TRENDING_DOWN
        elif volatility > atr:
            return MarketRegime.HIGH_VOLATILITY
        elif atr < 0.01:
            return MarketRegime.LOW_VOLATILITY
        else:
            return MarketRegime.RANGE_BOUND

    def detect_all_market_regimes(self):
        for symbol in self.symbols:
            self.detect_market_regime(symbol)


class MarketSentimentIndicators:
    def __init__(self, bloomberg_api: BloombergAPI, exchange_connector: ExchangeConnector):
        self.bloomberg_api = bloomberg_api
        self.exchange_connector = exchange_connector
        self.logger = self.setup_logging()

    def setup_logging(self) -> logging.Logger:
        logging.basicConfig(filename='market_sentiment_indicators.log',
                            level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger('MarketSentimentIndicators')
        return logger

    def fetch_sentiment_data(self, symbol: str) -> Dict[str, Any]:
        self.logger.info(f"Fetching sentiment data for {symbol}.")
        try:
            sentiment_data = self.bloomberg_api.get_sentiment_data(symbol)
            return sentiment_data
        except Exception as e:
            self.logger.error(f"Failed to fetch sentiment data for {symbol}: {e}")
            return {}

    def process_sentiment_data(self, sentiment_data: Dict[str, Any]) -> float:
        if not sentiment_data:
            self.logger.warning("No sentiment data provided.")
            return 0.0
        sentiment_score = sentiment_data.get('sentiment_score', 0.0)
        return sentiment_score

    def adjust_strategy_based_on_sentiment(self, symbol: str, sentiment_score: float):
        self.logger.info(f"Adjusting strategy for {symbol} based on sentiment score: {sentiment_score}")
        if sentiment_score > 0.5:
            self.logger.info(f"Sentiment is bullish for {symbol}. Adjusting position size.")
        elif sentiment_score < -0.5:
            self.logger.info(f"Sentiment is bearish for {symbol}. Reducing position size.")
        else:
            self.logger.info(f"Sentiment is neutral for {symbol}. No adjustment needed.")

    def update_sentiment_for_symbols(self, symbols: list):
        for symbol in symbols:
            sentiment_data = self.fetch_sentiment_data(symbol)
            sentiment_score = self.process_sentiment_data(sentiment_data)
            self.adjust_strategy_based_on_sentiment(symbol, sentiment_score)



