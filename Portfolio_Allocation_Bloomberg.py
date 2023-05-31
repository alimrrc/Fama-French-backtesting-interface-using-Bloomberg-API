#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 14:59:08 2023

@author: ali
"""
import math
import blpapi
from blpapi.exception import IndexOutOfRangeException
from typing import Dict
from dataclasses import dataclass
import datetime as dt
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.optimize import minimize
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
import holidays
from dateutil import rrule, relativedelta
import warnings

warnings.filterwarnings('ignore')

def get_business_days(start_date, end_date, country):
    """
        This function returns a list of all working days (weekdays that are not public holidays)
        between two specified dates, for a given country.

        Args:
            start_date (datetime.date): The start date of the working day search period.
            end_date (datetime.date): The end date of the working day search period.
            country (str): The name of the country for which the list of working days is to be obtained.

        Returns:
            list: A list of datetime objects representing the working days between the start and end dates,
                  excluding public holidays for the specified country.
        """
    # Create a list of all holidays in the specified country
    holiday_list = holidays.CountryHoliday(country, years=[start_date.year, end_date.year]).keys()

    # Use rrule to create an iterator of all weekdays between start and end dates
    weekdays = rrule.rrule(rrule.DAILY, byweekday=range(0, 5), dtstart=start_date, until=end_date)

    # Use a list comprehension to create a list of business days by excluding any dates in the holiday list
    business_days = [d for d in weekdays if d.date() not in holiday_list]

    # Return the list of business days
    return business_days

class Quote:
    """
       This class represents a quote from a financial asset at a given date.

       Attributes:
           date (datetime.datetime): The date and time of the quote.
           ticker (str): The ticker symbol of the financial asset.
           close (float): The closing price of the financial asset on the date of the quote.
           isinindex (bool): A Boolean indicator of whether the financial asset is a stock index (True) or an individual
            stock (False).
           size (float): The market cap of the financial asset, if available.
           value (float): The book value of the financial asset position, if available.
           momentum (float): The price change of the financial asset relative to a previous time period, if available.

       Methods:
           __init__(self, date, ticker, close, isinindex): Initializes a new instance of the Quote class with the
                                                           specified values for the attributes date, ticker, close
                                                           and isinindex.
       """
    date : datetime = None
    ticker : str = None
    close : float = None
    isinindex : bool = None
    size : float = None
    value : float = None
    momentum : float = None
    
    def __init__(self, date, ticker, close, isinindex):
        self.date = date
        self.ticker = ticker
        self.close = close
        self.isinindex = isinindex

def is_nan(x):
    """
    This function returns True if the supplied argument is a NaN (Not a Number) value, and False otherwise.

    Args:
        x (float or np.float64): The value to check.

    Returns:
        bool: True if the value is NaN, False otherwise.
    """
    return isinstance(x, float) and (math.isnan(x) or np.isnan(x))
    
class QuotesSerie:
    """
    This class represents a series of stock quotes.
    This is where the values of the quote object are retrieved.

    Attributes:
    -----------
        quote (Dict[datetime, Quote]): A dictionary containing quotes, indexed by their date.
           num_quote (int): The number of quotes in the series.

    Methods:
    --------
       __init__(self): Initializes a new instance of the QuoteSerie class with the specified values for the attributes
                        date, ticker, close and isinindex.
       AddQuote(quote: Quote): Adds a quote to the series.
       DeleteQuote(date: datetime): Removes a quote from the series.
       returns -> Dict[datetime, float]: Calculates and returns the daily returns of the series as a dictionary.
       ComputeMomentum() -> Dict[datetime, float]: Calculates and returns the moments of the series as a dictionary.
       """
    def __init__(self):
        self.quote = dict()
        self.num_quote: int = 0
    
    def AddQuote(self, quote: Quote):
        """
        Adds a quote to the series.

        Args:
            quote (Quote): The quote to be added to the series.
        """
        self.quote[quote.date] = quote
        self.num_quote += 1
        
    def DeleteQuote(self, date: datetime):
        """
        Removes a quote from the series.

        Args:
               date (datetime): The date of the quote to be removed.
           """
        del self.quote[date]
        self.num_quote -= 1
    
    @property
    def returns(self) -> Dict[datetime, float]:
        """
        Calculates and returns the daily returns of the series as a dictionary.

        Returns:
               Dict[datetime, float]: A dictionary containing daily returns for each date in the series.
           """
        returns_dict = {}
        prev_close = None
        for date in sorted(self.quote.keys()):
            quote = self.quote[date]
            if quote.ticker == "STLAP FP":
                pass
            if prev_close is not None:
                if not is_nan(quote.close):
                    if prev_close != 0.0:
                        returns_dict[date] = (quote.close - prev_close) / prev_close
                        prev_close = quote.close
                    else:
                        returns_dict[date] = 0.0
                        prev_close = quote.close
                        quote.isinindex = False
                else:
                    returns_dict[date] = 0.0
                    quote.isinindex = False
            else:
                if not is_nan(quote.close):
                    prev_close = quote.close
                else:
                    prev_close = 0.0

        return returns_dict
    
    def ComputeMomentum(self) -> Dict[datetime, float]:
        """
        Calculates and returns the momentum of the series as a dictionary.

        Returns:
                Dict[datetime, float]: A dictionary containing the momentum for each date in the series.
        """
        list_dates = sorted(self.quote.keys())
        # The for loop starts from index 250 of the sorted list of dates in the series. For each date, we get the
        # indices corresponding to the dates 250 days and 20 days before this date by using the index method of the
        # list_dates list.
        for date in list_dates[250:]:
            index_date = list_dates.index(date)
            date_250 = list_dates[index_date - 250]
            date_20 = list_dates[index_date - 20]
            # If the closure of the date 20 days before the current date or the closure of the date 250 days before
            # the current date is NaN (Not a Number), the isinindex variable of the Quote object corresponding to the
            # current date is set to False.
            if self.quote[date_20].close == np.nan or self.quote[date_250].close == np.nan:
                self.quote[date].isinindex = False
                # Otherwise, the momentum variable of the Quote object corresponding to the current date is calculated
                # by dividing the close of the date 20 days before the current date by the close of the date 250 days
                # before the current date, then subtracting 1.
            else:
                self.quote[date].momentum = self.quote[date_20].close / self.quote[date_250].close - 1

class Universe:
    """
    A class representing a universe of financial assets and their quotes.

    Attributes:
    -----------
    quoteseries: dict
       A dictionary of quotes series, with the ticker symbol as key and the quotes series as value.
    number_assets: int
       The number of assets in the universe.
    number_dates: int
       The number of dates in the universe.
    tickers: list
       A list of all the tickers in the universe.
    dates: list
       A list of all the dates in the universe after uniformizing dates.

    Methods:
    --------
    AddQuotesSerie(quote_serie: QuotesSerie)
       Adds a quotes series to the universe.
    DeleteQuotesSerie(ticker: str)
       Deletes a quotes series from the universe.
    UniformizeDates()
       Uniformizes dates of all the quotes series.
    """

    def __init__(self):
        self.quoteseries = dict()
        self.number_assets: int = 0
        self.number_dates: int = 0
        self.tickers = list()
        self.dates = list()
    
    def AddQuotesSerie(self, quote_serie: QuotesSerie):
        """
        Adds a series of quotes to the universe.

        Parameters:
            quote_serie (QuotesSerie): The series of quotes to add.

        Returns:
            None
        """
        dates = sorted(quote_serie.quote.keys())
        date = dates[0]
        quote = quote_serie.quote[date]
        ticker = quote.ticker
        self.quoteseries[ticker] = quote_serie
        self.number_assets += 1
        self.tickers.append(ticker)
        
    def DeleteQuotesSerie(self, ticker: str):
        """
        Removes a series of quotes from the universe.

        Parameters:
            ticker (str): The ticker of the quote series to be deleted.

        """
        del self.quoteseries[ticker]
        self.number_assets -= 1
        self.tickers.remove(ticker)
            
    def UniformizeDates(self):
        """
        Standardizes the dates of the quote series in the universe.
        This function ensures that all price series have the same dates by deleting missing dates or removing missing
        quotes from the quote series, then updates the universe dates variable and calculates the number of dates in
        the universe.

        Raises:
            Exception: If no dates are found in the universe.

        """
        # First, we Create a date_counts dictionary to store the number of times each date appears in the price series.
        date_counts = dict()
        temp_dates = list()

        # Browse each date in each price series and increase the number of date_counts if it is already present,
        # or add a new entry to date_counts if the date is new.
        for ticker in self.quoteseries.keys():
            for date in self.quoteseries[ticker].quote.keys():
                if self.quoteseries[ticker].quote[date].date in date_counts.keys():
                    date_counts[self.quoteseries[ticker].quote[date].date] += 1
                else:
                    date_counts[self.quoteseries[ticker].quote[date].date] = 1
        # Delete dates that do not appear in all price series. If a date does not appear in all price series,
        # the corresponding quote is removed from the price series.
        for date in date_counts.keys():
            if date_counts[date] != self.number_assets:
                for ticker in self.quoteseries.keys():
                    if date in self.quoteseries[ticker].quote.keys():
                        self.quoteseries[ticker].DeleteQuote(date)
            else:
                temp_dates.append(date)
        
        self.dates = sorted(temp_dates)
        tickers = sorted(self.quoteseries.keys())
        firstquoteserie = self.quoteseries[tickers[0]]
        self.number_dates = firstquoteserie.num_quote
        
        if self.number_dates == 0:
            raise Exception()
            
def ReballancingFrequency(x: int, what: str) -> timedelta:
    """
    Calculates the rebalancing frequency of a portfolio.

    Parameters:
       x (int): The number of time units.
       what (str): The unit of time. Can be "day(s)", "week(s)" or "month(s)".

   Raises:
       NotImplementedError: If the time unit is not supported.

   Returns:
       timedelta: The rebalancing frequency.
    """
    if what.lower() == "day" or what.lower() == "days":
        return timedelta(days=x)

    if what.lower() == "week" or what.lower() == "weeks":
        return timedelta(weeks=x)

    if what.lower() == "month" or what.lower() == "months":
        return timedelta(days=x*30)

    else:
        raise NotImplementedError()

def keep_unique_values(list1, list2):
    """
    This function returns two modified lists that no longer have any duplicates. This function will be useful
    if you don't want to have lists of tickers that you are going to short and lists of tickers that you are
     going to sell long with a common ticker.

        Args:
            list1: A list of values.
            list2: Another list of values.

        Returns:
            Two lists modified to contain only unique only the unique values.

    """
    unique_values = set(list1 + list2)
    for value in unique_values:
        count1 = list1.count(value)
        count2 = list2.count(value)
        if count1 > count2:
            if count2 != 0:
                list2 = [x for x in list2 if x != value]
            while count1 > 1:
                list1.remove(value)
                count1 -= 1
        elif count2 > count1:
            if count1 != 0:
                list1 = [x for x in list1 if x != value]
            while count2 > 1:
                list2.remove(value)
                count2 -= 1
        else:
            list1 = [x for x in list1 if x != value]
            list2 = [x for x in list2 if x != value]
    return (list1, list2)

class Strategy(Enum):
    """
    The Strategy class is an enumeration of different investment strategies that can be used to construct a portfolio.
    Each element of the enumeration represents a specific strategy and is associated with a string that serves as an
    identifier for that strategy.

    Four strategies are defined (one is the default).
    """
    MINVAR = "MinVar"
    MAXSHARPE = "MaxSharp"
    EQWEIGHT = "EqWeight"
    NOTHING = "Nothing"
                    
@dataclass
class Portfolio:
    """
   A class representing a portfolio.

   Attributes:
   -----------
   date: datetime
       The date of the portfolio.
   factor_weights: List[float]
       The factor weights of the portfolio.
   weights: List[float]
       The weights of the portfolio.
   value: float
       The value of the portfolio.
    """
    date : datetime = None
    factor_weights : list() = None
    weights : list() = None
    value : float = None

class Factor:
    """
    Class representing an investment factor.

    Attributes:
    ----------
    factor_name: str
        The name of the factor.
    long_tickers: dict
        A dictionary containing the tickers of the long assets associated with this factor.
    short_tickers : dict
        A dictionary containing the tickers of the short assets associated with this factor.
    returns : dict
        A dictionary containing the returns associated with each ticker.

    Methods:
    -------
    __init__(factor_name: str)
        Constructor of the class.
    """
    def __init__(self, factor_name: str):
        self.factor_name : str = None
        self.long_tickers = dict()
        self.short_tickers = dict()
        self.returns = dict()

class Backtest:
    """
    Class for performing a backtest of a given investment strategy on a given universe of assets with one
    or several factors.

    Args:
       universe (Universe): The universe of assets to be considered for the backtest.
       strategy (Strategy): The investment strategy to be tested. -->From the enum class.
       factors (list): List of factors to be considered in the backtest.
       reballancing_frequency: The frequency of portfolio reballancing.
       what: The unit of time. Can be "day(s)", "week(s)" or "month(s)". Using in Universe.ReballancingFrequency
       base (float): The starting base value of the portfolio.
       transaction_fee (float): The transaction fee to be applied to each trade.

    Attributes:
       strategy (Strategy): The investment strategy to be tested.
       factors (list): List of factors to be considered in the backtest.
       basis (float): The starting base value of the portfolio.
       transaction_fee (float): The transaction fee to be applied to each trade.
       universe (Universe): The universe of assets to be considered for the backtest.
       portfolio (dict): A dictionary containing the portfolio values at each rebalancing date.
       available_calendar (list): The calendar of available dates for the universe.
       factor_calendar (list): The calendar of dates for which factor values are available.
       calendar (list): The calendar of dates for which portfolio values are calculated.
       tickers (list): The list of tickers in the universe.
       universe_returns (dict): A dictionary containing the returns for each ticker in the universe.
       value (Factor): The factor representing value.
       size (Factor): The factor representing size.
       momentum (Factor): The factor representing momentum.
       ts_frequency (timedelta): The frequency of the time series.
       reballancing_frequency (RebalancingFrequency): The frequency of portfolio reballancing.
       reballancing_dates (list): The list of dates on which portfolio reballancing occurs.

    Methods:
       inindex(date): Returns a list of tickers that were in the index on the specified date.
       GetMomentum: Compute the momentum factor for each ticker in the universe at each factor calendar date.
    """
    def __init__(self, universe: Universe, 
                 strategy: Strategy, 
                 factors: list(),
                 reballancing_frequency, 
                 what,
                 base: float = 100,
                 transaction_fee: float = 0):
        self.strategy = strategy
        self.factors = factors
        self.basis = base
        self.transaction_fee = transaction_fee
        self.universe = universe
        self.portfolio = dict()
        self.available_calendar = self.universe.dates
        start: datetime = self.available_calendar[0]
        self.factor_calendar = [date for date in self.available_calendar if date >= datetime(start.year + 1, start.month, start.day, tzinfo=start.tzinfo)]
        self.calendar = [date for date in self.available_calendar if date >= datetime(start.year + 2, start.month, start.day, tzinfo=start.tzinfo)]
        self.tickers = self.universe.tickers
        self.universe_returns = dict()
        self.value = Factor("Value")
        self.size = Factor("Size")
        self.momentum = Factor("Momentum")
        
        for ticker in self.tickers:
            self.universe_returns[ticker] = self.universe.quoteseries[ticker].returns
        
        if "Momentum" in self.factors:
            self.GetMomentum()
                
        if "Size" in self.factors:
            self.GetSize()
        
        if "Value" in self.factors:
            self.GetValue()
        
        self.ts_frequency: timedelta(days=1)
        self.reballancing_frequency = ReballancingFrequency(reballancing_frequency, what)
        self.reballancing_dates = list()
        self.Run()
    
    def inindex(self, date: datetime):
        """
        The inindex method returns the list of tickers present in the index at a given date.

        Parameters:
        date: a datetime object representing the date for which we want to retrieve the tickers present in the index.
        Return:
        A list of strings representing the tickers present in the index at the given date.
        """
        inindex = []
        for ticker in self.tickers:
            if self.universe.quoteseries[ticker].quote[date].isinindex:
                inindex.append(ticker)
        return inindex
    
    def GetMomentum(self):
        """
        Compute the momentum factor for each ticker in the universe at each factor calendar date.
        Assign a short list and a long list of tickers based on the computed momentum factor.
        Compute the momentum factor returns at each factor calendar date for the short and long portfolios.

        Returns:
            This method does not return anything, it just modifies the attribute of the Backtest class.
            - self.momentum.long_tickers: a dictionary containing the 5 assets with the highest values for the factor, for each trading date
            - self.momentum.short_tickers: a dictionary containing the 5 assets with the lowest values for the factor, for each trading date
            - self.momentum.returns: a dictionary containing the return of the strategy for each trading date
        """
        for ticker in self.tickers:
            self.universe.quoteseries[ticker].ComputeMomentum()
        
        for date in self.factor_calendar:
            
            dict_momentum = {}
            for ticker in self.inindex(date):
                dict_momentum[self.universe.quoteseries[ticker].quote[date].momentum] = ticker
                
            list_momentum = sorted(dict_momentum.keys())
            
            self.momentum.short_tickers[date] = [dict_momentum[i] for i in list_momentum[:5]]
            self.momentum.long_tickers[date] = [dict_momentum[i] for i in list_momentum[-5:]]
            
            self.momentum.returns[date] = 0.0
            
            for ticker in self.inindex(date):
                if ticker in self.momentum.long_tickers[date]:
                    self.momentum.returns[date] += self.universe_returns[ticker][date] * 1/5
                if ticker in self.momentum.short_tickers[date]:
                    self.momentum.returns[date] -= self.universe_returns[ticker][date] * 1/5
    
    def GetSize(self):
        """
        Update the size of each stock in the universe at each date using the current market capitalization data
        retrieved from Bloomberg (thanks to BBG.fetch_series). Then, for each date in the factor_calendar,
        we calculate the 5 long and 5 short stocks with the largest and smallest size, respectively.
        Finally, compute the returns of each strategy based on the universe returns and the respective weights
        of each stock, which is equal to 1/5 for each stock.
        Returns:
            This method does not return anything, it just modifies the attribute of the Backtest class.
            - self.size.long_tickers: a dictionary containing the 5 assets with the highest values for the factor, for each trading date
            - self.size.short_tickers: a dictionary containing the 5 assets with the lowest values for the factor, for each trading date
            - self.size.returns: a dictionary containing the return of the strategy for each trading date
        """
        # The objective of the GetSize function of the Backtest class is to retrieve the market capitalisation (size)
        # of the stocks in the portfolio over a given date range, so we use BBG.fetch_series. Before that, we add the
        # word equity after the ticker name to successfully retrieve them from Bloomberg.

        list_tickers = [ticker + " Equity" for ticker in self.tickers]
        df = BBG.fetch_series(list_tickers, "CUR_MKT_CAP", self.factor_calendar[0].strftime('%Y%m%d'), self.factor_calendar[-1].strftime('%Y%m%d'), period="DAILY", calendar="ACTUAL", fx=None,fperiod=None, verbose=False)

        df_nan = df.isna()
        # The dates and securities are scanned to determine which ones do not have a market capitalisation,
        # and these securities and the corresponding data are removed from the lists and associated dictionaries in
        # the Universe class and in the Backtest class.
        for date in df.index:
            if date in self.factor_calendar:
                not_in_df = []
                for ticker in df.columns:
                    if df_nan.loc[date, ticker]:
                        self.universe.quoteseries[ticker[:-7]].quote[date].isinindex = False
                        not_in_df.append(ticker[:-7])
                if len(not_in_df) >= len(self.tickers) - 10:
                    for ticker in self.tickers:
                        self.universe.quoteseries[ticker].DeleteQuote(date)
                        del self.universe_returns[ticker][date]
                    self.factor_calendar.remove(date)
                    try:
                        self.calendar.remove(date)
                    except:
                        pass
                    self.available_calendar.remove(date)
                    self.universe.number_dates -= 1
                    self.universe.dates.remove(date)
                    del self.momentum.long_tickers[date]
                    del self.momentum.short_tickers[date]
            else:
                df = df.drop(labels=date, axis=0)

        list_dates = list(df.index)

        for date in self.factor_calendar:
            if not date in list_dates:
                for ticker in self.tickers:
                    self.universe.quoteseries[ticker].DeleteQuote(date)
                    del self.universe_returns[ticker][date]
                try:
                    self.available_calendar.remove(date)
                    self.factor_calendar.remove(date)
                except:
                    pass
                self.calendar.remove(date)
                self.universe.number_dates -= 1
                self.universe.dates.remove(date)
                del self.momentum.long_tickers[date]
                del self.momentum.short_tickers[date]
        
        for ticker in list_tickers:
            for date in list_dates:
                try:
                    self.universe.quoteseries[ticker[:-7]].quote[date].size = df[ticker][date]
                except:
                    self.universe.quoteseries[ticker[:-7]].quote[date].size = df[ticker][list_dates[list_dates.index(date)-1]]

        # Then, for each date in the factor_calendar,we calculate the 5 long and 5 short stocks with the largest and
        # smallest size,respectively.
        for date in self.factor_calendar:
            
            dict_size = {}
            for ticker in self.inindex(date):
                dict_size[self.universe.quoteseries[ticker].quote[date].size] = ticker
                
            list_size = sorted(dict_size.keys())
# modif long_tickers
            self.size.long_tickers[date] = [dict_size[i] for i in list_size[:5]]
            self.size.short_tickers[date] = [dict_size[i] for i in list_size[-5:]]
            
            self.size.returns[date] = 0.0

            #We compute the returns of each strategy based on the universe returns and the respective weights
            # of each stock, which is equal to 1/5 for each stock.
            for ticker in self.inindex(date):
                if ticker in self.size.long_tickers[date]:
                    self.size.returns[date] += self.universe_returns[ticker][date] * 1/5
                if ticker in self.size.short_tickers[date]:
                    self.size.returns[date] -= self.universe_returns[ticker][date] * 1/5
            
    def GetValue(self):
        """
        This GetValue function is a method of the Backtest class. This method retrieves the values of the stocks in the
         portfolio for each date and updates the lists of longs and shorts for each date in the factor calendar.

        Specifically, the method uses Bloomberg's fetch_series function to fetch the values of the stocks from the
        specified start date to the specified end date. It then processes the missing values in the data table and
        removes the missing values for all dates in the factor calendar.

        Finally, for each date in the factor calendar, the method sorts the stocks according to their value and assigns
        the first 5 stocks as shorts and the last 5 as longs. The weighted returns of each long and short are calculated
         and stored in the long_returns and short_returns lists.

        Returns:
            This method does not return anything, it just modifies the attribute of the Backtest class.
            - self.value.long_tickers: a dictionary containing the 5 assets with the highest values for the factor, for each trading date
            - self.value.short_tickers: a dictionary containing the 5 assets with the lowest values for the factor, for each trading date
            - self.value.returns: a dictionary containing the return of the strategy for each trading date
        """


        list_tickers = [ticker + " Equity" for ticker in self.tickers]
        df = BBG.fetch_series(list_tickers, "BOOK_VAL_PER_SH", self.factor_calendar[0].strftime('%Y%m%d'), self.factor_calendar[-1].strftime('%Y%m%d'), period="DAILY", calendar="ACTUAL", fx=None,fperiod=None, verbose=False)

        df_nan = df.isna()
        for date in df.index:
            if date in self.factor_calendar:
                not_in_df = []
                for ticker in df.columns:
                    if df_nan.loc[date, ticker]:
                            self.universe.quoteseries[ticker[:-7]].quote[date].isinindex = False
                            not_in_df.append(ticker[:-7])
                if len(not_in_df) >= len(self.tickers) - 10:
                    for ticker in self.tickers:
                        self.universe.quoteseries[ticker].DeleteQuote(date)
                        del self.universe_returns[ticker][date]
                    self.available_calendar.remove(date)
                    self.factor_calendar.remove(date)
                    try:
                        self.calendar.remove(date)
                    except:
                        pass
                    self.universe.number_dates -= 1
                    self.universe.dates.remove(date)
                    del self.momentum.long_tickers[date]
                    del self.momentum.short_tickers[date]
                    del self.size.long_tickers[date]
                    del self.size.short_tickers[date]
            else:
                df = df.drop(labels = date, axis = 0)

        list_dates = list(df.index)

        for date in self.factor_calendar:
            if not date in list_dates:
                for ticker in self.tickers:
                    self.universe.quoteseries[ticker].DeleteQuote(date)
                    del self.universe_returns[ticker][date]
                try:
                    self.available_calendar.remove(date)
                    self.factor_calendar.remove(date)
                except:
                    pass
                self.calendar.remove(date)
                self.universe.number_dates -= 1
                self.universe.dates.remove(date)
                del self.momentum.long_tickers[date]
                del self.momentum.short_tickers[date]
        
        for ticker in list_tickers:
            for date in list_dates:
                try:
                    self.universe.quoteseries[ticker[:-7]].quote[date].value = df[ticker][date]
                except:
                    self.universe.quoteseries[ticker[:-7]].quote[date].value = df[ticker][list_dates[list_dates.index(date) - 1]]
        
        for date in self.factor_calendar:
            
            dict_value = {}
            for ticker in self.inindex(date):
                dict_value[self.universe.quoteseries[ticker].quote[date].value] = ticker
                
            list_value = sorted(dict_value.keys())

# modif
            self.value.long_tickers[date] = [dict_value[i] for i in list_value[:5]]
            self.value.short_tickers[date] = [dict_value[i] for i in list_value[-5:]]
            
            self.value.returns[date] = 0.0
            
            for ticker in self.inindex(date):
                if ticker in self.value.long_tickers[date]:
                    self.value.returns[date] += self.universe_returns[ticker][date] * 1/5
                if ticker in self.value.short_tickers[date]:
                    self.value.returns[date] -= self.universe_returns[ticker][date] * 1/5
            
    def Run(self):
        """
        Runs the backtesting algorithm by rebalancing the portfolio and updating it on each trading day based on the
        reballancing frequency specified during initialization.

        """
        # Rebalance portfolio with the specified strategy
        self.ReballancePortfolio(0, self.strategy)

        # Set initial reballancing date and time to wait for the next reballancing
        last_reballancing_date = self.calendar[0]
        time_to_wait = self.reballancing_frequency

        # Loop through all trading days in the calendar
        for i in range(1, len(self.calendar)):

            # Check if it's time to rebalance the portfolio
            time_since_reballancing = timedelta(days = (self.calendar[i] - last_reballancing_date).days)
            if time_since_reballancing >= time_to_wait:
                self.ReballancePortfolio(i, self.strategy)
                last_reballancing_date = self.calendar[i]
                # If it's not time to rebalance, update the portfolio
            else:
                self.UpdatePortfolio(i)
        
    def ReballancePortfolio(self, i: int, strategy: Strategy):
        """
        Rebalances the portfolio on the specified trading day by computing the new weights for each asset in the universe
        based on the specified strategy, and then updating the portfolio accordingly. Also calculates the transaction volume
        and applies transaction fees.

        Parameters:
            i (int): index of the current trading day in the calendar
            strategy (Strategy): strategy used to compute new asset weights

        """
        # Initialize variables for portfolio information
        portfolio_return = 0.0
        portfolio_value = 0.0
        transaction_volume = 0.0
        portfolio_weights = [0.0 for i in range(self.universe.number_assets)]

        # Get the date for the current trading day and add it to the reballancing dates list
        date = self.calendar[i]
        self.reballancing_dates.append(date)

        # Compute new asset weights using the specified strategy
        portfolio_new_weights, portfolio_factor_new_weights = self.ComputeWeights(date, strategy)

        # If it's not the first trading day, calculate the portfolio return and transaction volume
        if i > 0:
            for j in range(self.universe.number_assets):
                portfolio_weights[j] = (1 + self.universe_returns[self.tickers[j]][date]) * self.portfolio[self.calendar[i-1]].weights[j]
                portfolio_return += self.universe_returns[self.tickers[j]][date] * portfolio_weights[j]
                transaction_volume += abs(portfolio_weights[j] - portfolio_new_weights[j])
            portfolio_value = self.portfolio[self.calendar[i-1]].value * (1 + portfolio_return)

            # If it's the first trading day, calculate only the transaction volume
        else:
            for j in range(self.universe.number_assets):
                transaction_volume += abs(portfolio_weights[j] - portfolio_new_weights[j])
            portfolio_value = self.basis

        # Apply transaction fees and update the portfolio with the new information
        portfolio_value = portfolio_value * (1 - transaction_volume * self.transaction_fee)
            
        self.portfolio[date] = Portfolio(date, portfolio_factor_new_weights, portfolio_new_weights, portfolio_value)
        
    def UpdatePortfolio(self, i: int):
        """

        Updates the portfolio on the specified trading day by recalculating the asset weights and factor weights based
        on the previous day's weights and returns.

            Parameters:
                i (int): index of the current trading day in the calendar

        """
        # First, we initialize some variables such as portfolio_return, portfolio_value,
        # portfolio_weights, and portfolio_factor_weights.

        portfolio_return = 0.0
        portfolio_value = 0.0
        portfolio_weights = [0.0 for i in range(self.universe.number_assets)]
        portfolio_factor_weights = [0.0 for factor in self.factors]

        # Next, we retrieve the current date from the calendar list.
        date = self.calendar[i]

        # We calculate the portfolio_return for the current date based on the returns of the assets in the portfolio.
        # Then, for each asset in the portfolio, we calculate the portfolio_weights using the current weights and the
        # asset returns.
        for j in range(self.universe.number_assets):
            portfolio_return += self.universe_returns[self.tickers[j]][date] * self.portfolio[self.calendar[i-1]].weights[j]
            portfolio_weights[j] = (1 + self.universe_returns[self.tickers[j]][date]) * self.portfolio[self.calendar[i-1]].weights[j]

        # For each factor in the portfolio, we calculate the portfolio_factor_weights based on the factor
        # returns and the previous factor weights.
        sum_factor_weights = 0.0
        for j in range(len(self.factors)):
            if self.factors[j] == "Momentum":
                portfolio_factor_weights[j] = ((1 + self.momentum.returns[date]) * self.portfolio[self.calendar[i-1]].factor_weights[j] )
                sum_factor_weights += portfolio_factor_weights[j]
            elif self.factors[j] == "Size":
                portfolio_factor_weights[j] = ((1 + self.size.returns[date]) * self.portfolio[self.calendar[i-1]].factor_weights[j] )
                sum_factor_weights += portfolio_factor_weights[j]
            elif self.factors[j] == "Value":
                portfolio_factor_weights[j] = ((1 + self.value.returns[date]) * self.portfolio[self.calendar[i-1]].factor_weights[j] )
                sum_factor_weights += portfolio_factor_weights[j]
        
        for j in range(len(self.factors)):
            portfolio_factor_weights[j] = portfolio_factor_weights[j] / sum_factor_weights

        
        # After this, we update the portfolio_value by multiplying the previous portfolio value with the current
        # portfolio return.
        portfolio_value = self.portfolio[self.calendar[i-1]].value * (1 + portfolio_return)

        # Then, we adjust the portfolio_weights to make sure they sum to 1.
        for j in range(self.universe.number_assets):
            portfolio_weights[j] = portfolio_weights[j] / (1 + portfolio_return)

        # Finally, we create a new Portfolio object with the updated values for the current date, and we add it to
        # the portfolio dictionary
        self.portfolio[date] = Portfolio(date, portfolio_factor_weights, portfolio_weights, portfolio_value)

    def ComputeWeights(self, date: datetime, strategy_name: Strategy = Strategy.NOTHING) -> list():
        """
        This function compute_weight takes in the date and strategy_name and returns a list of weights and factor
         weights based on the given strategy.
        """
        # If the strategy is NOTHING, it will return the list of weights and factor weights with zeros for all tickers
        # and factors.
        if strategy_name == Strategy.NOTHING:
            return [0 for ticker in self.tickers], [0 for factor in self.factors]
        # The inindex variable calculates which tickers were present in the index on that given date
        # (thanks to the function inindex).
        inindex = self.inindex(date)

        # If the portfolio is equally weighted, the weights of the factors are equal to 0 and the weight of each
        # asset is equally weighted.
        if strategy_name == Strategy.EQWEIGHT:

            res = list()

            for ticker in self.tickers:
                if ticker in inindex:
                    res.append(1 / len(inindex))
                else:
                    res.append(0)

            return res, [0 for factor in self.factors]

        # The function first creates a pandas DataFrame named returns with the columns of the factors "Momentum",
        # "Size", and "Value" and fills it with the respective returns for the past six months.
        returns = pd.DataFrame(columns=self.factors)

        for factor in self.factors:
            if "Momentum" == factor:
                returns_momentum = pd.Series(self.momentum.returns)
                six_months_ago = date - self.reballancing_frequency
                returns_momentum = returns_momentum[
                    (returns_momentum.index <= date) & (returns_momentum.index >= six_months_ago)]
                returns["Momentum"] = returns_momentum
            if "Size" == factor:
                returns_size = pd.Series(self.size.returns)
                six_months_ago = date - self.reballancing_frequency
                returns_size = returns_size[(returns_size.index <= date) & (returns_size.index >= six_months_ago)]
                returns["Size"] = returns_size
            if "Value" == factor:
                returns_value = pd.Series(self.value.returns)
                six_months_ago = date - self.reballancing_frequency
                returns_value = returns_value[(returns_value.index <= date) & (returns_value.index >= six_months_ago)]
                returns["Value"] = returns_value
        #  Constraints are then defined for the optimisation, which requires the sum of the weights to be equal to 1.
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

        bounds = []
        initial_weights = np.ones(len(self.factors)) / len(self.factors)
        # We set limits for the weights, which must be between 0 and 1.
        for factor in self.factors:
            bounds.append((0, 1))

        bounds = tuple(bounds)
        # We optimize based on the strategy name by minimizing variance for MINVAR.
        if strategy_name == Strategy.MINVAR:

            def variance(w):
                var = np.dot(w.T, np.dot(returns.cov(), w))
                return var

            res = minimize(variance,
                           initial_weights,
                           method='SLSQP',
                           constraints=constraints,
                           bounds=bounds)

            factor_weights = res.x

            weights_dict = {}
            for ticker in self.tickers:
                weights_dict[ticker] = 0.0

            for ticker in self.tickers:
                if "Momentum" in self.factors:
                    index_momentum = self.factors.index("Momentum")
                    if ticker in self.momentum.long_tickers[date]:
                        weights_dict[ticker] += factor_weights[index_momentum] * 1 / 5
                    elif ticker in self.momentum.short_tickers[date]:
                        weights_dict[ticker] -= factor_weights[index_momentum] * 1 / 5
                if "Size" in self.factors:
                    index_size = self.factors.index("Size")
                    if ticker in self.size.long_tickers[date]:
                        weights_dict[ticker] += factor_weights[index_size] * 1 / 5
                    elif ticker in self.size.short_tickers[date]:
                        weights_dict[ticker] -= factor_weights[index_size] * 1 / 5
                if "Value" in self.factors:
                    index_value = self.factors.index("Value")
                    if ticker in self.value.long_tickers[date]:
                        weights_dict[ticker] += factor_weights[index_value] * 1 / 5
                    elif ticker in self.value.short_tickers[date]:
                        weights_dict[ticker] -= factor_weights[index_value] * 1 / 5

            weights = []
            for ticker in self.tickers:
                weights.append(weights_dict[ticker])

            return weights, factor_weights
        # We optimize based on the strategy name by minimizing variance for MAXSHARPE.
        if strategy_name == Strategy.MAXSHARPE:

            cum_returns = (1 + returns).cumprod() - 1
            cum_returns = np.array(cum_returns.tail(1))

            def sharpe(w):
                vol = np.sqrt(np.dot(w.T, np.dot(returns.cov(), w)))
                er = np.sum(cum_returns * w)
                sr = er / vol
                return -sr

            res = minimize(sharpe,
                           initial_weights,
                           method='SLSQP',
                           constraints=constraints,
                           bounds=bounds)

            factor_weights = res.x

            weights_dict = {}
            for ticker in self.tickers:
                weights_dict[ticker] = 0.0

            for ticker in self.tickers:
                if "Momentum" in self.factors:
                    index_momentum = self.factors.index("Momentum")
                    if ticker in self.momentum.long_tickers[date]:
                        weights_dict[ticker] += factor_weights[index_momentum] * 1 / 5
                    elif ticker in self.momentum.short_tickers[date]:
                        weights_dict[ticker] -= factor_weights[index_momentum] * 1 / 5
                if "Size" in self.factors:
                    index_size = self.factors.index("Size")
                    if ticker in self.size.long_tickers[date]:
                        weights_dict[ticker] += factor_weights[index_size] * 1 / 5
                    elif ticker in self.size.short_tickers[date]:
                        weights_dict[ticker] -= factor_weights[index_size] * 1 / 5
                if "Value" in self.factors:
                    index_value = self.factors.index("Value")
                    if ticker in self.value.long_tickers[date]:
                        weights_dict[ticker] += factor_weights[index_value] * 1 / 5
                    elif ticker in self.value.short_tickers[date]:
                        weights_dict[ticker] -= factor_weights[index_value] * 1 / 5

            weights = []
            for ticker in self.tickers:
                weights.append(weights_dict[ticker])

            return weights, factor_weights

        else:
            raise NotImplementedError()

    def Plots(self):
        """
        Plots the portfolio value, portfolio factor weights, and portfolio weights
        of the backtest, as well as the drawdowns of the backtest.

        Returns:
        fig (matplotlib.figure.Figure): The matplotlib figure containing the plots.
        """
        # we create a figure and four subplots with the add_subplot function
        fig = Figure(figsize=(16,16))
        ax1 = fig.add_subplot(411)
        ax2 = fig.add_subplot(412)
        ax3 = fig.add_subplot(413)
        ax4 = fig.add_subplot(414)

        # The first for loop draws vertical lines for each rebalancing date in the ax1 and ax2 subgraphs.
        for date in self.reballancing_dates:
            ax1.axvline(date, color="gray", linestyle="--")
        # We extract the portfolio values and plot the portfolio values in ax1.
        portfolios = self.portfolio.values()
        portfolio_values = [portfolio.value for portfolio in portfolios]
        ax1.plot(self.calendar, portfolio_values)
        
        # Set the title and axis labels
        ax1.set_title("Portfolio value of the" + self.strategy.value + " backtest")
        ax1.set_ylabel('Portfolio Value')
        ax1.legend()

        for date in self.reballancing_dates:
            ax2.axvline(date, color="gray", linestyle="-")
        # The for loop iterates over the factors and plots the factor weights for each factor in the ax2 subgraph.
        for i in range(len(self.factors)):
            portfolios = self.portfolio.values()
            portfolio_weights = [portfolio.factor_weights[i] for portfolio in portfolios]
            ax2.plot(self.calendar, portfolio_weights, label=self.factors[i])

        # Set the title and axis labels
        ax2.set_title("Portfolio factor weights of the " + self.strategy.value + " backtest")
        ax2.set_ylabel('Weights')
        ax2.legend()

        for date in self.reballancing_dates:
            ax3.axvline(date, color="gray", linestyle="-")
        # The for loop iterates over the tickers in the portfolio and plots the weights of each stock in the
        # ax3 subgraph.
        for i in range(len(self.tickers)):
            portfolios = self.portfolio.values()
            portfolio_weights = [portfolio.weights[i] for portfolio in portfolios]
            ax3.plot(self.calendar, portfolio_weights, label=self.tickers[i])
            
        # Set the title and axis labels
        ax3.set_title("Portfolio weights of the " + self.strategy.value + " backtest")
        ax3.set_ylabel('Weights')
        ax3.legend()

        # We call the Perf class and pass our portfolio as an argument. Then we get the drawdown value.
        perfs = Perfs(self.portfolio)
        drawdowns = perfs.get_drawdowns()
        # Drawdowns are displayed in the ax4 subplot.
        ax4.plot(self.calendar, drawdowns)
        
        # Set the title and axis labels
        ax4.set_title('Drawdowns of the backtest')
        ax4.set_xlabel('Dates')
        ax4.set_ylabel('Drawdowns')
        ax4.legend()
        
        return fig
    
class Perfs:
    """
    This class is used to calculate various performance metrics of a portfolio.

    Attributes:
    -----------
    portfolio : dict
        A dictionary containing portfolio objects as values.

    Methods:
    --------
    get_returns()
        Returns the returns of the portfolio.
    get_overall_perf()
        Returns the overall performance of the portfolio.
    get_annualized_perf()
        Returns the annualized performance of the portfolio.
    get_daily_volatility()
        Returns the daily volatility of the portfolio.
    get_monthly_volatility()
        Returns the monthly volatility of the portfolio.
    get_annualized_volatility()
        Returns the annualized volatility of the portfolio.
    get_sharpe_ratio()
        Returns the Sharpe ratio of the portfolio.
    get_drawdowns()
        Returns the drawdowns of the portfolio.
    get_maximum_drawdown()
        Returns the maximum drawdown of the portfolio.
    get_historical_var(alpha=0.05)
        Returns the historical value at risk of the portfolio.
    get_performances_dico()
        Returns a dictionary containing various performance metrics of the portfolio.
    """
    def __init__(self, portfolio):
        """
        Initializes the class with a portfolio.

        Parameters:
        ----------
        portfolio: dict
            Dictionary of portfolios.

        Attributes:
        ----------
        self.portfolio : dict
            Dictionary of portfolios.
        self.list_portfolio_objects : list
            List of portfolio objects.
        self.list_portfolio_values : list
            List of portfolio values.
        self.list_portfolio_weights : list
            List of portfolio weights.

        """
        self.portfolio = portfolio
        self.list_portfolio_objects = self.portfolio.values()
        self.list_portfolio_values = [portfolio.value for portfolio in self.list_portfolio_objects]
        self.list_portfolio_weights = [portfolio.weights for portfolio in self.list_portfolio_objects]

    def get_returns(self):
        """
        Calculates portfolio returns.

        Returns:
        -------
        Returns of the portfolio.

        """
        returns = np.array(self.list_portfolio_values[1:]) / np.array(self.list_portfolio_values[:-1]) - 1
        return returns

    def get_overall_perf(self):
        """
        Calculates the overall performance of the portfolio.

        Returns:
        -------
        overall_perf: float
            Overall performance of the portfolio.

        """
        return self.list_portfolio_values[-1] / self.list_portfolio_values[0] - 1

    def get_annualized_perf(self):
        """
        Calculates the overall performance of the portfolio.

        Returns:
        -------
        overall_perf: float
            Overall performance of the portfolio.

        """
        return (1+self.get_overall_perf()) ** (252/len(self.list_portfolio_values)) - 1

    def get_daily_volatility(self):
        """
        Calcule la volatilité quotidienne du portefeuille.

        Returns:
        -------
        daily_volatility : float
            Volatilité quotidienne du portefeuille.

        """
        returns = self.get_returns()
        return np.std(returns)

    def get_monthly_volatility(self):
        """
        Calculates the monthly volatility of the portfolio.

        Returns:
        -------
        monthly_volatility : float
            Monthly portfolio volatility.

        """
        returns = self.get_returns()
        return np.std(returns) * np.sqrt(21)

    def get_annualized_volatility(self):
        """
        Calculates the annualised volatility of the portfolio.

        Returns:
        -------
        annualized_volatility : float
            Annualized portfolio volatility.

        """
        returns = self.get_returns()
        return np.std(returns) * np.sqrt(252)

    def get_sharpe_ratio(self):
        """
        Calculates the Sharpe ratio of the portfolio.

        Returns:
        -------
        sharpe_ratio : float
            Sharpe ratio of the portfolio.

        """
        return self.get_annualized_perf() / self.get_annualized_volatility()
    
    def get_drawdowns(self):
        """
        Returns a list of drawdowns of the portfolio, which is defined as the loss
        in value from a peak value to a subsequent trough value over a specific
        time period, expressed as a percentage.

        Returns:
        -------
            list: A list of drawdowns, where each drawdown is expressed as a percentage.
        """
        peak = self.list_portfolio_values[0]
        drawdowns = []
        for value in self.list_portfolio_values:
            if value > peak:
                drawdowns.append(0)
                peak = value
            else:
                drawdowns.append((value - peak) / peak)
        return drawdowns
        

    def get_maximum_drawdown(self):
        """
        Returns the maximum drawdown of the portfolio, which is defined as the
        maximum loss from a peak value to a subsequent trough value over a specific
        time period, expressed as a percentage.

        Returns:
        -------
            float: The maximum drawdown expressed as a percentage.
        """
        drawdowns = self.get_drawdowns()
        return min(drawdowns)

    def get_historical_var(self, alpha=0.05):
        """
        Calculate the historical value-at-risk (VaR) at the specified alpha level.

        Parameters:
            alpha (float): The confidence level at which to calculate VaR. Default is 0.05.

        Returns:
            float: The historical VaR at the specified alpha level.
        """
        returns = self.get_returns()
        return np.percentile(returns, alpha * 100)

    def get_performances_dico(self):
        """
        Returns a dictionary containing various performance metrics of the portfolio,
        such as overall performance, annualized performance, volatility, Sharpe ratio,
        maximum drawdown, and historical Value-at-Risk.

        Returns:
        -------
            dict: A dictionary containing various performance metrics, where each metric
            is represented as a key-value pair, with the metric name as the key and the
            corresponding value as the value.
        """
        # We create a dictionary containing the results for each portfolio
        perf_dict = {
            "Overall Perf": f'{round(self.get_overall_perf()*100, 2)}%',
            "Annualized Performance": f'{round(self.get_annualized_perf()*100, 2)}%',
            "Daily volatility": f'{round(self.get_daily_volatility()*100, 2)}%',
            "Monthly Volatility": f'{round(self.get_monthly_volatility()*100, 2)}%',
            "Annualized Volatility": f'{round(self.get_annualized_volatility()*100, 2)}%',
            "Sharpe Ratio": f'{round(self.get_sharpe_ratio()*100, 2)}%',
            "Maximum Drawdown": f'{round(self.get_maximum_drawdown()*100, 2)}%',
            "Historical Var": f'{round(self.get_historical_var()*100, 2)}%'
        }

        return perf_dict
        

class BBG(object):
    """
    Authors: Daniel Dantas, Gustavo Amarante, Gustavo Soares, Wilson Felicio
    This class is a wrapper around the Bloomberg API. To work, it requires an active bloomberg terminal running on
    windows (the API is not comaptible with other OS), a python 3.6 environment and the installation of the bloomberg
    API. Check out the guides on our github repository to learn how to install the API.
    """

    @staticmethod
    def fetch_series(securities, fields, startdate, enddate, period="DAILY", calendar="ACTUAL", fx=None,
                     fperiod=None, verbose=False):
        """
        Fetches time series for given tickers and fields, from startdate to enddate.
        Output is a DataFrame with tickers on the columns. If a single field is passed, the index are the dates.
        If a list of fields is passed, a multi-index DataFrame is returned, where the index is ['FIELD', date].
        Requests can easily get really big, this method allows for up to 30k data points.
        This replicates the behaviour of the BDH function of the excel API
        :param securities: str or list of str
        :param fields: str or list of str
        :param startdate: str, datetime or timestamp
        :param enddate: str, datetime or timestamp
        :param period: 'DAILY', 'WEEKLY', 'MONTHLY', 'QUARTERLY', 'SEMI ANNUAL' OR 'YEARLY'. Periodicity of the series
        :param calendar: 'ACTUAL', 'CALENDAR' or 'FISCAL'
        :param fx: str with a currency code. Converts the series to the chosen currency
        :param fperiod: ???
        :param verbose: prints progress
        :return:  DataFrame or Multi-index DataFrame (if more than one field is passed)
        """

        startdate = BBG._assert_date_type(startdate)
        enddate = BBG._assert_date_type(enddate)

        bbg_start_date = BBG._datetime_to_bbg_string(startdate)
        bbg_end_date = BBG._datetime_to_bbg_string(enddate)

        if startdate > enddate:
            ValueError("Start date is later than end date")

        session = blpapi.Session()

        if not session.start():
            raise ConnectionError("Failed to start session")

        try:
            if not session.openService("//blp/refdata"):
                raise ConnectionError("Failed to open //blp/refdat")

            # Obtain the previously opened service
            refdata_service = session.getService("//blp/refdata")

            # Create and fill the request for historical data
            request = refdata_service.createRequest("HistoricalDataRequest")

            # grab securities
            if type(securities) is list:
                for sec in securities:
                    request.getElement("securities").appendValue(sec)
            else:
                request.getElement("securities").appendValue(securities)

            # grab fields
            if type(fields) is list:
                for f in fields:
                    request.getElement("fields").appendValue(f)
            else:
                request.getElement("fields").appendValue(fields)

            request.set("periodicityAdjustment", calendar)
            request.set("periodicitySelection", period)
            request.set("startDate", bbg_start_date)
            request.set("endDate", bbg_end_date)
            request.set("maxDataPoints", 30000)

            if not (fx is None):
                request.set("currency", fx)

            if not (fperiod is None):
                overrides_bdh = request.getElement("overrides")
                override1_bdh = overrides_bdh.appendElement()
                override1_bdh.setElement("fieldId", "BEST_FPERIOD_OVERRIDE")
                override1_bdh.setElement("value", fperiod)

            if verbose:
                print("Sending Request:", request.getElement("date").getValue())

            # send request
            session.sendRequest(request)

            # process received response
            results = {}

            while True:
                ev = session.nextEvent()

                for msg in ev:

                    if verbose:
                        print(msg)

                    if msg.messageType().__str__() == "HistoricalDataResponse":
                        sec_data = msg.getElement("securityData")
                        sec_name = sec_data.getElement("security").getValue()
                        field_data = sec_data.getElement("fieldData")

                        if type(fields) is list:

                            results[sec_name] = {}

                            for day in range(field_data.numValues()):

                                fld = field_data.getValue(day)

                                for fld_i in fields:
                                    if fld.hasElement(fld_i):
                                        results[sec_name] \
                                            .setdefault(fld_i, []).append([fld.getElement("date").getValue(),
                                                                           fld.getElement(fld_i).getValue()])
                        else:
                            results[sec_name] = []
                            for day_i in range(field_data.numValues()):
                                fld = field_data.getValue(day_i)
                                results[sec_name].append([
                                    fld.getElement("date").getValue(),
                                    fld.getElement(fields).getValue()])

                if ev.eventType() == blpapi.Event.RESPONSE:  # Response completly received, break out of the loop
                    break

        finally:
            session.stop()

        if not type(securities) is list:
            results = results[securities]

        # parse the results as a DataFrame
        df = pd.DataFrame()

        if not (type(securities) is list) and not (type(fields) is list):
            # single ticker and single field
            # returns a dataframe with a single column
            results = np.array(results)
            df[securities] = pd.Series(index=pd.to_datetime(results[:, 0]), data=results[:, 1])

        elif (type(securities) is list) and not (type(fields) is list):
            # multiple tickers and single field
            # returns a single dataframe for the field with the ticker on the columns

            for tick in results.keys():
                aux = np.array(results[tick])

                if len(aux) == 0:
                    df[tick] = np.nan
                else:
                    df = pd.concat([df, pd.Series(index=pd.to_datetime(aux[:, 0]), data=aux[:, 1], name=tick)],
                                   axis=1, join='outer', sort=True)

        elif not (type(securities) is list) and (type(fields) is list):
            # single ticker and multiple fields
            # returns a single dataframe for the ticker with the fields on the columns

            for fld in results.keys():
                aux = np.array(results[fld])
                df[fld] = pd.Series(index=pd.to_datetime(aux[:, 0]), data=aux[:, 1])

        else:
            # multiple tickers and multiple fields
            # returns a multi-index dataframe with [field, ticker] as index

            for tick in results.keys():

                for fld in results[tick].keys():
                    aux = np.array(results[tick][fld])
                    df_aux = pd.DataFrame(data={'FIELD': fld,
                                                'TRADE_DATE': pd.to_datetime(aux[:, 0]),
                                                'TICKER': tick,
                                                'VALUE': aux[:, 1]})
                    df = df.append(df_aux)

            df['VALUE'] = df['VALUE'].astype(float, errors='ignore')

            df = pd.pivot_table(data=df, index=['FIELD', 'TRADE_DATE'], columns='TICKER', values='VALUE')

        return df

    @staticmethod
    def fetch_contract_parameter(securities, field):
        """
        Grabs a characteristic of a contract, like maturity dates, first notice dates, strikes, contract sizes, etc.
        Returns a DataFrame with the tickers on the index and the field on the columns.
        This replicates the behaviour of the BDP Function from the excel API.
        OBS: For now, it only allows for a single field. An extension that allows for multiple fields is a good idea.
        :param securities: str or list of str
        :param field: str
        :return: DataFrame
        """

        # TODO allow for a list of fields

        session = blpapi.Session()
        session.start()

        if not session.openService("//blp/refdata"):
            raise ConnectionError("Failed to open //blp/refdat")

        service = session.getService("//blp/refdata")
        request = service.createRequest("ReferenceDataRequest")

        if type(securities) is list:

            for each in securities:
                request.append("securities", str(each))

        else:
            request.append("securities", securities)

        request.append("fields", field)
        session.sendRequest(request)

        name, val = [], []
        end_reached = False
        while not end_reached:

            ev = session.nextEvent()

            if ev.eventType() == blpapi.Event.RESPONSE or ev.eventType() == blpapi.Event.PARTIAL_RESPONSE:

                for msg in ev:

                    for i in range(msg.getElement("securityData").numValues()):
                        sec = str(msg.getElement("securityData").getValue(i).getElement(
                            "security").getValue())  # here we get the security
                        name.append(sec)

                        value = msg.getElement("securityData").getValue(i).getElement("fieldData").getElement(
                            field).getValue()
                        val.append(value)  # here we get the field value we have selected

            if ev.eventType() == blpapi.Event.RESPONSE:
                end_reached = True
                session.stop()

        df = pd.DataFrame(val, columns=[field], index=name)

        return df

   
    @staticmethod
    def fetch_index_weights(index_name, ref_date):
        """
        Given an index (e.g. S&P500, IBOV) and a date, it returns a DataFrame of its components as the index an
        their respective weights as the value for the given date.
        :param index_name: str
        :param ref_date: str, datetime or timestamp
        :return: DataFrame
        """

        ref_date = BBG._assert_date_type(ref_date)

        session = blpapi.Session()

        if not session.start():
            raise ConnectionError("Failed to start session.")

        if not session.openService("//blp/refdata"):
            raise ConnectionError("Failed to open //blp/refdat")

        service = session.getService("//blp/refdata")
        request = service.createRequest("ReferenceDataRequest")

        request.append("securities", index_name)
        request.append("fields", "INDX_MWEIGHT_HIST")

        overrides = request.getElement("overrides")
        override1 = overrides.appendElement()
        override1.setElement("fieldId", "END_DATE_OVERRIDE")
        override1.setElement("value", ref_date.strftime('%Y%m%d'))
        session.sendRequest(request)  # there is no need to save the response as a variable in this case

        end_reached = False
        df = pd.DataFrame()
        while not end_reached:

            ev = session.nextEvent()

            if ev.eventType() == blpapi.Event.RESPONSE:

                for msg in ev:

                    security_data = msg.getElement('securityData')
                    security_data_list = [security_data.getValueAsElement(i) for i in range(security_data.numValues())]

                    for sec in security_data_list:

                        field_data = sec.getElement('fieldData')
                        field_data_list = [field_data.getElement(i) for i in range(field_data.numElements())]

                        for fld in field_data_list:

                            for v in [fld.getValueAsElement(i) for i in range(fld.numValues())]:

                                s = pd.Series()

                                for d in [v.getElement(i) for i in range(v.numElements())]:
                                    s[str(d.name())] = d.getValue()

                                df = df.append(s, ignore_index=True)

                if not df.empty:
                    df.columns = ['', ref_date]
                    df = df.set_index(df.columns[0])

                end_reached = True

        return df

    
    @staticmethod
    def _assert_date_type(input_date):
        """
        Assures the date is in datetime format
        :param input_date: str, timestamp, datetime
        :return: input_date in datetime format
        """

        if not (type(input_date) is dt.date):

            if type(input_date) is pd.Timestamp:
                input_date = input_date.date()

            elif type(input_date) is str:
                input_date = pd.to_datetime(input_date).date()

            else:
                raise TypeError("Date format not supported")

        return input_date

    @staticmethod
    def _datetime_to_bbg_string(input_date):
        """
        converts datetime to string in bloomberg format
        :param input_date:
        :return:
        """
        return str(input_date.year) + str(input_date.month).zfill(2) + str(input_date.day).zfill(2)


def datahisto(start_date: datetime, end_date: datetime, index_name: str):
    """
    This function returns three elements including a dictionary and two arrays.

    Args :
    - start_date (datetime) : This is the start date
    - end_date (datetime) : This is the end date
    - index_name (str) : This is the name of the index


    Returns:
    tickers_dico : It is a dictionary whose key is a date and the value
    associated to this date is the list of tickers
    tab_tickers : It is an array containing all the tickers.
    tab_date : It is an array containing all the dates.

    """

    index_names = {
        "SPX Index": "US",
        "INDU Index": "US",
        "COMP Index": "US",
        "UKX Index": "GB",
        "CAC Index": "FR",
        "DAX Index": "DE",
        "SX5E Index": "EU",
        "NKY Index": "JP",
        "HSI Index": "HK",
        "SHCOMP Index": "CN",
        "AS51 Index": "AU",
        "SENSEX Index": "IN",
        "S&P/TSX Composite Index": "CA",
        "IPC Index": "MX",
        "IBEX 35 Index": "ES",
        "FTSE MIB Index": "IT",
        "SMI Index": "CH",
        "RTS Index": "RU",
        "Bovespa Index": "BR",
        "S&P/ASX 200 Index": "AU",
        "KOSPI Index": "KR",
        "TSEC Weighted Index": "TW",
        "SET Index": "TH",
        "NZX 50 Index": "NZ"
    }

    try:
        country = index_names[index_name]
    except:
        raise ValueError()

    tickers_dico = {}  # dico[date] = list of tickers

    list_date = get_business_days(start_date, end_date, country)
    list_tickers = []  # All tickers without duplicates

    for date in list_date:

        if date == list_date[0]:

            df = BBG.fetch_index_weights(index_name, date.strftime('%Y%m%d'))
            tickers_dico[date] = df.index
            for i in df.index:
                list_tickers.append(i)
            last_fetched_month = date.month

        # The composition of the S&P changes every 3 months. We have therefore chosen to retrieve the composition
        # of the indices every 3 months (once a quarter).
        elif date.month % 3 == 0 and last_fetched_month < date.month:

            df = BBG.fetch_index_weights(index_name, date.strftime('%Y%m%d'))
            tickers_dico[date] = df.index
            for i in df.index:
                list_tickers.append(i)
            last_fetched_month = date.month

        else:
            tickers_dico[date] = tickers_dico[list_date[list_date.index(date) - 1]]

    list_tickers = list(set(list_tickers))

    return tickers_dico, list_tickers, list_date


def datahistobool(tickers_dico, list_tickers, list_date):
    """
    This function returns a dictionary consisting of boolean values for each ticker at each date.

    Args :
    - tickers_dico (dict): a dictionary whose key is a date and the value associated with this date is the list of tickers
    - list_tickers (list): an array containing all the tickers
    - list_date (list): an array containing all the dates

    Returns:
    - A dictionary whose keys are dates. To each date is associated a sub-dictionary.
      This sub-dictionary has for key a ticker and the associated value is True if
      the ticker exists at this date, False otherwise.

    """

    tickers_dico_bool = {}

    for ticker in list_tickers:

        sous_dico = {}

        for date in list_date:
            if ticker in tickers_dico[date]:
                sous_dico[date] = True
            else:
                sous_dico[date] = False

        tickers_dico_bool[ticker] = sous_dico

    return tickers_dico_bool


def get_quotes(tickers_dico_bool, start: datetime, end: datetime):
    """
    The get_quotes function retrieves the closing price data for a given ticker list, for a specified date range.
     The function takes as input a tickers_dico_bool dictionary whose keys are ticker names and values are
     sub-dictionaries containing dates and booleans, indicating whether the ticker is present on the given date or not.

    The function first extracts the list of tickers from tickers_dico_bool and adds the suffix "Equity" to get the
    Bloomberg codes of the assets. Data is then extracted from Bloomberg for this ticker list, for the specified
    date range.

    The function then creates a Universe object and a QuotesSerie object for each ticker.
    These objects are populated with the price data retrieved from Bloomberg, and for each date, a Quote object
    is created for the corresponding ticker. The Quote object contains information about the date, the ticker name,
    the closing price and a boolean indicating whether the ticker was present on that date or not.

    Finally, the Quote objects are stored in the QuotesSerie object, which is added to the Universe object.
    The Universe object containing all the data is returned as output.

    Args:
    - tickers_dico_bool (dict): a dictionary whose keys are tickers and each value is a sub-dictionary.
                                Each sub-dictionary has for key a date and the associated value is True
                                 if the ticker exists at this date, False otherwise.
    - start (datetime): the start date of the historical period
    - end (datetime): the end date of the historical period

    Returns:
    -  Universe
    """

    list_tickers = list(tickers_dico_bool.keys())
    list_tickers = [ticker + " Equity" for ticker in list_tickers]

    df = BBG.fetch_series(list_tickers, "PX_LAST", start.strftime('%Y%m%d'), end.strftime('%Y%m%d'), period="DAILY",
                          calendar="ACTUAL", fx=None, fperiod=None, verbose=False)
    
    list_tickers = df.columns
    list_dates = df.index

    universe = Universe()

    for ticker in list_tickers:

        quote_serie = QuotesSerie()

        for date in list_dates:
            try:
                if df[ticker][date] == np.nan:
                    quote = Quote(date, ticker[:-7], df[ticker][date], False)
                else:
                    quote = Quote(date, ticker[:-7], df[ticker][date], tickers_dico_bool[ticker[:-7]][date])
                quote_serie.AddQuote(quote)
            except:
                pass

        universe.AddQuotesSerie(quote_serie)

    return universe


class BacktestInterface:
    """
    The BacktestInterface class is a class that defines a graphical interface that allows the user to specify various
     parameters needed to perform backtesting. The various elements of the interface are :

    - Two input fields for the start date and end date of the backtesting.
    - A drop-down menu to select the backtesting strategy from three options: Min Variance, Max Sharpe and Eq Weight.
    - Checkboxes to select the different factors that will be used in the backtesting. The proposed factors are
     Momentum, Value and Size.
    - An input field for the ticker symbol of the index to be used in backtesting.
    - An input field for the transaction costs in basis points.
    - An input field for the frequency of portfolio reallocation in backtesting, with a drop-down menu to specify the
     time unit (days, weeks, months).
    - A submit button to start backtesting with the selected parameters.
    """
    def __init__(self):
        
        self.root = Tk()
        self.root.title('Backtest Interface')
        #self.root.resizable(height=False, width = False)
        
        # Add a label for start date
        self.start_date_label = Label(self.root, text="Start Date (dd/mm/yyyy)")
        self.start_date_label.pack()
        
        # Add an entry box for start date
        self.start_date_entry = Entry(self.root)
        self.start_date_entry.pack()
        
        # Add a label for end date
        self.end_date_label = Label(self.root, text="End Date (dd/mm/yyyy)")
        self.end_date_label.pack()
        
        # Add an entry box for end date
        self.end_date_entry = Entry(self.root)
        self.end_date_entry.pack()
        
        # Add a label for strategy
        self.strategy_label = Label(self.root, text="Select Strategy")
        self.strategy_label.pack()
        
        # Add a dropdown list for strategy
        strategy_options = ["Min Variance", "Max Sharpe", "Eq Weight"]
        self.strategy_var = StringVar(self.root)
        self.strategy_var.set(strategy_options[1])
        self.strategy_dropdown = OptionMenu(self.root, self.strategy_var, *strategy_options)
        self.strategy_dropdown.pack()
        
        # Add check label for factors
        self.factors_label = Label(self.root, text="Factors")
        self.factors_label.pack()
        
        self.momentum_var = BooleanVar(value=False)
        self.momentum_check = Checkbutton(self.root, text="Momentum", variable=self.momentum_var)
        self.momentum_check.pack()
        
        self.value_var = BooleanVar(value=False)
        self.value_check = Checkbutton(self.root, text="Value", variable=self.value_var)
        self.value_check.pack()
        
        self.size_var = BooleanVar(value=False)
        self.size_check = Checkbutton(self.root, text="Size", variable=self.size_var)
        self.size_check.pack()
        
        # Add a label for index ticker
        self.ticker_label = Label(self.root, text="Ticker of the Index")
        self.ticker_label.pack()
        
        # Add an entry box for index ticker
        self.ticker_entry = Entry(self.root)
        self.ticker_entry.pack()
        
        # Add a label for transaction costs
        self.transaction_costs_label = Label(self.root, text="Transaction Costs (in basis points)")
        self.transaction_costs_label.pack()
        
        # Add an entry box for transaction costs
        self.transaction_costs_entry = Entry(self.root)
        self.transaction_costs_entry.pack()
        
        # Add a label for reallocation frequency
        self.reallocation_frequency_label = Label(self.root, text="Reallocation Frequency (ex: 2 Months)")
        self.reallocation_frequency_label.pack()
        
        # Add an entry box for reallocation frequency value
        self.reallocation_frequency_value_entry = Entry(self.root)
        self.reallocation_frequency_value_entry.pack(side=LEFT)
        
        # Add a dropdown list for reallocation frequency units
        reallocation_frequency_unit_options = ["Days", "Weeks", "Months"]
        self.reallocation_frequency_unit_var = StringVar(self.root)
        self.reallocation_frequency_unit_var.set(reallocation_frequency_unit_options[0])
        self.reallocation_frequency_unit_dropdown = OptionMenu(self.root, self.reallocation_frequency_unit_var, *reallocation_frequency_unit_options)
        self.reallocation_frequency_unit_dropdown.pack(side=LEFT)
        
        def submit_func():
            start_date = self.start_date_entry.get()
            end_date = self.end_date_entry.get()
            strategy = self.strategy_var.get()
            momentum = self.momentum_var.get()
            size = self.size_var.get()
            value = self.value_var.get()
            ticker = self.ticker_entry.get()
            factors = list()
            if momentum == True:
                factors.append("Momentum")
            if size == True:
                factors.append("Size")
            if value == True:
                factors.append("Value")
            transaction_costs = self.transaction_costs_entry.get()
            reallocation_frequency_value = self.reallocation_frequency_value_entry.get()
            reallocation_frequency_unit = self.reallocation_frequency_unit_var.get()
            self.Func(start_date, end_date, strategy, factors, ticker, transaction_costs, reallocation_frequency_value, reallocation_frequency_unit)
        
        # Add a button to submit the form
        self.submit_button = Button(self.root, text="Submit", command=submit_func)
        self.submit_button.pack()
        
        self.root.mainloop()
    
    # Define the function that will be called when the user clicks the Submit button
    def Func(self, start_date, end_date, strategy, factors, ticker, transaction_costs, reallocation_frequency_value, reallocation_frequency_unit):

        print("Start Date:", start_date)
        print("End Date:", end_date)
        print("Strategy:", strategy)
        print("Factors:", factors)
        print("Ticker:", ticker)
        print("Transaction Costs:", transaction_costs)
        print("Reallocation Frequency:", reallocation_frequency_value, reallocation_frequency_unit)
        
        try:
            try:
                start = datetime.strptime(start_date, "%d/%m/%Y")
                start = datetime(start.year - 2, start.month, start.day)
                end = datetime.strptime(end_date, "%d/%m/%Y")
                if start >= end:
                    raise ValueError("Start date must be before the end date")
            except:
                raise ValueError("Wrong date input(s)")
                
            try:
                transaction_costs = float(transaction_costs)/10000
            except:
                raise ValueError("Incorrect transaction cost value")
                
            if strategy == "Min Variance":
                my_strategy = Strategy.MINVAR
            elif strategy == "Max Sharpe":
                my_strategy = Strategy.MAXSHARPE
            elif strategy == "Eq Weight":
                my_strategy = Strategy.EQWEIGHT
            else:
                raise ValueError("Incorrect Strategy")
                
            if factors == list():
                raise ValueError("No Factor Selected")  
            
            try:
                try:
                    reallocation_value = int(reallocation_frequency_value)
                    reallocation_unit = reallocation_frequency_unit
                except:
                    raise ValueError("You must enter a number and a unit for the reallocation frequency")
                if reallocation_value > 12 and reallocation_unit == "Months":
                    ValueError("The reallocation frequency can be max 1 year")
                if reallocation_value > 52 and reallocation_unit == "Weeks":
                    ValueError("The reallocation frequency can be max 1 year")
                if reallocation_value > 252 and reallocation_unit == "Days":
                    ValueError("The reallocation frequency can be max 1 year")
            except ValueError as e:
                raise e
                
            try:
                list_tickers = list()
                tickers_dico, list_tickers, list_date = datahisto(start, end, ticker)

                if list_tickers == list():
                    ValueError()
            except:
                raise ValueError("Index member of index ticker not found")
            
            
                
        except ValueError as e:
            messagebox.showerror("Error", str(e))

        tickers_dico_bool = datahistobool(tickers_dico, list_tickers, list_date)
        invest_universe = get_quotes(tickers_dico_bool, start, end)
    
        invest_universe.UniformizeDates()
    
        my_backtest = Backtest(invest_universe, my_strategy, factors, reallocation_value, reallocation_unit, 100, transaction_costs)
        
        # Create new window to display figures and canvas objects
        plots_window = Toplevel(self.root)
        plots_window.geometry("1000x800")
        plots_window.title("Results")
        
        fig = my_backtest.Plots()
        canvas = FigureCanvasTkAgg(fig, master=plots_window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

        # We create a new window called performance to display our performance metrics.
        perf_window = Toplevel()
        perf_window.title("Performance")

        # Here, we pass our portfolio as an argument in the Perfs box in order to calculate the performance.
        perf = Perfs(my_backtest.portfolio)
        # We retrieve the dictionary which has as key the perfromance indicators and as value the performance metrics.
        perf_dico = perf.get_performances_dico()
        
        perf_str = ''
        # We create a new sheet to display our performance. We loop through the keys of our dictionary to retrieve
        # the name of the perfromance indicators (=key of the dictionary) and the associated value.
        for key, value in perf_dico.items():
            perf_str += f'{key}: {value}\n'

        # We write our performance in the sheet.
        text = Text(perf_window)
        text.pack()
        text.insert(END, perf_str)
# The backtest interface is launched.
my_interface = BacktestInterface()
        
