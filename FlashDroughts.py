import requests
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import scipy.stats as stats
import seaborn as sns
import matplotlib.dates as mdates
from datetime import datetime
import matplotlib.gridspec as gridspec
import scipy.stats
from scipy.stats import linregress 
import spei as si
from scipy.stats import pearson3, gamma, norm
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib import cm

class KDEhistogram:
    """ 
    Class to create kernel density estimation (KDE) histograms for 
    Potential Evapotranspiration (PET), PET residuals, 
    precipitation (Precip), and volumetric water content (VWC)
    
    Attributes:
    sdate (str or datetime): Start date of the analysis period.
    edate (str or datetime): End date of the analysis period.
    lat (float): Latitude of the location.
    lon (float): Longitude of the location.
        
    """
    def __init__(self, sdate, edate, lat, lon):
        """ 
        Initialize a KDEhistogram object for a specific location and time window.
        ONLY WORKS FOR LAT AND LON IN THE NORTHEAST UNITED STATE AND FROM 2002 - PRESENT
        
        Args:
            sdate (str): Start date in 'YYYY-MM-DD' format.
            edate (str): End date in 'YYYY-MM-DD' format.
            lat (float): Latitude in decimal degrees. 
            lon (float): Longitude in decimal degrees.
        """
        self.sdate = sdate
        self.edate = edate
        self.lat = lat
        self.lon = lon

    def url_to_dataframe(self):
        """
        Fetches daily climate data (precipitation, PET, VWC) from the RCC-ACIS API
        and converts it into a pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing daily values with the following columns:
                - 'date' (str): Date in 'YYYY-MM-DD' format
                - 'precip' (float): Precipitation (in)
                - 'pet' (float): Potential evapotranspiration (in)
                - 'vwc' (float): Volumetric water content (unitless)

        Notes:
            - Only produces dates from March 1 to October 31
            - The first value of the VWC time series is always NaN and is removed from the DataFrame.
            - The API requires a valid token; replace the token string before production use.
        
        Raises:
            requests.exceptions.RequestException: If the API request fails.
            ValueError: If the response data is missing expected keys.
        """

        url = f"https://csf-irrigation-api-worker.rcc-acis.workers.dev/vwc/?sdate={self.sdate}&edate={self.edate}&lat={self.lat}&lon={self.lon}&token=api-4a0607-token"

        req = requests.get(url)
        data = req.text

        data = json.loads(data)
        df = pd.DataFrame({
            "date": data["dates"],
            "precip": data["precip"],
            "pet": data["pet"],
            "vwc": data["vwc"]
        })

        df = df[1::] # The first value of the VWC is always NAN
        return df


    def _parse_and_clean(self, df):
        """
        Parses and cleans the raw DataFrame returned by the API.

        Operations performed:
            - Converts 'date' column to datetime objects.
            - Converts 'pet', 'precip', and 'vwc' columns to numeric types.
            - Converts PET from inches to millimeters (by multiplying by -25.4).
            - PET becomes negative for onset calculations
            - Sets 'date' as the index.
            - Removes any duplicate dates from the index.

        Args:
            df (pd.DataFrame): Raw DataFrame containing 'date', 'pet', 'precip', and 'vwc' columns.

        Returns:
            pd.DataFrame: Cleaned and indexed DataFrame with numeric types and no duplicate dates.
        """
        df['date'] = pd.to_datetime(df['date'])
        df['pet'] = pd.to_numeric(df['pet'], errors='coerce') * -25.4  # inches to mm
        df['precip'] = pd.to_numeric(df['precip'], errors='coerce')
        df['vwc'] = pd.to_numeric(df['vwc'], errors='coerce')

        df = df.set_index('date')

        df = df[~df.index.duplicated()]
        return df

    def _assign_season(self, df):
        """
        Assigns a meteorological season label to each row in the DataFrame based on its date index.

        Seasons are defined as:
            - Spring: March (3) to May (5)
            - Summer: June (6) to August (8)
            - Fall: September (9) to October (10)
            - Winter: November (11) to February (2)

        Args:
            df (pd.DataFrame): DataFrame with a datetime index.

        Returns:
            pd.DataFrame: Original DataFrame with an added 'season' column containing
                        one of {'spring', 'summer', 'fall', 'winter'}.
        """
        def get_season(date): # function to assign season to daily values
            month = date.month
            if 3 <= month <= 5:
                return 'spring'
            elif 6 <= month <= 8:
                return 'summer'
            elif 9 <= month <= 10:
                return 'fall'
            else:
                return 'winter'

        df['season'] = df.index.to_series().apply(get_season) # creates season column
        return df

    def _resample_weekly(self, df):
        """
        Resamples daily climate data to weekly resolution using Tuesday as the week's end ('W-TUE') and drops incomplete weeks.

        Operations:
            - Sums PET and precipitation over each week.
            - Takes the first VWC value in each week.(Tuesday value)
            - Assigns the most common season label (mode) within each week.
            - Sets negative precipitation values to 0 (assumed to be data errors).
            - Adds a 'year' column based on the resampled index.

        Args:
            df (pd.DataFrame): Daily DataFrame with datetime index and columns:
                ['pet', 'precip', 'vwc', 'season'].

        Returns:
            pd.DataFrame: Weekly-resampled DataFrame with columns:
                ['pet', 'precip', 'vwc', 'season', 'year'].
        """
        weekly_pet = df['pet'].resample('W-TUE').sum() # weekly sums starting on a Tuesday
        weekly_precip = df['precip'].resample('W-TUE').sum() # weekly sums starting on a Tuesday
        weekly_vwc = df['vwc'].resample('W-TUE').first() # Takes the Tuesday value

        def safe_majority(x): # function to find the most frequent season in each week
            counts = x.value_counts()
            return counts.idxmax() if not counts.empty else np.nan

        season = df['season'].resample('W-TUE').agg(safe_majority) # assigns the correct season
    
        combined = pd.concat([weekly_pet, weekly_precip, weekly_vwc, season], axis=1)
        combined.columns = ['pet', 'precip', 'vwc', 'season']
        combined.loc[combined['precip'] < 0, 'precip'] = 0 # no negative precip vals
        combined['year'] = combined.index.year # creates year column

        full_weeks = df.resample('W-TUE').count()['pet'] == 7 # Use a count of the original daily records in each weekly bin
        combined = combined[full_weeks]  # Drop weeks that don't have all 7 days of data
        return combined

    def _apply_rolling(self, df, column, weeks):
        """
        Applies a rolling sum to a specified column over a defined number of weeks.

        Args:
            df (pd.DataFrame): Input DataFrame containing the column to be smoothed.
            column (str): Name of the column to apply the rolling sum to ('vwc', 'precip', 'pet').
            weeks (int): Size of the rolling window (in number of time steps, weekly).

        Returns:
            pd.DataFrame: Original DataFrame with a new column added:
                        '{column} {weeks} rolling' containing the rolling sum.

        Notes:
            - Uses `min_periods=weeks`, so the result will be NaN until a full window is available.
            - Assumes the DataFrame index is datetime-based and properly resampled (weekly).
        """
        df[f'{column} {weeks} rolling'] = df[column].rolling(window=weeks, min_periods=weeks).sum()
        return df


    def weekly_dataframe(self, column, weeks):
        """
        Builds a cleaned, season-tagged, weekly-resampled DataFrame with rolling sums
        for a specific variable.

        Pipeline steps:
            1. Fetches raw daily climate data from the API.
            2. Parses and cleans the data (e.g., type conversions, unit corrections).
            3. Assigns meteorological seasons to each row.
            4. Resamples the data to weekly resolution (ending on Tuesdays).
            5. Computes a rolling sum over the specified number of weeks.

        Args:
            column (str): Name of the column to compute rolling sums on ('vwc', 'precip', 'pet').
            weeks (int): Size of the rolling window (in weekly time steps).

        Returns:
            pd.DataFrame: A weekly-resolution DataFrame including:
                - 'pet', 'precip', 'vwc', 'season', 'year'
                - '{column} {weeks} rolling': rolling sum over the selected weeks
        """
        df = self.url_to_dataframe()
        df = self._parse_and_clean(df)
        df = self._assign_season(df)
        df = self._resample_weekly(df)
        df = self._apply_rolling(df,column, weeks)
        return df



    def individual_dataframes(self, season, column, weeks):
        """
        Filters the full weekly DataFrame to return data for a specific season only.

        Args:
            season (str): The season to filter by. Should be one of:
                            {'spring', 'summer', 'fall', 'winter'}.
            column (str): The name of the variable to apply the rolling sum to.
            weeks (int): The size of the rolling window (in weeks).

        Returns:
            pd.DataFrame: A DataFrame containing only rows for the specified season,
                            including the original variables and the rolling sum column.

        Notes:
            - This method builds the full weekly DataFrame and then filters it by season.
            - Returns a `.copy()` to avoid unintended side effects from modifying a view.
        """
        df = self.weekly_dataframe(column, weeks)
        return df[df['season'] == season].copy()


    def _linear_regression(self, season_df, season, column, show_plot =True):
        """
        Performs a linear regression of the specified column over the days within a season,
        using a standardized calendar year (2000) to align dates across multiple years.

        Args:
            season_df (pd.DataFrame): DataFrame indexed by dates containing the data for one season.
            column (str): Name of the column on which to perform the regression.
            show_plot (bool, optional): If True, plots the regression result.

        Returns:
            tuple: Regression parameters (slope, intercept, r_value, p_value, std_err) and
                fitted values (y_fit) aligned with the valid dates.

        Notes:
            - Dates are converted to a fixed reference year (2000) to compare seasonal trends across years.
            - Missing or non-numeric values in the specified column are ignored in the regression.
            - Only rows with valid x (days) and y (column values) are used.
        """
        season_df = season_df.copy()
        season_df = season_df.sort_index()
        season_df.index = pd.to_datetime(season_df.index)  # ensure index is datetime
        fake_dates = pd.to_datetime(['2000-' + dt.strftime('%m-%d') for dt in season_df.index])

        x = (fake_dates - fake_dates[0]).days.values 
        y = pd.to_numeric(season_df[column], errors='coerce')  
        valid = (~np.isnan(x)) & (~y.isna())
        x_vals = x[valid]
        y_vals = y[valid]
        index_vals = fake_dates[valid] # filters out invaild data
        
        slope, intercept, r_value, p_value, std_err = linregress(x_vals, y_vals)
        y_fit = slope * x_vals + intercept

        equation = f'y = {slope:.2f}x + {intercept:.2f}'
        r_square = f'R^2 = {r_value**2:.2f}'
        
        if show_plot: # if true dates will be in the 2000s
            plt.figure(figsize=(10, 5))
            plt.plot(index_vals, y_vals, '.', label='Observed')
            plt.plot(index_vals, y_fit, '-', color='red', label='Fitted Line')
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))  
            plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval = 2)) 
            plt.title(f'PET {season}')
            # plt.title(f'Linear Regression for {column} ({season}) {equation}')
            plt.xlabel('Date')
            plt.ylabel('PET')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        return (slope, intercept), equation, r_square



    def residuals(self, season, column, weeks):
        """
        Calculates residuals of the rolling sum data relative to a linear trend within a specific season.

        Process:
            - Retrieves the weekly DataFrame filtered by season, with rolling sums applied.
            - Performs linear regression on the rolling sum column to model the seasonal trend.
            - Calculates residuals by subtracting predicted (fitted) values from actual rolling sums.
            - Returns the DataFrame with a new column containing residuals.

        Args:
            season (str): The season to analyze ('spring', 'summer', or 'fall').
                        Note: 'winter' is excluded here because there is no winter data.
            column (str): The climate variable column to analyze ('pet').
            weeks (int): The rolling window size in weeks.

        Returns:
            pd.DataFrame: DataFrame filtered to the specified season including the
                        rolling sum and a new residuals column named
                        '{column} {weeks} rolling Residuals'.

        Notes:
            - Seasons are mapped to fixed start dates in year 2000 for normalization:
                spring: March 1
                summer: June 1
                fall: September 1
            - Residuals indicate deviations from the linear seasonal trend.
            - only used for PET
        """
        df = self.individual_dataframes(season, column, weeks).copy()
        column_name = f'{column} {weeks} rolling' #only applied to rolling columns even if rolling is 1
        
        (slope, intercept), equation, r2 = self._linear_regression( df, season, column_name, show_plot = False)
        fake_dates = pd.to_datetime(['2000-' + dt.strftime('%m-%d') for dt in df.index])
        season_start = {'spring': '2000-03-01','summer': '2000-06-01','fall':   '2000-09-01'}[season]
        day_offsets = (fake_dates - pd.Timestamp(season_start)).days.values # maps the actual dates to the fake 2000s data

        predicted = slope * day_offsets + intercept
        df[f'{column_name} Residuals'] = df[column_name] - predicted
        return df
    


    def weekly_plot(self, season, weeks, residuals=None):  
        """
        Generate a 3-panel histogram plot of VWC, PET, and precipitation for a given season and rolling window.
        
        Parameters:
        -----------
        season : str
            One of 'spring', 'summer', or 'fall'.
        weeks : int
            Rolling window size in weeks.
        residuals : list of str, optional
            If 'pet' is included, PET residuals are plotted instead of raw values.
        
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Matplotlib figure with the histograms.
        """
        if residuals is None:
            residuals = []
        fig = plt.figure(figsize=(12, 12))
        gs = gridspec.GridSpec(3, 3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])

        weekly_vwc = self.individual_dataframes(season, 'vwc', weeks)
        sns.histplot(weekly_vwc, x=f"vwc {weeks}", kde=True, ax=ax1, bins = 15)
        ax1.set_xlabel("VWC")

        # PET
        if 'pet' in residuals:
            weekly = self.residuals(season, f'pet {weeks} rolling', weeks)
            sns.histplot(weekly, x=f"pet {weeks} Residuals", kde=True, ax=ax2)
            ax2.set_xlabel(f"PET (mm) {weeks} Residuals")
        else:
            weekly_pet = self.individual_dataframes(season, 'pet', weeks)
            sns.histplot(weekly, x=f"pet {weeks} ", kde=True, ax=ax2)
            ax2.set_xlabel("PET (mm)")

        weekly_precip = self.individual_dataframes(season, 'precip', weeks)
        sns.histplot(weekly_precip, x="precip", kde=True, ax=ax3)
        ax3.set_xlabel("Precip Weekly (mm)")

        fig.text(0.5, 1, season , ha='center', fontsize=16, weight='bold')
        plt.tight_layout()
        return fig

    
    def kde_histogram(self, weeks, residuals=None):
        """
        Generate KDE histograms of VWC, PET/PET Residuals, and Precip for each season.
        
        Parameters:
        -----------
        weeks : int
            Rolling window size in weeks.
        residuals : list of str, optional
            If 'pet' is in the list, use PET residuals instead of raw values.
        
        Returns:
        --------
        Displays a 3x3 grid of histograms (one row per season).
        """

        if residuals is None:
            residuals = []
        seasons = ['spring', 'summer', 'fall' ]
        season_colors = {'spring':'b', 'summer':'darkorange', 'fall':'m' }
        fig = plt.figure(figsize=(12, 12))
        gs = gridspec.GridSpec(3, 3)

        for col, season in enumerate(seasons):
            color = season_colors[season]

            weekly_vwc =  self.residuals(season, 'vwc', weeks) if 'vwc' in residuals else self.individual_dataframes(season, 'vwc', weeks)
            weekly_pet = self.residuals(season, 'pet', weeks) if 'pet' in residuals else self.individual_dataframes(season, 'pet', weeks)
            weekly_precip = self.individual_dataframes(season, 'precip', weeks)

            # VWC raw
            ax1 = fig.add_subplot(gs[col, 0])
            if f'vwc' in residuals:
                sns.histplot(weekly_vwc, x= f"vwc {weeks} rolling Residuals", kde=True, ax=ax1, color=color)
                ax1.set_xlabel("VWC Residuals")
            else:
                sns.histplot(weekly_vwc, x=f"vwc {weeks} rolling", kde=True, ax=ax1, color=color)
                ax1.set_xlabel("VWC")
            ax1.set_title(season)

            # PET raw or residuals
            ax2 = fig.add_subplot(gs[col, 1])
            if f'pet' in residuals:
                sns.histplot(weekly_pet, x= f"pet {weeks} rolling Residuals", kde=True, ax=ax2, color=color)
                ax2.set_xlabel("PET (mm) Residuals")
            else:
                sns.histplot(weekly_pet, x=f"pet {weeks} rolling", kde=True, ax=ax2, color=color)
                ax2.set_xlabel("PET (mm)")
            ax2.set_title(season)

            # Precip raw
            ax3 = fig.add_subplot(gs[col, 2])
            sns.histplot(weekly_precip, x=f"precip {weeks} rolling", kde=True, ax=ax3, color=color)
            ax3.set_xlabel("Precip Weekly (mm)")
        plt.tight_layout()
        return plt.show(fig)


class DroughtDetection:
    """
    A class for detecting flash drought conditions based on climate data such as 
    volumetric water content (VWC), precipitation, and potential evapotranspiration (PET), and PET residuals.

    Attributes
    ----------
    lat : float
        Latitude of the location being analyzed.
    lon : float
        Longitude of the location being analyzed.
    residual_column : str
        The name of the column ('pet') used for residual analysis.
    weeks : int
        The number of weeks used in the rolling window (2, 4).
    rolling_column : str
        The base climate variable for which a rolling average is calculated ('precip', 'pet', 'vwc').
    histogram : KDEhistogram
        KDE-based histogram generator initialized for the location and date range.
    weekly : pd.DataFrame
        Weekly climate data for the specified location, including calculated columns.

    Methods
    -------
    _build_weekly_dataframe(residual_column):
        Builds and returns a weekly dataframe containing rolling and residual values.
    """
    def __init__(self, lat, lon, rolling_column, weeks, residual_column):
        """
        Initializes the DroughtDetection object for a specific location and analysis setup.
        ONLY WORKS FOR LAT AND LON IN THE NORTHEAST UNITED STATE AND FROM 2002 - PRESENT

        Parameters
        ----------
        lat : float
            Latitude of the location to analyze.
        lon : float
            Longitude of the location to analyze.
        rolling_column : str
            The climate variable to apply a rolling window to ('pet').
        weeks : int
            The number of weeks to use in the rolling calculation. 
        residual_column : str
            Column name to be used for residual analysis ('pet').

        Notes:
        - The rolling_column needs an input and can use 1 for the weeks for no rolling (vwc).

        """
        self.lat = lat
        self.lon = lon
        self.residual_column = residual_column
        self.weeks = weeks
        self.rolling_column = rolling_column
        self.histogram = KDEhistogram('2002-03-01', '2024-10-31', lat, lon) # creates dates for total time period for calulations purposes.
        self.weekly = self._build_weekly_dataframe(residual_column=self.residual_column)
        
    def _build_weekly_dataframe(self, residual_column = None):
        """
        Constructs and returns a weekly dataframe for the location, optionally including residuals.

        This method pulls either:
        - Standard weekly-aggregated climate data (if no residual column is specified), or
        - Residuals computed for spring, summer, and fall seasons (if a residual column is provided).

        Winter data is always excluded because of lack of data.

        Parameters
        ----------
        residual_column : str, optional
            The name of the residual column to include in the dataframe.
            If None, uses the rolling average of the specified base column instead.

        Returns
        -------
        pd.DataFrame
            A cleaned and sorted weekly dataframe, with residuals and/or rolling values,
            and with winter weeks removed.
        """
       
        if residual_column is None:
            df = self.histogram.weekly_dataframe(self.rolling_column, self.weeks)
            month = df.index.month
            day = df.index.day

            remove_winter = (df['season'] == 'winter') # not complete season of winter data

            weekly = df[~remove_winter]
            weekly.index = pd.to_datetime(weekly.index, errors='coerce')
            weekly = weekly.sort_index()

        else:
            spring = self.histogram.residuals('spring', self.residual_column, self.weeks)
            summer = self.histogram.residuals('summer', self.residual_column, self.weeks)
            fall = self.histogram.residuals('fall', self.residual_column, self.weeks)

            spring.index = pd.to_datetime(spring.index, errors='coerce')
            summer.index = pd.to_datetime(summer.index, errors='coerce')
            fall.index = pd.to_datetime(fall.index, errors='coerce')

            weekly = pd.concat([spring, summer, fall])
            weekly = weekly.sort_index() # creates a dataframe with the full time period with a complete residual column

        return weekly
    
    def residual_dataframe(self):
        """
        Returns
        -------
        pd.DataFrame
            A DataFrame containing weekly climate data and residuals.
            example if pet is the rolling column and residuals 
            Columns include:
            - 'pet': Daily or raw PET values
            - 'pet {weeks} rolling': Rolling weekly PET sums or averages
            - 'pet {weeks} Residuals': Residuals from linear regression on seasonal PET 
            - 'vwc': Volumetric water content
            - 'vwc {weeks} rolling': Weekly rolling VWC
            - 'precip': Precipitation values
            - 'season': Season label (spring, summer, fall)
            - 'year': Year of the observation
        """
        return self.weekly

    def variable_plot(self, column):
        """
        Plots the selected weekly climate variable over time (2002-2024).

        Parameters
        ----------
        column : str
            The name of the column in the weekly DataFrame to plot.
            Options include:
            - 'pet'
            - 'pet {weeks} rolling'
            - 'pet {weeks} Residuals'
            - 'vwc'
            - 'vwc {weeks} rolling'
            - 'precip'
            - 'precip {weeks} rolling'

        Displays
        --------
        A time series line plot of the specified variable.
        """
        weekly = self.weekly
        plt.plot(weekly.index, weekly[column])
        plt.title(f"Weekly {column}")
        plt.xlabel("Date")
        plt.ylabel(f"{column}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def gamma(self,column):
        """
        Computes Standardized Precipitation Index (SPI) values for a given column using 
        a Gamma distribution (scipy), applied separately to each season.

        Parameters
        ----------
        column : str
            Name of the column in the weekly DataFrame to calculate SPI for.
            Examples:
            - 'precip {weeks} rolling'
            - 'pet {weeks} rolling'
            - 'pet {weeks} rolling Residuals'
            - 'vwc {weeks} rolling'

        Returns
        -------
        pd.DataFrame
            The weekly DataFrame with an additional column:
            - f"{column} gamma": SPI values computed using Gamma distribution.

        Notes
        -----
        - SPI is calculated seasonally (spring, summer, fall) to account for differing 
        distribution characteristics.
        - Missing values are ignored during computation.
        - Winter values are excluded by design.
        """
        df = self.weekly.copy()
        all_spi = pd.Series(index=df.index, dtype='float64')
        for season in ['spring', 'summer', 'fall']:
            season_df = df[df['season'] == season]
            season_series = season_df[column].dropna()

            shape, loc, scale = gamma.fit(season_series)
            cdf_vals = gamma.cdf(season_series, a=shape, loc=loc, scale=scale)
            spi_vals = norm.ppf(cdf_vals)
            spi_vals = np.clip(spi_vals, -2.5, 2.5)

            all_spi.loc[season_series.index] = spi_vals
        self.weekly[f'{column} gamma'] = all_spi
        return self.weekly
    
    def pearson(self, column):
        """
        Computes Standardized Precipitation Index (SPI) values for a given column using 
        a Pearson III distribution (scipy), applied separately to each season.

        Parameters
        ----------
        column : str
            Name of the column in the weekly DataFrame to calculate SPI for.
            Examples:
            - 'precip {weeks} rolling'
            - 'pet {weeks} rolling'
            - 'pet {weeks} rolling Residuals'
            - 'vwc {weeks} rolling'

        Returns
        -------
        pd.DataFrame
            The weekly DataFrame with an additional column:
            - f"{column} pearson": SPI values computed using Pearson III distribution.

        Notes
        -----
        - SPI is calculated seasonally (spring, summer, fall) to account for differing 
        distribution characteristics.
        - Missing values are ignored during computation.
        - Winter values are excluded by design.
        """
        df = self.weekly.copy()
        all_spi = pd.Series(index=df.index, dtype='float64')

        for season in ['spring', 'summer', 'fall']:
            season_df = df[df['season'] == season]
            season_series = season_df[column].dropna()

            skew, loc, scale = pearson3.fit(season_series)
            cdf_vals = pearson3.cdf(season_series, skew, loc=loc, scale=scale)
            spi_vals = norm.ppf(cdf_vals)
            spi_vals = np.clip(spi_vals, -2.5, 2.5)
            all_spi.loc[season_series.index] = spi_vals

        self.weekly[f'{column} pearson'] = all_spi
        return self.weekly

    def normal(self, column):
        """
        Computes Standardized Precipitation Index (SPI) values for a given column using 
        a Normal distribution (scipy), applied separately to each season.

        Parameters
        ----------
        column : str
            Name of the column in the weekly DataFrame to calculate SPI for.
            Examples:
            - 'precip {weeks} rolling'
            - 'pet {weeks} rolling'
            - 'pet {weeks} rolling Residuals'
            - 'vwc {weeks} rolling'

        Returns
        -------
        pd.DataFrame
            The weekly DataFrame with an additional column:
            - f"{column} normal": SPI values computed using Normal distribution.

        Notes
        -----
        - SPI is calculated seasonally (spring, summer, fall) to account for differing 
        distribution characteristics.
        - Missing values are ignored during computation.
        - Winter values are excluded by design.
        """
        df = self.weekly.copy()
        all_spi = pd.Series(index=df.index, dtype='float64')
        for season in ['spring', 'summer', 'fall']:
            season_df = df[df['season'] == season]
            season_series = season_df[column].dropna()
          
            mu, sigma = norm.fit(season_series)
            cdf_vals = norm.cdf(season_series, loc=mu, scale=sigma)
            spi_vals = norm.ppf(cdf_vals)
            spi_vals = np.clip(spi_vals, -2.5, 2.5)

            all_spi.loc[season_series.index] = spi_vals
        self.weekly[f'{column} normal'] = all_spi
        return self.weekly
    
    def _categories_func(self, val):
        """
        Categorize a z-score into drought/wetness severity classes.

        Categories (returned as integers):
            4 → Extreme Drought (≤ -2.05)
            3 → Severe Drought (-2.05 to -1.645)
            2 → Drought (-1.645 to -1.282)
            1 → Mild Drought (-1.282 to -.84)
            0 → Abnormally dry(-.84 to -.5)
            -1 → Normal (> -.5)

        Parameters:
            val (float or None): z-score.

        Returns:
            int or None: Category code (-1-4), or None if val is None or NaN.
        """

        if val is None or pd.isna(val):
            return None
        if val <= -2.05:
            return 4 #'Extreme Drought'
        elif -2.05 < val <= -1.645:
            return 3 #'Severe Drought'
        elif -1.645 < val <= -1.282:
            return 2 #'Drought'
        elif -1.282 < val <= -.84:
            return 1  # 'Mild Drought'
        elif -.84 < val <= -.5:
            return 0 #'Abnormal Dry'
        elif val > -.5:
            return  -1 # Normal

    def categories(self, column, functions):
        """
        Assigns drought/wetness categories to a given column for one or more distributions.

        This method:
        - Computes z-score using the specified distribution(s)
        - Maps each z-score to a category (-1 to 4) using `_categories_func`
        - Adds new columns to the DataFrame for each distribution method's category

        Parameters
        ----------
        column : str
            The name of the data column to compute SPI values for.
            Example: 'pet 4 rolling Residuals', 'vwc 1 rolling' etc.

        functions : list of str
            List of SPI calculation methods to apply. Can include:
            - 'normal'
            - 'pearson'
            - 'gamma'

        Returns
        -------
        pd.DataFrame
            A copy of the weekly DataFrame with added columns:
            - "{column} {func}" ('precip 2 rolling gamma'): z-score
            - "{column} {func} category" ( 'precip 2 rolling gamma category'): Integer category labels
        """
        if 'normal' in functions:
            self.normal(column)
        if 'pearson' in functions:
            self.pearson(column)
        if 'gamma' in functions:
            self.gamma(column)
        weekly = self.weekly.copy()

        for function in functions: #applies functions and creates a new column of categories
            col_name = f'{column} {function}'
            weekly[f"{col_name} category"] = weekly[col_name].apply(self._categories_func)

        self.weekly = weekly
        return weekly
    
    # UPDATE THIS
    # def residual_thresholds(self, column, weeks):
    #     """
    #     Computes threshold residual values corresponding to standard SPI category bounds
    #     using fitted statistical distributions (normal, gamma, pearson).

    #     This helps interpret SPI values in terms of actual residual units (e.g., mm of PET).

    #     Parameters
    #     ----------
    #     column : str
    #         The name of the column in `self.weekly` containing the residuals to analyze.

    #     weeks : int
    #         The rolling window size (e.g., 2, 4 weeks), used to label output columns.

    #     Returns
    #     -------
    #     pd.DataFrame
    #         DataFrame with:
    #         - 'Z-Score': standard SPI thresholds
    #         - 'Category': drought/wetness category names
    #         - Residual thresholds for each distribution (in same units as `column`)
    #         Columns are named like 'normal 4 rolling Residual'.
    #     """
    #     spi_bounds = [-2.0, -1.5, -1.0, 1.0, 1.5, 2.0] 
    #     categories = ["Extreme Drought", "Severe Drought", "Mild Drought", "Normal", "Moderately Wet", "Severely Wet"]

    #     results = {'Z-Score': spi_bounds, "Category": categories}

    #     for dist in ['normal', 'gamma', 'pearson']:
    #         series = self.weekly[column].dropna()

    #         if dist == 'normal':
    #             params = stats.norm.fit(series)
    #             dist_obj = stats.norm(*params)
    #         elif dist == 'gamma':
    #             params = stats.gamma.fit(series)
    #             dist_obj = stats.gamma(*params)
    #         elif dist == 'pearson':
    #             params = stats.pearson3.fit(series)
    #             dist_obj = stats.pearson3(*params)

    #         residuals = [dist_obj.ppf(stats.norm.cdf(spi)) for spi in spi_bounds]
    #         results[f'{dist} {weeks} rolling Residual'] = residuals

    #     df = pd.DataFrame(results)
    #     return df

    def _onset_two_weeks(self, column, functions):
        """
        Detects flash drought onset events based on a 2-category increase over two weeks.

        For each function provided (e.g., 'gamma', 'normal', 'pearson'), this method:
        - Calculates z-score and classifies values into categorical drought levels.
        - Scans for instances where the category increases (gets dryer) by 2 or more over two weeks,
        marking that as a potential drought onset.
        - Tracks the peak drought value and category following the onset.
        - Identifies the end of the drought as the point where category stabilizes or reverses trend.
            - excludes values where the end point is in another year
        - Returns a DataFrame summarizing all detected events.

        Parameters
        ----------
        column : str
            The name of the variable column used for z-score calculation (e.g., 'pet rolling 2 residuals').
        functions : list of str
            Distribution types to apply for transformation. Supported: 'normal', 'gamma', 'pearson'.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing detected flash drought onset events with the following columns:
            - onset_date : Timestamp
            - end_date : Timestamp
            - onset {column} category : int
            - end {column} category : int
            - onset {column} : float
            - end {column} : float
            - function : str
        """
        weekly = self.categories(column, functions)
        all_onsets = []
        # Loops over each function
        for function in functions: 
            in_onset = False # tracks if the onset is already in the list (gets rid of multiple flash droughts in same jump)
            start_index = None
            onset_record = {}
            
            peak_index = None # tracks the peak for detecting end pt
            peak_category = None 
            peak_value = None

            # detects a category change of an increase in 2
            for i in range(2, len(weekly)):
                cat_now = weekly.iloc[i][f'{column} {function} category'] # current cat
                cat_past = weekly.iloc[i - 2][f'{column} {function} category'] # cat from 2 weeks ago

                if pd.notna(cat_now) and pd.notna(cat_past):
                    change = cat_now - cat_past # calculates difference

                    if not in_onset and change >= 2 and cat_past >= 0:  # if change is +2 or greater and in onset
                        in_onset = True 
                        start_index = i-2
                        onset_record = { "onset_date": weekly.index[start_index], "onset_category": cat_past, "onset_value": weekly.iloc[start_index][f'{column} {function}']}
                        
                        peak_index = i
                        peak_category = cat_now
                        peak_value = weekly.iloc[i][f'{column} {function}']
                        
                    # tracks the peak so that it doesnt repeat and to find end pt    
                    elif in_onset:
                        spi_now = weekly.iloc[i][f'{column} {function}']
    
                        if peak_value is None or spi_now < peak_value:
                            peak_value = spi_now
                            peak_index = i
                            peak_category = weekly.iloc[i][f'{column} {function} category']

                        elif spi_now >= peak_value: # finds end pt
                            end_date = weekly.index[peak_index]
                            onset_year = onset_record["onset_date"].year
                            end_year = end_date.year

                            if onset_year == end_year: # makes sure its not in different years
                                onset_record.update({
                                    "end_date": end_date,
                                    "end_category": peak_category,
                                    "end_value": peak_value,
                                    "function": function
                                })
                                all_onsets.append((
                                    onset_record["onset_date"],
                                    onset_record["end_date"],
                                    onset_record["onset_category"],
                                    onset_record["end_category"],
                                    onset_record["onset_value"],
                                    onset_record["end_value"],
                                    onset_record["function"]
                                ))
                            in_onset = False
                            onset_record = {}
                            peak_value = None
                            peak_index = None
                            peak_category = None

        #records onsets
        if in_onset and peak_index is not None:
            onset_record.update({
                "end_date": weekly.index[peak_index],
                "end_category": peak_category,
                "end_value": peak_value,
                "function": function
            })
            all_onsets.append((
                onset_record["onset_date"],
                onset_record["end_date"],
                onset_record["onset_category"],
                onset_record["end_category"],
                onset_record["onset_value"],
                onset_record["end_value"],
                onset_record["function"]
            ))
        # onset_weeks_df['onset_date'] = pd.to_datetime(onset_weeks_df['onset_date'], errors='coerce')
        onset_weeks_df = pd.DataFrame(all_onsets, columns=['onset_date', 'end_date',f'onset {column} category',f'end {column} category', f'onset {column}', f'end {column}', 'function'])
        return onset_weeks_df

    def two_week_onset(self, column, functions):
        """
        Filters for two-week flash drought onsets where the starting drought category is not normal.

        Parameters
        ----------
        column : str
            The variable name to analyze for SPI-based drought detection (e.g., 'pet 2 rolling Residuals').
        functions : list of str
            List of distribution methods to use, such as ['gamma', 'normal', 'pearson'].

        Returns
        -------
        pd.DataFrame
            Filtered DataFrame of flash drought onset events where the end category > 4.
        """
        df = self._onset_two_weeks(column, functions)
        return df
            

    def _onset_four_weeks(self, column, functions):
        """
        Detects flash drought onset events based on a 2-category increase over four weeks.

        For each function provided (e.g., 'gamma', 'normal', 'pearson'), this method:
        - Calculates z-score and classifies values into categorical drought levels.
        - Scans for instances where the category increases (gets dryer) by 2 or more over four weeks,
        marking that as a potential drought onset.
        - Tracks the peak drought value and category following the onset.
        - Identifies the end of the drought as the point where category stabilizes or reverses trend.
            - excludes values where the end point is in another year
        - Returns a DataFrame summarizing all detected events.

        Parameters
        ----------
        column : str
            The name of the variable column used for z-score calculation (e.g., 'pet rolling 2 residuals').
        functions : list of str
            Distribution types to apply for transformation. Supported: 'normal', 'gamma', 'pearson'.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing detected flash drought onset events with the following columns:
            - onset_date : Timestamp
            - end_date : Timestamp
            - onset {column} category : int
            - end {column} category : int
            - onset {column} : float
            - end {column} : float
            - function : str
        """
        weekly = self.categories(column, functions)
        all_onsets = []
        for function in functions:
            in_onset = False
            start_index = None
            onset_record = {}
            
            peak_index = None
            peak_category = None
            peak_value = None

            # Tracks for a four week time period
            for i in range(4, len(weekly)):
                cat_now = weekly.iloc[i][f'{column} {function} category']
                cat_past = weekly.iloc[i - 4][f'{column} {function} category']

                if pd.notna(cat_now) and pd.notna(cat_past):
                    change = cat_now - cat_past

                    if not in_onset and change >= 2 and cat_past >= 0:  
                        in_onset = True 
                        start_index = i-4
                        onset_record = { "onset_date": weekly.index[start_index], "onset_category": cat_past, "onset_value": weekly.iloc[start_index][f'{column} {function}']}
                        
                        peak_index = i
                        peak_category = cat_now
                        peak_value = weekly.iloc[i][f'{column} {function}']
                        
                        
                    elif in_onset:
                        spi_now = weekly.iloc[i][f'{column} {function}']
    
                        if peak_value is None or spi_now < peak_value:
                            peak_value = spi_now
                            peak_index = i
                            peak_category = weekly.iloc[i][f'{column} {function} category']

                        elif spi_now >= peak_value:
                            end_date = weekly.index[peak_index]
                            onset_year = onset_record["onset_date"].year
                            end_year = end_date.year

                            if onset_year == end_year: 
                                onset_record.update({
                                    "end_date": end_date,
                                    "end_category": peak_category,
                                    "end_value": peak_value,
                                    "function": function
                                })
                                all_onsets.append((
                                    onset_record["onset_date"],
                                    onset_record["end_date"],
                                    onset_record["onset_category"],
                                    onset_record["end_category"],
                                    onset_record["onset_value"],
                                    onset_record["end_value"],
                                    onset_record["function"]
                                ))
                            in_onset = False
                            onset_record = {}
                            peak_value = None
                            peak_index = None
                            peak_category = None

        if in_onset and peak_index is not None:
            onset_record.update({
                "end_date": weekly.index[peak_index],
                "end_category": peak_category,
                "end_value": peak_value,
                "function": function
            })
            all_onsets.append((
                onset_record["onset_date"],
                onset_record["end_date"],
                onset_record["onset_category"],
                onset_record["end_category"],
                onset_record["onset_value"],
                onset_record["end_value"],
                onset_record["function"]
            ))
        # onset_weeks_df['onset_date'] = pd.to_datetime(onset_weeks_df['onset_date'], errors='coerce')
        onset_weeks_df = pd.DataFrame(all_onsets, columns=['onset_date', 'end_date',f'onset {column} category',f'end {column} category', f'onset {column}', f'end {column}', 'function'])
        return onset_weeks_df
        
    def four_week_onset(self, column, functions):
        """
        Filters for four-week flash drought onsets where the onset start category is starts in or above 0.


        Parameters
        ----------
        column : str
            The variable name to analyze for SPI-based drought detection (e.g., 'pet 2 rolling Residuals').
        functions : list of str
            List of distribution methods to use, such as ['gamma', 'normal', 'pearson'].

        Returns
        -------
        pd.DataFrame
            Filtered DataFrame of flash drought onset events where the end category > 0.
        """
        df = self._onset_four_weeks(column, functions)
        return df

    def display_two_wk_onsets(self, column, functions):
        """
        Aggregates and displays flash drought onsets detected across multiple distributions.

        This method filters two-week onset events (where the end drought category > 4)
        for each distribution method in `functions`. It groups onsets by onset date and
        combines the function names where multiple distributions detect the same onset date.

        Parameters
        ----------
        column : str
            The variable name to analyze for SPI-based drought detection (e.g., 'pet 2 rolling Residuals').
        functions : list of str
            List of distribution methods to use, ['gamma', 'normal', 'pearson'].

        Returns
        -------
        pd.DataFrame
            A summary DataFrame indexed by onset_date, with columns:
            - 'function': comma-separated list of distributions detecting onset
            - other onset information (e.g., end date, categories, values) from the first detected entry
        """
        all_data = []
        for function in functions: # tracks all distributions
            func = self.two_week_onset(column, [function])
            if not func.empty:
                func['function'] = function
                func.set_index('onset_date', inplace=True, drop=True)
                all_data.append(func)
        if not all_data:
            return pd.DataFrame(columns=['onset_date', 'functions'])
        df = pd.concat(all_data).sort_index()
        
        # Combines onsets with multiple distributions
        functions_grouped = df.groupby(df.index)['function'].agg(lambda x: ', '.join(x.dropna().unique()))
        others_grouped = df.drop(columns='function').groupby(df.index).first()  

        grouped = pd.concat([functions_grouped, others_grouped], axis=1)
        grouped.index.name = 'onset_date' #index of dateTime
        return grouped.reset_index()
    
    def display_four_wk_onsets(self, column, functions):
        """
        Aggregates and displays flash drought onsets detected across multiple distributions.

        This method filters four-week onset events (where the end drought category > 4)
        for each distribution method in `functions`. It groups onsets by onset date and
        combines the function names where multiple distributions detect the same onset date.

        Parameters
        ----------
        column : str
            The variable name to analyze for SPI-based drought detection (e.g., 'pet 2 rolling Residuals').
        functions : list of str
            List of distribution methods to use, ['gamma', 'normal', 'pearson'].

        Returns
        -------
        pd.DataFrame
            A summary DataFrame indexed by onset_date, with columns:
            - 'function': comma-separated list of distributions detecting onset
            - other onset information (e.g., end date, categories, values) from the first detected entry
        """
        all_data = []
        for function in functions:
            func = self.four_week_onset(column, [function])
            if not func.empty:
                func['function'] = function
                func.set_index('onset_date', inplace=True, drop=True)
                all_data.append(func)
        if not all_data:
            return pd.DataFrame(columns=['onset_date', 'functions'])
        df = pd.concat(all_data).sort_index()
        
        functions_grouped = df.groupby(df.index)['function'].agg(lambda x: ', '.join(x.dropna().unique()))
        others_grouped = df.drop(columns='function').groupby(df.index).first()  

        grouped = pd.concat([functions_grouped, others_grouped], axis=1)
        grouped.index.name = 'onset_date'
        return grouped.reset_index()

    def _z_score_two_wk_onset(self, column, functions):
        """
        Detects flash drought onset events based on a 2 z-score decrease over two weeks.

        For each function provided (e.g., 'gamma', 'normal', 'pearson'), this method:
        - Calculates z-score
        - Scans for instances where the z-score decreases (gets dryer) by 2 or more over two weeks,
        marking that as a potential drought onset.
        - Tracks the peak drought value and category following the onset.
        - Identifies the end of the drought as the point where category stabilizes or reverses trend.
            - excludes values where the end point is in another year
        - Returns a DataFrame summarizing all detected events.

        Parameters
        ----------
        column : str
            The name of the variable column used for z-score calculation (e.g., 'pet rolling 2 residuals').
        functions : list of str
            Distribution types to apply for transformation. Supported: 'normal', 'gamma', 'pearson'.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing detected flash drought onset events with the following columns:
            - onset_date : Timestamp
            - end_date : Timestamp
            - onset {column} : float
            - end {column} : float
            - function : str
        """
        if 'normal' in functions:
            self.normal(column)
        if 'pearson' in functions:
            self.pearson(column)
        if 'gamma' in functions:
            self.gamma(column)
        weekly = self.weekly.copy()
        all_onsets = []
        for function in functions:
            in_onset = False
            start_index = None
            onset_record = {}
            
            peak_index = None
            peak_category = -1 
            peak_value = None
            col_name = f'{column} {function}'
            onsets_weeks = []
            # Tracks for 2 weeks z-score
            for i in range(2, len(weekly)):
                z_now = weekly.iloc[i][col_name]
                z_past = weekly.iloc[i - 2][col_name]

                if pd.notna(z_now) and pd.notna(z_past):
                    change = z_now - z_past
                
                    # decreases by 2
                    if not in_onset and change <= -2:  
                        in_onset = True 
                        start_index = i-2
                        onset_record = { "onset_date": weekly.index[start_index], "onset_value": weekly.iloc[start_index][col_name]}
                        
                        peak_index = i
                        peak_value = weekly.iloc[i][col_name]
                    elif in_onset:
                        spi_now = weekly.iloc[i][col_name]
    
                        if peak_value is None or spi_now < peak_value:
                            peak_value = spi_now
                            peak_index = i
                        elif spi_now >= peak_value:
                            end_date = weekly.index[peak_index]
                            onset_year = onset_record["onset_date"].year
                            end_year = end_date.year
                            if onset_year == end_year: 
                                onset_record.update({
                                    "end_date": end_date,
                                    "end_value": peak_value,
                                    "function": function
                                })
                                all_onsets.append((
                                    onset_record["onset_date"],
                                    onset_record["end_date"],
                                    onset_record["onset_value"],
                                    onset_record["end_value"],
                                    onset_record["function"]
                                ))
                            in_onset = False
                            onset_record = {}
                            peak_value = None
                            peak_index = None
            if in_onset and peak_index is not None:
                onset_record.update({
                    "end_date": weekly.index[peak_index],
                    "end_value": peak_value,
                    "function": function
                })
                all_onsets.append((
                    onset_record["onset_date"],
                    onset_record["end_date"],
                    onset_record["onset_value"],
                    onset_record["end_value"],
                    onset_record["function"]
                ))

                            

        onset_weeks_df = pd.DataFrame(all_onsets, columns=['onset_date', 'end_date', f'onset {column}', f'end {column}', 'function'])
                    
        return onset_weeks_df

    def z_score_two_wks(self, column, functions):
        """
        Filters for two-week flash drought onsets where the ending drought z-score is less than -1.

        This method calls `_z_score_two_wks` to detect onset events, then filters those events
        to retain only cases where the end z-score is less than -1

        Parameters
        ----------
        column : str
            The variable name to analyze for SPI-based drought detection (e.g., 'pet 2 rolling Residuals').
        functions : list of str
            List of distribution methods to use, such as ['gamma', 'normal', 'pearson'].

        Returns
        -------
        pd.DataFrame
            Filtered DataFrame of flash drought onset events where the end z-score < -1.
        """
        z_score = self._z_score_two_wk_onset(column, functions)
        return z_score[z_score[f'end {column}'] < -1]


    def _z_score_four_wk_onset(self, column, functions):
        """
        Detects flash drought onset events based on a 2 z-score decrease over four weeks.

        For each function provided (e.g., 'gamma', 'normal', 'pearson'), this method:
        - Calculates z-score
        - Scans for instances where the z-score decreases (gets dryer) by 2 or more over four weeks,
        marking that as a potential drought onset.
        - Tracks the peak drought value and category following the onset.
        - Identifies the end of the drought as the point where category stabilizes or reverses trend.
            - excludes values where the end point is in another year
        - Returns a DataFrame summarizing all detected events.

        Parameters
        ----------
        column : str
            The name of the variable column used for z-score calculation (e.g., 'pet rolling 2 residuals').
        functions : list of str
            Distribution types to apply for transformation. Supported: 'normal', 'gamma', 'pearson'.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing detected flash drought onset events with the following columns:
            - onset_date : Timestamp
            - end_date : Timestamp
            - onset {column} : float
            - end {column} : float
            - function : str
        """
        if 'normal' in functions:
            self.normal(column)
        if 'pearson' in functions:
            self.pearson(column)
        if 'gamma' in functions:
            self.gamma(column)
        weekly = self.weekly.copy()
        all_onsets = []
        for function in functions:
            in_onset = False
            start_index = None
            onset_record = {}
            
            peak_index = None
            peak_category = -1 
            peak_value = None
            col_name = f'{column} {function}'
            onsets_weeks = []
            for i in range(4, len(weekly)):
                z_now = weekly.iloc[i][col_name]
                z_past = weekly.iloc[i - 4][col_name]

                if pd.notna(z_now) and pd.notna(z_past):
                    change = z_now - z_past
                
                    if not in_onset and change <= -2:  
                        in_onset = True 
                        start_index = i-4
                        onset_record = { "onset_date": weekly.index[start_index], "onset_value": weekly.iloc[start_index][col_name]}
                        
                        peak_index = i
                        peak_value = weekly.iloc[i][col_name]
                    elif in_onset:
                        spi_now = weekly.iloc[i][col_name]
    
                        if peak_value is None or spi_now < peak_value:
                            peak_value = spi_now
                            peak_index = i
                        elif spi_now >= peak_value:
                            end_date = weekly.index[peak_index]
                            onset_year = onset_record["onset_date"].year
                            end_year = end_date.year
                            if onset_year == end_year: 
                                onset_record.update({
                                    "end_date": end_date,
                                    "end_value": peak_value,
                                    "function": function
                                })
                                all_onsets.append((
                                    onset_record["onset_date"],
                                    onset_record["end_date"],
                                    onset_record["onset_value"],
                                    onset_record["end_value"],
                                    onset_record["function"]
                                ))
                            in_onset = False
                            onset_record = {}
                            peak_value = None
                            peak_index = None
            if in_onset and peak_index is not None:
                onset_record.update({
                    "end_date": weekly.index[peak_index],
                    "end_value": peak_value,
                    "function": function
                })
                all_onsets.append((
                    onset_record["onset_date"],
                    onset_record["end_date"],
                    onset_record["onset_value"],
                    onset_record["end_value"],
                    onset_record["function"]
                ))

                            

        onset_weeks_df = pd.DataFrame(all_onsets, columns=['onset_date', 'end_date', f'onset {column}', f'end {column}', 'function'])
                    
        return onset_weeks_df

    def z_score_four_wks(self, column, functions):
        """
        Filters for four-week flash drought onsets where the ending drought z-score is less than -1.

        This method calls `_z_score_four_wks` to detect onset events, then filters those events
        to retain only cases where the end z-score is less than -1

        Parameters
        ----------
        column : str
            The variable name to analyze for SPI-based drought detection (e.g., 'pet 2 rolling Residuals').
        functions : list of str
            List of distribution methods to use, such as ['gamma', 'normal', 'pearson'].

        Returns
        -------
        pd.DataFrame
            Filtered DataFrame of flash drought onset events where the end z-score < -1.
        """
        z_score = self._z_score_four_wk_onset(column, functions)
        return z_score[z_score[f'end {column}'] < -1]

    
    def plot_gradient_line_with_fill(self, ax, dates, values, cmap='GnBu_r', vmin=-3, vmax=3):
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap(cmap)
        colors = cmap(norm(values))

        ax.plot(dates, values, color='black', linewidth=1.5)


        for i in range(len(values)-1):
            xs = [dates[i], dates[i+1]]
            ys1 = [values[i], values[i+1]]
            ys2 = [ax.get_ylim()[0], ax.get_ylim()[0]]  

            ax.fill_between(xs, ys1, ys2,
                            color=colors[i],
                            linewidth=0,
                            alpha=0.6)
        return norm, cmap
    
    
    
    def two_week_spei_plot(self, column, functions, syear, eyear, z_score = False):
        """
        Plots weekly z-score values for different statistical distributions and highlights
        flash drought 2 week onset points over a specified year range.

        For each SPI function (gamma, normal, pearson), this method:
        - Calculates z-scorw using the specified distribution.
        - Optionally identifies flash drought onset points based on a z-score threshold or category jump.
        - Plots the time series with color mapping and marks detected onsets with a blue star.

        Parameters
        ----------
        column : str
            The name of the column containing the variable of interest (e.g., 'pet residuals').
        functions : list of str
            List of distribution names to compute z-score. Options include: ['normal', 'gamma', 'pearson'].
        syear : int
            Start year of the plot window (inclusive).
        eyear : int
            End year of the plot window (inclusive).
        z_score : bool, optional
            If True, detect flash drought onsets using z-score thresholding.
            If False, detect onsets based on categorical SPI jumps (default is False).

        Returns
        -------
        None
            Displays a matplotlib figure with SPI time series for each function, with onset points marked.
        """
        if 'normal' in functions:
            self.normal(column)
        if 'pearson' in functions:
            self.pearson(column)
        if 'gamma' in functions:
            self.gamma(column)
        weekly = self.weekly.dropna()
        weekly.index = pd.to_datetime(weekly.index)

        n = len(functions)
        f, ax = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True)
        if n == 1:
            ax = [ax]

        # Category boundary and label setup
        category_bounds = [-3, -2.05, -1.645, -1.282, -0.84, -0.5, 0]
        # category_ticks = [(category_bounds[i] + category_bounds[i+1]) / 2 for i in range(len(category_bounds)-1)]
        category_labels = [
            'Extreme Drought',
            'Severe Drought',
            'Drought',
            'Mild Drought',
            'Abnormal Dry',
            'Normal',
        ]
        for i, function in enumerate(functions):
            # uses either z-score or category
            if z_score:
                onset = self.z_score_two_wks(column, [function])
            else:
                onset = self.two_week_onset(column, [function])
                
            onset = onset[onset['function'] == function]

            for year, group in self.weekly.groupby('year'):
                dates = group.index
                values = group[f"{column} {function}"].values
                # norm, cmap = self.plot_gradient_line_with_fill(ax[i], dates, values, cmap='GnBu_r', vmin=-3, vmax=3)
                ax[i].plot(dates, values, color='black', linewidth=1)
                ax[i].fill_between(dates, values, color='lightblue', alpha=0.5)
            ax[i].scatter(onset['onset_date'], onset[f'onset {column}'], color = 'blue', marker = '*', s = 150)
            ax[i].invert_yaxis()
            ax[i].set_ylabel(function, fontsize=30)
            ax[i].axhline(y=-0.5, color='m', linestyle='--', linewidth=2)
            # ax[i].axhline(y=-1.282, color='black', linestyle='--', linewidth=1)
            # ax[i].grid()

            # Create secondary y-axis for category labels
            ax2 = ax[i].twinx()
            ax2.set_yticks(category_bounds)
            ax2.set_yticklabels(['' for _ in category_bounds])
            # label_x = pd.to_datetime(str(eyear)) + pd.Timedelta(days =25)
            # for j in range(len(category_labels)):
            #     mid = (category_bounds[j] + category_bounds[j+1]) / 2
            #     if i == n - 1:
            #         ax2.text(
            #             x=label_x, 
            #             y=mid,
            #             s=category_labels[j],
            #             va='center',
            #             ha='left',
            #             fontsize=11
            #         )
            for bound in category_bounds:
                ax2.axhline(bound, color='gray', linestyle='--', linewidth=0.4)
            ymin, ymax = ax[i].get_ylim()
            ax2.set_ylim(ymin, ymax)
            # ax2.invert_yaxis()

        ax[0].set_xlim(pd.to_datetime(str(syear)), pd.to_datetime(str(eyear)))
        
        if z_score:
            plt.suptitle(f' Z-SCORE SPEI for each Distribution for {column.capitalize()} {syear} - {eyear} rate = -2 z_score/2 weeks')
        else:
            # plt.suptitle(f'SPEI for {column.capitalize()} {syear} - {eyear} rate = 2 categories/2 weeks')
            plt.title("Precipitation Onset Dates for 2 category per 2 weeks", fontsize = 20)
        plt.show()

    def four_week_spei_plot(self, column, functions, syear, eyear, z_score = False):
        """
        Plots weekly z-score values for different statistical distributions and highlights
        flash drought 4 week onset points over a specified year range.

        For each SPI function (gamma, normal, pearson), this method:
        - Calculates z-scorw using the specified distribution.
        - Optionally identifies flash drought onset points based on a z-score threshold or category jump.
        - Plots the time series with color mapping and marks detected onsets with a blue star.

        Parameters
        ----------
        column : str
            The name of the column containing the variable of interest (e.g., 'pet residuals').
        functions : list of str
            List of distribution names to compute z-score. Options include: ['normal', 'gamma', 'pearson'].
        syear : int
            Start year of the plot window (inclusive).
        eyear : int
            End year of the plot window (inclusive).
        z_score : bool, optional
            If True, detect flash drought onsets using z-score thresholding.
            If False, detect onsets based on categorical SPI jumps (default is False).

        Returns
        -------
        None
            Displays a matplotlib figure with SPI time series for each function, with onset points marked.
        """
        if 'normal' in functions:
            self.normal(column)
        if 'pearson' in functions:
            self.pearson(column)
        if 'gamma' in functions:
            self.gamma(column)
        weekly = self.weekly.dropna()
        weekly.index = pd.to_datetime(weekly.index)

        n = len(functions)
        f, ax = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True)
        if n == 1:
            ax = [ax]

        # Category boundary and label setup
        category_bounds = [-3, -2.05, -1.645, -1.282, -0.84, -0.5, 0]
        # category_ticks = [(category_bounds[i] + category_bounds[i+1]) / 2 for i in range(len(category_bounds)-1)]
        category_labels = [
            'Extreme Drought',
            'Severe Drought',
            'Drought',
            'Mild Drought',
            'Abnormal Dry',
            'Normal',
        ]
        for i, function in enumerate(functions):
            # uses either z-score or category
            if z_score:
                onset = self.z_score_four_wks(column, [function])
            else:
                onset = self.four_week_onset(column, [function])
                
            onset = onset[onset['function'] == function]

            for year, group in self.weekly.groupby('year'):
                dates = group.index
                values = group[f"{column} {function}"].values
                # norm, cmap = self.plot_gradient_line_with_fill(ax[i], dates, values, cmap='GnBu_r', vmin=-3, vmax=3)
                ax[i].plot(dates, values, color='black', linewidth=0.5)
                ax[i].fill_between(dates, values, color='lightblue', alpha=0.5)
            ax[i].scatter(onset['onset_date'], onset[f'onset {column}'], color = 'blue', marker = '*')
            ax[i].invert_yaxis()
            ax[i].set_ylabel(function, fontsize=14)
            ax[i].axhline(y=-0.5, color='m', linestyle='--', linewidth=1)
            # ax[i].axhline(y=-1.282, color='black', linestyle='--', linewidth=1)
            ax[i].grid()

            ax2 = ax[i].twinx()
            ax2.set_yticks(category_bounds)
            ax2.set_yticklabels(['' for _ in category_bounds])
            label_x = pd.to_datetime(str(eyear)) + pd.Timedelta(days =20)
            for j in range(len(category_labels)):
                mid = (category_bounds[j] + category_bounds[j+1]) / 2
                if i == n - 1:
                    ax2.text(
                        x=label_x, 
                        y=mid,
                        s=category_labels[j],
                        va='center',
                        ha='left',
                        fontsize=9
                    )
            for bound in category_bounds:
                ax2.axhline(bound, color='gray', linestyle='--', linewidth=0.4)
            ymin, ymax = ax[i].get_ylim()
            ax2.set_ylim(ymin, ymax)
            # ax2.invert_yaxis()

        ax[0].set_xlim(pd.to_datetime(str(syear)), pd.to_datetime(str(eyear)))
        
        if z_score:
            plt.suptitle(f' Z-SCORE SPEI for each Distribution for {column.capitalize()} {syear} - {eyear} rate = -2 z_score/4 weeks')
        else:
            plt.suptitle(f'SPEI for {column.capitalize()} {syear} - {eyear} rate = 2 categories/4 weeks')
        plt.show()
    
   
class FlashDrought:
    def __init__(self, locations):
        """
        Initialize FlashDrought with a list of locations for drought analysis.

        Parameters
        ----------
        locations : list of dict
            Each dict must contain keys:
            - 'state' : str, the state name
            - 'county' : str, the county name
            - 'lat' : float, latitude coordinate
            - 'lon' : float, longitude coordinate

        Example
        -------
        locations = [
            {'state': 'NY', 'county': 'Albany', 'lat': 42.6526, 'lon': -73.7562},
            {'state': 'NY', 'county': 'Schenectady', 'lat': 42.8142, 'lon': -73.9396},
            {'state': 'NY', 'county': 'Rensselaer', 'lat': 42.7284, 'lon': -73.6918}
        ]
        fd = FlashDrought(locations)
        """
        self.locations = locations
            
    def combined_df(self):
        """
        Generate a summary DataFrame of flash drought statistics for each location.

        This method runs flash drought detection across different variables (PET, VWC, Precip)
        using various configurations:
            - Two thresholds: categorical change (2 category increase) and z-score threshold (-2)
            - Two timescales: 2-week and 4-week intervals
            - Two detection types: 2-week and 4-week onset detection

        For each location, the method records:
            - Number of detected droughts by each method
            - Most common onset month for each detection configuration

        Returns
        -------
        pd.DataFrame
            A DataFrame where each row corresponds to a location and contains:
                - County, latitude, longitude
                - Counts of flash droughts under each condition
                - The most frequent onset month(s) for each condition
        """
        results = [] 
        for loc in self.locations:
            state = loc['state']
            county = loc['county']
            lat = loc['lat']
            lon = loc['lon']
            # pet2 = DroughtDetection(lat = lat, lon = lon, rolling_column = 'pet', residual_column='pet', weeks = 2)
            pet4 = DroughtDetection(lat = lat, lon = lon, rolling_column = 'pet', residual_column='pet', weeks = 4)
            vwc = DroughtDetection(lat = lat, lon = lon , rolling_column = 'vwc', residual_column=None, weeks = 1)
            # precip2 = DroughtDetection(lat = lat, lon = lon , rolling_column = 'precip', residual_column=None, weeks = 2)
            precip4 = DroughtDetection(lat= lat, lon= lon, rolling_column = 'precip', residual_column=None, weeks = 4)

            row = {
                'State': state,
                'County': county,
                'lat': lat,
                'lon': lon,
                # '# Droughts PET (2 CAT/2wk)(2 wk intervals)': len(pet2.two_week_onset('pet 2 rolling Residuals', ['pearson'])),
                '# Droughts PET (2 CAT/2wk)(4 wk intervals)': len(pet4.two_week_onset('pet 4 rolling Residuals', ['pearson'])),
                # '# Droughts PET (2 CAT/4wk)(2 wk intervals)': len(pet2.four_week_onset('pet 2 rolling Residuals', ['pearson'])),
                '# Droughts PET (2 CAT/4wk)(4 wk intervals)': len(pet4.two_week_onset('pet 4 rolling Residuals', ['pearson'])),
                # '# Droughts PET (-2 Z-SCORE/2wk)(2 wk intervals)': len(pet2.z_score_two_wks('pet 2 rolling Residuals', ['pearson'])),
                # '# Droughts PET (-2 Z-SCORE/2wk)(4 wk intervals)': len(pet4.z_score_two_wks('pet 4 rolling Residuals', ['pearson'])),
                # '# Droughts PET (-2 Z-SCORE/4wk)(2 wk intervals)': len(pet2.z_score_four_wks('pet 2 rolling Residuals', ['pearson'])),
                # '# Droughts PET (-2 Z-SCORE/4wk)(4 wk intervals)': len(pet4.z_score_four_wks('pet 4 rolling Residuals', ['pearson'])),

                '# Droughts VWC (2 CAT/2wk)(Tuesdays)': len(vwc.two_week_onset('vwc 1 rolling', ['normal'])),
                '# Droughts VWC (2 CAT/4wk)(Tuesdays)': len(vwc.four_week_onset('vwc 1 rolling', ['normal'])),
                # '# Droughts VWC (-2 Z-SCORE/2wk)(Tuesdays)': len(vwc.z_score_two_wks('vwc 1 rolling', ['normal'])),
                # '# Droughts VWC (-2 Z-SCORE/4wk)(Tuesdays)': len(vwc.z_score_four_wks('vwc 1 rolling', ['normal'])),

                # '# Droughts Precip (2 CAT/2wk)(2 wk intervals)': len(precip2.two_week_onset('precip 2 rolling', ['pearson'])),
                '# Droughts Precip (2 CAT/2wk)(4 wk intervals)': len(precip4.two_week_onset('precip 4 rolling', ['pearson'])),
                # '# Droughts Precip (2 CAT/4wk)(2 wk intervals)': len(precip2.four_week_onset('precip 2 rolling', ['pearson'])),
                '# Droughts Precip (2 CAT/4wk)(4 wk intervals)': len(precip4.two_week_onset('precip 4 rolling', ['pearson'])),
                # '# Droughts Precip (-2 Z-SCORE/2wk)(2 wk intervals)': len(precip2.z_score_two_wks('precip 2 rolling', ['pearson'])),
                # '# Droughts Precip (-2 Z-SCORE/2wk)(4 wk intervals)': len(precip4.z_score_two_wks('precip 4 rolling', ['pearson'])),
                # '# Droughts Precip (-2 Z-SCORE/4wk)(2 wk intervals)': len(precip2.z_score_four_wks('precip 2 rolling', ['pearson'])),
                # '# Droughts Precip (-2 Z-SCORE/4wk)(4 wk intervals)': len(precip4.z_score_four_wks('precip 4 rolling', ['pearson']))


                # 'Highest Month PET (2 CAT/2wk)(2 wk intervals)': (pet2.two_week_onset('pet 2 rolling Residuals', ['pearson']))['onset_date'].dt.strftime('%B').value_counts().pipe(lambda x: x[x == x.max()]).index.tolist(),
                # 'Highest Month PET (2 CAT/2wk)(4 wk intervals)': (pet4.two_week_onset('pet 4 rolling Residuals', ['pearson']))['onset_date'].dt.strftime('%B').value_counts().pipe(lambda x: x[x == x.max()]).index.tolist(),
                # 'Highest Month PET (2 CAT/4wk)(2 wk intervals)': (pet2.four_week_onset('pet 2 rolling Residuals', ['pearson']))['onset_date'].dt.strftime('%B').value_counts().pipe(lambda x: x[x == x.max()]).index.tolist(),
                # 'Highest Month PET (2 CAT/4wk)(4 wk intervals)': (pet4.two_week_onset('pet 4 rolling Residuals', ['pearson']))['onset_date'].dt.strftime('%B').value_counts().pipe(lambda x: x[x == x.max()]).index.tolist(),
                # 'Highest Month PET (-2 Z-SCORE/2wk)(2 wk intervals)': (pet2.z_score_two_wks('pet 2 rolling Residuals', ['pearson']))['onset_date'].dt.strftime('%B').value_counts().pipe(lambda x: x[x == x.max()]).index.tolist(),
                # 'Highest Month PET (-2 Z-SCORE/2wk)(4 wk intervals)': (pet4.z_score_two_wks('pet 4 rolling Residuals', ['pearson']))['onset_date'].dt.strftime('%B').value_counts().pipe(lambda x: x[x == x.max()]).index.tolist(),
                # 'Highest Month PET (-2 Z-SCORE/4wk)(2 wk intervals)': (pet2.z_score_four_wks('pet 2 rolling Residuals', ['pearson']))['onset_date'].dt.strftime('%B').value_counts().pipe(lambda x: x[x == x.max()]).index.tolist(),
                # 'Highest Month PET (-2 Z-SCORE/4wk)(4 wk intervals)': (pet4.z_score_four_wks('pet 4 rolling Residuals', ['pearson']))['onset_date'].dt.strftime('%B').value_counts().pipe(lambda x: x[x == x.max()]).index.tolist(),

                # 'Highest Month VWC (2 CAT/2wk)(Tuesdays)': (vwc.two_week_onset('vwc 1 rolling', ['normal']))['onset_date'].dt.strftime('%B').value_counts().pipe(lambda x: x[x == x.max()]).index.tolist(),
                # 'Highest Month VWC (2 CAT/4wk)(Tuesdays)': (vwc.four_week_onset('vwc 1 rolling', ['normal']))['onset_date'].dt.strftime('%B').value_counts().pipe(lambda x: x[x == x.max()]).index.tolist(),
                # 'Highest Month VWC (-2 Z-SCORE/2wk)(Tuesdays)': (vwc.z_score_two_wks('vwc 1 rolling', ['normal']))['onset_date'].dt.strftime('%B').value_counts().pipe(lambda x: x[x == x.max()]).index.tolist(),
                # 'Highest Month VWC (-2 Z-SCORE/4wk)(Tuesdays)': (vwc.z_score_four_wks('vwc 1 rolling', ['normal']))['onset_date'].dt.strftime('%B').value_counts().pipe(lambda x: x[x == x.max()]).index.tolist(),

                # 'Highest Month Precip (2 CAT/2wk)(2 wk intervals)': (precip2.two_week_onset('precip 2 rolling', ['pearson']))['onset_date'].dt.strftime('%B').value_counts().pipe(lambda x: x[x == x.max()]).index.tolist(),
                # 'Highest Month Precip (2 CAT/2wk)(4 wk intervals)': (precip4.two_week_onset('precip 4 rolling', ['pearson']))['onset_date'].dt.strftime('%B').value_counts().pipe(lambda x: x[x == x.max()]).index.tolist(),
                # 'Highest Month Precip (2 CAT/4wk)(2 wk intervals)': (precip2.four_week_onset('precip 2 rolling', ['pearson']))['onset_date'].dt.strftime('%B').value_counts().pipe(lambda x: x[x == x.max()]).index.tolist(),
                # 'Highest Month Precip (2 CAT/4wk)(4 wk intervals)': (precip4.two_week_onset('precip 4 rolling', ['pearson']))['onset_date'].dt.strftime('%B').value_counts().pipe(lambda x: x[x == x.max()]).index.tolist(),
                # 'Highest Month Precip (-2 Z-SCORE/2wk)(2 wk intervals)': (precip2.z_score_two_wks('precip 2 rolling', ['pearson']))['onset_date'].dt.strftime('%B').value_counts().pipe(lambda x: x[x == x.max()]).index.tolist(),
                # 'Highest Month Precip (-2 Z-SCORE/2wk)(4 wk intervals)': (precip4.z_score_two_wks('precip 4 rolling', ['pearson']))['onset_date'].dt.strftime('%B').value_counts().pipe(lambda x: x[x == x.max()]).index.tolist(),
                # 'Highest Month Precip (-2 Z-SCORE/4wk)(2 wk intervals)': (precip2.z_score_four_wks('precip 2 rolling', ['pearson']))['onset_date'].dt.strftime('%B').value_counts().pipe(lambda x: x[x == x.max()]).index.tolist(),
                # 'Highest Month Precip (-2 Z-SCORE/4wk)(4 wk intervals)': (precip4.z_score_four_wks('precip 4 rolling', ['pearson']))['onset_date'].dt.strftime('%B').value_counts().pipe(lambda x: x[x == x.max()]).index.tolist(),
                
            }

            results.append(row)
        return pd.DataFrame(results)


