import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

import datetime
import pandas as pd
import numpy as np
import math
from scipy.optimize import minimize

from joblib import Parallel, delayed
from tqdm import tqdm
import scipy.stats as stats
from sortedcontainers import SortedDict
from scipy.stats import expon, gaussian_kde, kstest

from google.cloud.bigquery import (Client as BQClient, job, Table,
                                   TableReference, DatasetReference)

class TickTable(object):
    """ Tick table for each exchange """
    def __init__(self, exch, series):
        if exch == "XTKS":
            if series == "TOPIX Large70" or series == "TOPIX Core30":
                self.ttb = [(1000, 0.1), (3000, 0.5), (10000, 1), (30000, 5),
                            (100000, 10), (300000, 50), (1000000, 100),
                            (3000000, 500), (10000000, 1000), (30000000, 5000),
                            (float('Inf'), 10000)]
            else:
                self.ttb = [(3000, 1), (5000, 5), (30000, 10), (50000, 50),
                            (300000, 100), (500000, 500), (3000000, 1000),
                            (5000000, 5000), (30000000, 10000),
                            (50000000, 50000), (float('Inf'), 100000)]
        elif exch == "XSES":
            self.ttb = [(0.2, 0.001), (0.995, 0.005), (float('Inf'), 0.01)]
        elif exch == "NASDAQ":
            self.ttb = [(1, 0.0001), (float('Inf'), 0.01)]

    def getTickSize(self, px):
        for tick_info in self.ttb:
            if px <= tick_info[0]:
                return tick_info[1]

    def priceToTick(self, px):
        for tick_info in self.ttb:
            if px <= tick_info[0]:
                return px / tick_info[1]

    def tickToPrice(self, px, ticks):
        for tick_info in self.ttb:
            if px <= tick_info[0]:
                return ticks * tick_info[1]


class Clock(object):
    """ A clock that ticks """
    def __init__(self):
        self.timed_callback_ = SortedDict()
        # Default to epoch
        self.current_time_ = datetime.datetime.fromtimestamp(0)

    def now(self):
        return self.current_time_

    def insert_timed_callback(self, t, cb):
        if t not in self.timed_callback_:
            self.timed_callback_[t] = []
        self.timed_callback_[t].append(cb)

    def advance(self):
        if not self.timed_callback_:
            return False
        for t, all_cb in self.timed_callback_.items():
            self.current_time_ = t
            for cb in all_cb:
                cb()
            self.timed_callback_.pop(t, None)
            break
        return True


class EventIdBuffer(object):
    """ Buffer of event id within a rolling window """
    def __init__(self, buffer_interval):
        self.buffer_interval_ = buffer_interval
        self.timed_events = []

    def capture(self, new_time, new_event_id):
        self.timed_events = [
            x for x in self.timed_events
            if x[0] > new_time - self.buffer_interval_
        ]
        self.timed_events.append((new_time, new_event_id))

    def clear(self):
        self.timed_events = []


def roundTime(dt, roundTo):
    """Round a datetime object to any time lapse in seconds
   dt : datetime.datetime object, default now.
   roundTo : Closest number of seconds to round to, default 1 minute.
   Author: Thierry Husson 2012 - Use it as you want but don't blame me.
   """
    seconds = (dt.replace(tzinfo=None) - dt.min).seconds
    rounding = (seconds + roundTo / 2) // roundTo * roundTo
    return dt + datetime.timedelta(0, rounding - seconds, -dt.microsecond)


def extractDataFramesFromVar(filePath):
    """
    Extract dataframes for each types from var file
    """
    df_map = {}
    with open(filePath) as f:
        rows = []
        headers = []
        is_end_of_headers = False
        for line in f.readlines():
            if is_end_of_headers:
                rows.append([x.strip() for x in line.split(',')])
            elif "END_OF_HEADERS" in line:
                is_end_of_headers = True
            else:
                headers.append([x.strip() for x in line.split(',')])
        for header in headers:
            df_type = header[0]
            df_rows = []
            for row in rows:
                if df_type in row:
                    df_rows.append(row[1:])
            df_map[df_type] = pd.DataFrame(df_rows, columns=header[1:])
            if "time" in header:
                # Convert to timestamp and float seconds since midnight
                df_map[df_type]['datetime'] = pd.to_datetime(
                    df_map[df_type]['time'])
                df_map[df_type]['date'] = pd.to_datetime(
                    df_map[df_type]['datetime']).dt.date
                df_map[df_type]['time'] = pd.to_datetime(
                    df_map[df_type]['datetime']).dt.time
                df_map[df_type]['time_elapsed'] = (
                    (df_map[df_type]['datetime'] -
                     df_map[df_type]['datetime'].dt.normalize()) /
                    pd.Timedelta('1 second')).astype(float)
                df_map[df_type].set_index('datetime', inplace=True)
                df_map[df_type] = df_map[df_type].infer_objects()
    return df_map


def text_progessbar(seq, total=None):
    step = 1
    tick = time.time()
    while True:
        time_diff = time.time() - tick
        avg_speed = time_diff / step
        total_str = 'of %n' % total if total else ''
        print('step', step, '%.2f' % time_diff,
              'avg: %.2f iter/sec' % avg_speed, total_str)
        step += 1
        yield next(seq)


all_bar_funcs = {
    'tqdm': lambda args: lambda x: tqdm(x, **args),
    'txt': lambda args: lambda x: text_progessbar(x, **args),
    'False': lambda args: iter,
    'None': lambda args: iter,
}


def ParallelExecutor(use_bar='tqdm', **joblib_args):
    def aprun(bar=use_bar, **tq_args):
        def tmp(op_iter):
            if str(bar) in all_bar_funcs.keys():
                bar_func = all_bar_funcs[str(bar)](tq_args)
            else:
                raise ValueError("Value %s not supported as bar type" % bar)
            return Parallel(**joblib_args)(bar_func(op_iter))

        return tmp

    return aprun


def DifferentiateScalar(f, a, b, n):
    """
    Compute the discrete derivative of a Python function
    f on [a,b] using n intervals. Internal points apply
    a centered difference, while end points apply a one-sided
    difference.
    """
    x = np.linspace(a, b, n + 1)  # mesh
    df = np.zeros_like(x)  # df/dx
    f_vec = f(x)
    dx = x[1] - x[0]
    # Internal mesh points
    for i in range(1, n):
        df[i] = (f_vec[i + 1] - f_vec[i - 1]) / (2 * dx)
    # End points
    df[0] = (f_vec[1] - f_vec[0]) / dx
    df[-1] = (f_vec[-1] - f_vec[-2]) / dx
    return df


def DifferentiateVector(f, a, b, n):
    """
    Compute the discrete derivative of a Python function
    f on [a,b] using n intervals. Internal points apply
    a centered difference, while end points apply a one-sided
    difference. Vectorized version.
    """
    x = np.linspace(a, b, n + 1)  # mesh
    df = np.zeros_like(x)  # df/dx
    f_vec = f(x)
    dx = x[1] - x[0]
    # Internal mesh points
    df[1:-1] = (f_vec[2:] - f_vec[:-2]) / (2 * dx)
    # End points
    df[0] = (f_vec[1] - f_vec[0]) / dx
    df[-1] = (f_vec[-1] - f_vec[-2]) / dx
    return df


def TraperzoidalScalar(f, a, b, n):
    """
    Compute the integral of f from a to b with n intervals,
    using the Trapezoidal rule.
    """
    h = (b - a) / float(n)
    I = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        x = a + i * h
        I += f(x)
    I = h * I
    return I


def TraperzoidalDual(f, a, b, c, d, n):
    """
    Compute the integral of f from a to b, x to y with n intervals,
    using the Trapezoidal rule.
    """
    h = (b - a) / float(n)
    I = 0.5 * (f(a, c) + f(b, d))
    for i in range(1, n):
        x = a + i * h
        I += TraperzoidalScalar(lambda y: f(x, y), c, d, n)
    I = h * I
    return I


def TraperzoidalVector(f, a, b, n):
    """
    Compute the integral of f from a to b with n intervals,
    using the Trapezoidal rule. Vectorized version.
    """
    x = np.linspace(a, b, n + 1)
    f_vec = f(x)
    f_vec[0] /= 2.0
    f_vec[-1] /= 2.0
    h = (b - a) / float(n)
    I = h * np.sum(f_vec)
    return I


def _recursive(timestamps, beta):
    r_array = np.zeros(len(timestamps))
    for i in range(1, len(timestamps)):
        r_array[i] = np.exp(
            -beta * (timestamps[i] - timestamps[i - 1])) * (1 + r_array[i - 1])
    return r_array


def log_likelihood(timestamps, mu, alpha, beta, runtime):
    r = _recursive(timestamps, beta)
    return -runtime * mu + alpha / beta * np.sum(
        np.exp(-beta *
               (runtime - timestamps)) - 1) + np.sum(np.log(mu + alpha * r))


def crit(params, *args):
    mu, alpha, beta = params
    timestamps, runtime = args
    ll = -log_likelihood(timestamps, mu, alpha, beta, runtime)
    if math.isinf(ll):
        return 1e9
    else:
        return ll


def EstimateHawkesProcessResidual(time_series, hawkes_mu, hawkes_alpha,
                                  hawkes_beta):
    # Estimate the residuals
    def theo_intensity(t1, t2):

        theo_int = hawkes_mu * (t2 - t1)
        for i in range(1, len(time_series)):
            tk = time_series[i]
            if tk <= t1:
                delta_t_1 = (t1 - tk)
                delta_t_2 = (t2 - tk)

                if delta_t_1 < 1e3:
                    theo_int -= hawkes_alpha / hawkes_beta * (
                        np.exp(-delta_t_2 * hawkes_beta) -
                        np.exp(-delta_t_1 * hawkes_beta))
            else:
                break
        return theo_int

    resid = [0] * len(time_series)
    for i in range(1, len(time_series)):
        resid[i] = theo_intensity(time_series[i - 1], time_series[i])
    return resid[1:]


def crit_fixed_branching_ratio(params, *args):
    mu, alpha, branching_ratio = params
    beta = alpha / branching_ratio
    timestamps, runtime = args
    ll = -log_likelihood(timestamps, mu, alpha, beta, runtime)
    if math.isinf(ll):
        return 1e9
    else:
        return ll


def EstimateHawkesProcess(time_series, alpha_min, alpha_max, branching_ratio):
    # From https://stats.stackexchange.com/questions/24685/finding-the-mle-for-a-univariate-exponential-hawkes-process
    poisson_params = EstimatePoissonProcess(time_series)
    poisson_mean = poisson_params[0]
    poisson_score = poisson_params[1]
    m = poisson_mean / 2
    n = 1.0
    a = alpha_min
    normalized_t = time_series.to_numpy()
    normalized_t = (normalized_t - min(normalized_t))
    normalized_t = normalized_t / max(normalized_t)
    normalized_t.sort()
    res = minimize(
        crit_fixed_branching_ratio,
        [m, a, branching_ratio],
        args=(normalized_t, n),
        bounds=((poisson_mean / 2, poisson_mean * 2), (alpha_min, alpha_max),
                [0, 1]),
        options={'maxiter': 1000},
    )
    hawkes_mu = res.x[0]
    hawkes_alpha = res.x[1]
    hawkes_beta = hawkes_alpha / branching_ratio
    best_score = res.fun
    if best_score == 1e9:
        return None

    hawkes_resid = EstimateHawkesProcessResidual(
        normalized_t,
        hawkes_mu,
        hawkes_alpha,
        hawkes_beta,
    )
    df = pd.DataFrame(hawkes_resid, columns=['Nt'])
    df = df.dropna()
    df = df.reset_index()
    res = stats.probplot(df['Nt'] + 1, dist=stats.expon(1))
    _, p_val = kstest(df['Nt'], 'expon')

    # Estimate the mean intensity
    hawkes_avg_evt = hawkes_mu / (1.0 - hawkes_alpha / hawkes_beta)

    return [
        hawkes_mu, hawkes_alpha, hawkes_beta, best_score, hawkes_avg_evt,
        res[1][0], res[1][1], res[1][2], p_val, df['Nt']
    ]


def EstimatePoissonProcess(time_series):
    normalized_t = time_series
    normalized_t = (normalized_t - min(normalized_t))
    normalized_t = normalized_t / max(normalized_t)
    dur_t = normalized_t.diff().dropna().to_numpy()
    poisson_mean = 1.0 / np.mean(dur_t)
    poisson_score = -log_likelihood(normalized_t.to_numpy(), poisson_mean, 0,
                                    1, 1.0)
    return (poisson_mean, poisson_score)


def ReplaceNaN(x):
    """
    Replace NaN values in a list with nearby values
    """
    for j in range(1, len(x)):
        if np.isnan(x[j]):
            x[j] = x[j - 1]
    for j in range(0, len(x) - 1):
        if np.isnan(x[j]):
            x[j] = x[j + 1]
    x = [v for v in x if not np.isnan(v)]
    return x

def QueryHistoricalTradePx(symbol, start_date, end_date, bar_size):
    # Query index volatility - proxied by DBS
    rows = []
    px_query = f"SELECT time,price FROM `ghpr-prod.Box.trade` WHERE date >= '{start_date}' and date < '{end_date}' and sym = '{symbol}';"
    for row in BQClient(project='ghpr-prod').query(px_query).result():
        rows.append((row[0] + datetime.timedelta(hours=8), row[1]))
    price_ts = pd.DataFrame(rows, columns=['Datetime', 'Px'])
    price_ts = price_ts[price_ts['Px'] > 0]
    price_ts.drop_duplicates(subset="Datetime", keep=False, inplace=True)
    price_ts = price_ts.set_index('Datetime')
    price_ts = price_ts.resample(bar_size).bfill()
    price_ts['Time'] = price_ts.index.time
    price_ts['Date'] = price_ts.index.date
    if "XTKS" in symbol:
        price_ts = price_ts[price_ts['Time'] < datetime.time(14, 0, 0)]
        price_ts = price_ts[price_ts['Time'] > datetime.time(8, 0, 0)]
    elif "XSES" in symbol:
        price_ts = price_ts[price_ts['Time'] < datetime.time(17, 0, 0)]
        price_ts = price_ts[price_ts['Time'] > datetime.time(9, 0, 0)]
    price_ts['PxChange'] = price_ts['Px'].diff().astype(float)
    price_ts['PrevChange'] = price_ts.shift(1)['PxChange']
    price_ts = price_ts.dropna(axis=0)
    price_ts['Symbol'] = symbol
    return price_ts