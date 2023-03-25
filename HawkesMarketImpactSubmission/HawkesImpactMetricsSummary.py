from math import e
import warnings
import joblib
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

from contextlib import closing
from sqlalchemy import create_engine
from typing import NamedTuple
from utils import extractDataFramesFromVar, EstimateHawkesProcess
from tqdm import tqdm
from scipy.stats import anderson

from scipy.integrate import cumtrapz
from scipy.integrate import dblquad
import os
import glob
import copy
from datetime import datetime
from datetime import timedelta
import argparse
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import sympy as sy

import statsmodels.api as sm
import scipy.stats as stats
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import expon, gaussian_kde

# User inputs
symbol_list = ["FLEX"]
total_mkt_vol = 9700000
kMinCostRatio = 4.5
nAMO = 20
exec_pct = 0.1
min_k = -10
max_k = -0.0001
start_time = '10:00'
end_time = '15:00'
interest_cost_ratio = 1 / 10000.0  # 1 bps
startDt = datetime(2019, 10, 1)
endDt = datetime(2019, 10, 7)
exchange = "NASDAQ"
dataPath = "./Data"
outputFilePath = f"./Data/"

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


class OptimalStrategyArgs(NamedTuple):
    omega: float
    gamma: float
    eta: float
    totalPos: float
    grid_sz: float  # in seconds
    T: float


class MarketImpactMetricsSummary:
    def __init__(self, symbol):
        self.sym = str(symbol)

    def load_raw_data(self, all_df_map, filename):
        temp_map = extractDataFramesFromVar(filename)
        # Filter out trading hours only
        for df_type in temp_map:
            if isinstance(temp_map[df_type].index, pd.DatetimeIndex):
                # Only extract time-series type of data
                temp_map[df_type] = temp_map[df_type].between_time(
                    start_time, end_time)
                if not temp_map[df_type].empty:
                    first_time_mark = float(
                        temp_map[df_type].iloc[0].time_elapsed)
                    temp_map[df_type]['time_elapsed'] = temp_map[df_type][
                        'time_elapsed'] - first_time_mark
                if df_type in all_df_map:
                    all_df_map[df_type] = all_df_map[df_type].append(
                        temp_map[df_type])
                else:
                    all_df_map[df_type] = temp_map[df_type]
        return all_df_map

    def estimateOrderFlowParams(self, tick_df):
        try:
            # Read the OFI from file
            ofi_data = tick_df["OFI"].copy(deep=True)
            ofi_data['ofi'] = ofi_data['ofi'].astype(float)
            ofi_data['lastMidPx'] = ofi_data['lastMidPx'].astype(float)
            ofi_df = ofi_data[['ofi']].resample('5S').sum()
            # Calculate ACF statistics
            acf = sm.tsa.stattools.acf(ofi_df['ofi'].values,
                                       nlags=10,
                                       alpha=0.05,
                                       qstat=1)
            lag_reject_str = ""
            for i in range(0, 10):
                is_reject = acf[0][i] < acf[1][i][1]
                lag_reject_str += str(int(is_reject))
            # Regress OFI
            avg_px = np.average(ofi_data['lastMidPx'])
            ofi_df = ofi_data[['ofi']].resample('1T').sum()
            px_df = ofi_data[['lastMidPx']].resample('1T').last()
            px_df['bpsChange'] = (px_df.lastMidPx -
                                  px_df.lastMidPx.shift(1)) / avg_px * 10000

            regress_df = ofi_df.merge(px_df, on='datetime')
            regress_df = regress_df.reset_index()
            regress_df = regress_df.dropna()
            regress_df['zScore'] = np.abs(stats.zscore(regress_df['ofi']))
            regress_df = regress_df[regress_df['zScore'] < 3]
            x = regress_df['ofi'].values.tolist()
            y = regress_df['bpsChange'].values.tolist()
            #x = sm.add_constant(x)
            res = sm.OLS(y, x).fit()
            if res.params[0] < 0:
                return pd.DataFrame()

            # Read trade data to calculate permanent cost
            agg_trade_df = tick_df["AGG_TRADE"].copy(deep=True)
            agg_trade_df = agg_trade_df[agg_trade_df['side'] == 'SELL']
            agg_trade_df.qty = agg_trade_df.qty.astype('float')
            avg_sell_sz = np.average(agg_trade_df['qty'])
            perm_cost = avg_sell_sz * exec_pct * res.params[0]
            params = {
                "OFI R2": res.rsquared * 100,
                "OFI coeff": res.params[0],
                "OFI p-value": res.pvalues[0],
                "Permanent cost": perm_cost,
            }
            return pd.DataFrame(params, index=[self.sym])
        except:
            print("Cannot regress OFI params")
            return pd.DataFrame()

    def estimateQuoteParams(self, tick_df, lambda_):
        # Read the trade from file
        agg_trade_df = tick_df["AGG_TRADE"].copy(deep=True)
        if agg_trade_df.empty:
            return pd.DataFrame()

        # Filter out only sell trades
        agg_trade_df = agg_trade_df[agg_trade_df['side'] == 'SELL']
        if agg_trade_df.empty:
            return pd.DataFrame()

        # Reset elapsed time
        agg_trade_df.qty = agg_trade_df.qty.astype('float')
        agg_trade_df['time_elapsed'] -= float(
            agg_trade_df.iloc[0].time_elapsed)

        # Aggregate time to 100 ms bins
        agg_trade_df = agg_trade_df.resample('100ms').first()
        agg_trade_df = agg_trade_df.dropna()

        # Estimate trade size
        avg_sell_sz = max(
            [np.median(agg_trade_df['qty']),
             np.average(agg_trade_df['qty'])])
        total_sell_sz = np.sum(agg_trade_df['qty'])

        # Read the L2 snapshot from file
        l2_data = tick_df["L2_SNAPSHOT"].copy(deep=True)
        for field in l2_data.columns:
            if field not in ['date', 'time']:
                l2_data[field] = l2_data[field].astype(float)

        # Find frequency of Poisson trade
        time_diff_df = pd.DataFrame()
        for dt in agg_trade_df['date'].unique():
            dt_trade_df = agg_trade_df[agg_trade_df['date'] == dt]
            dt_trade_df = dt_trade_df.reset_index()
            time_diff_df = time_diff_df.append(dt_trade_df['datetime'].diff().dropna()/ np.timedelta64(1, 's'))
        avg_duration = np.average(time_diff_df)
        
        # Resample to avg Poisson duration
        l2_data = l2_data.resample(str(int(avg_duration))+'S').first()
        l2_data = l2_data.dropna()

        # Extract bid data since we are selling
        avg_px = np.average(l2_data['bidPx_0'])

        def to_bps(px):
            return px / avg_px * 10000

        def priceImpact(trade_size, bandwidth):
            def calcJointKernel(px_index):
                depth_sz_1 = pd.DataFrame()
                depth_sz_2 = pd.DataFrame()
                for j in range(0, px_index + 1):
                    if j == 0:
                        depth_sz_1 = list(l2_data['bidQty_' + str(j)])
                    else:
                        depth_sz_1 = [
                            x + y for x, y in zip(
                                depth_sz_1, list(l2_data['bidQty_' + str(j)]))
                        ]
                for j in range(0, px_index + 2):
                    if j == 0:
                        depth_sz_2 = list(l2_data['bidQty_' + str(j)])
                    else:
                        depth_sz_2 = [
                            x + y for x, y in zip(
                                depth_sz_2, list(l2_data['bidQty_' + str(j)]))
                        ]
                values = np.vstack(
                    [np.asarray(depth_sz_1),
                     np.asarray(depth_sz_2)])
                return gaussian_kde(values, bw_method=bandwidth)

            px_impact = 0
            for depth_idx in range(0, 4):
                cur_gap = np.average(
                    to_bps(l2_data['bidPx_0'] -
                           l2_data['bidPx_' + str(depth_idx)]))
                next_gap = np.average(
                    to_bps(l2_data['bidPx_0'] -
                           l2_data['bidPx_' + str(depth_idx + 1)]))
                joint_kernel = calcJointKernel(depth_idx)

                def fyx(y, x):
                    return (
                        (trade_size - x) * next_gap +
                        cur_gap * x) * joint_kernel.pdf([x, y]) / trade_size

                hit_exp = dblquad(fyx,
                                  0,
                                  trade_size,
                                  trade_size,
                                  trade_size * 100,
                                  epsabs=0.1,
                                  epsrel=0.01)
                px_impact += hit_exp[0]
            return px_impact

        eta = None
        # Test to find a reasonable bandwidth
        for bandwidth in [25, 50, 75, 100]:
            # Calculate offset from BBO
            x = []
            y = []
            last_px_impact = 0
            for i in range(0, nAMO):
                sell_sz = avg_sell_sz * (i + 1)
                impact = priceImpact(sell_sz, bandwidth)
                if impact > 0.9 * last_px_impact:
                    x.append(sell_sz)
                    y.append(impact)
                    last_px_impact = impact
                else:
                    break

            # Perform regression
            res = sm.OLS(y, x).fit()
            # Calculate cost coefficient. Discard if cost is negative
            if res.params[0] < 0:
                continue

            cost_ratio = lambda_ / res.params[0]

            if cost_ratio < kMinCostRatio:
                continue
            else:
                eta = res.params[0]
                break

        if eta is None:
            print("Cannot regress quote params")
            return pd.DataFrame()

        params = {
            "Quote coeff": eta,
            "Quote p-value": res.pvalues[0],
            "Quote R2": res.rsquared * 100,
            "Trade size": avg_sell_sz,
            "Volume": total_sell_sz,
        }
        return pd.DataFrame(params, index=[self.sym])

    def estimateHawkesParams(self, tick_df, lambda_, eta):
        # Read the trade from file
        agg_trade_df = tick_df["AGG_TRADE"].copy(deep=True)
        if agg_trade_df.empty:
            return pd.DataFrame()

        # Filter out only sell trades
        agg_trade_df = agg_trade_df[agg_trade_df['side'] == 'SELL']
        if agg_trade_df.empty:
            return pd.DataFrame()

        # Reset elapsed time
        agg_trade_df.qty = agg_trade_df.qty.astype('float')
        agg_trade_df['time_elapsed'] -= float(
            agg_trade_df.iloc[0].time_elapsed)

        # Aggregate time to 100 ms bins
        agg_trade_df = agg_trade_df.resample('100ms').first()
        agg_trade_df = agg_trade_df.dropna()

        # estimate Hawkes parameters by minimizing the score (negative log-likelihood)
        min_score = 1e10
        cost_ratio = lambda_ / eta

        hawkes_params = []

        # search for best fit params
        for branching_ratio in np.linspace(0.15, 0.95, 50):
            # Convert to notation in the paper
            n = branching_ratio
            zeta = cost_ratio

            def calc_discrimator(k):
                return zeta**2 - 4 * (1 / n - 1)**2 * k

            min_alpha = (zeta +
                         np.sqrt(calc_discrimator(max_k))) / (2 *
                                                              (1 / n - 1)**2)
            max_alpha = (zeta +
                         np.sqrt(calc_discrimator(min_k))) / (2 *
                                                              (1 / n - 1)**2)
            test_hawkes_params = EstimateHawkesProcess(
                agg_trade_df['time_elapsed'], min_alpha, max_alpha, n)
            if test_hawkes_params is None:
                continue
            score = test_hawkes_params[3]
            residuals = test_hawkes_params[9]
            test_statistic = anderson(residuals, 'expon').statistic
            if score < min_score and test_statistic < 1.9:
                min_score = score
                hawkes_params = test_hawkes_params
                hawkes_ratio = hawkes_params[1] / hawkes_params[2]
                hawkes_statistic = test_statistic
                k = zeta * max_alpha - max_alpha**2 * (1 / n - 1)**2
        if not hawkes_params:
            return pd.DataFrame()
        # Parameters of Hawkes
        hawkes_mu = hawkes_params[0]
        hawkes_alpha = hawkes_params[1]
        hawkes_beta = hawkes_params[2]
        hawkes_score = hawkes_params[3]
        hawkes_rate = hawkes_params[4]
        hawkes_ratio = hawkes_alpha / hawkes_beta
        # Q-Q plot of Hawkes
        hawkes_qq_slope = hawkes_params[5]
        hawkes_qq_intercept = hawkes_params[6]
        hawkes_qq_r2 = hawkes_params[7]

        # Check for near-critical Hawkes
        if hawkes_ratio > 0.9:
            print("Cannot estimate Hawkes params due to branching ratio {}".
                  format(hawkes_ratio))
            return pd.DataFrame()

        # Check if the estimate makes sense
        omega = (hawkes_beta - hawkes_alpha)
        gamma = hawkes_alpha * lambda_ / omega
        theta = -omega * (omega - gamma / eta)
        if theta > max_k:
            print(
                "Cannot estimate Hawkes params due to theta {}".format(theta))
            return pd.DataFrame()

        params = {
            "Hawkes mu": hawkes_mu,
            "Hawkes alpha": hawkes_alpha,
            "Hawkes beta": hawkes_beta,
            "Hawkes ratio": hawkes_ratio,
            "Hawkes score": hawkes_score,
            "Hawkes average rate": hawkes_rate,
            "Hawkes Q-Q slope": hawkes_qq_slope,
            "Hawkes Q-Q intercept": hawkes_qq_intercept,
            "Hawkes Q-Q R2": hawkes_qq_r2,
            "Hawkes Anderson statistic": hawkes_statistic
        }
        return pd.DataFrame(params, index=[self.sym])

    def calculateStaticCost(self, lambda_, gamma, x0):
        return x0 * (lambda_ + gamma) / 2

    def calculateTWAPCost(self, args):
        # Load parameters to calculate the optimal trading strategy
        x0 = args.totalPos
        T = args.T
        omega = args.omega
        gamma = args.gamma
        eta = args.eta
        grid_sz = args.grid_sz

        # Based on the formula in the paper
        def E_t(t):
            return x0 / T

        def X_t(t):
            return x0 * t / T

        def C_t(t):
            return t * x0 ** 2 * eta / (T ** 2) - \
                (sy.exp(-t * omega) * x0 ** 2 * gamma * (1 + sy.exp(t * omega) * (-1 + t * omega))) / (T ** 2 * omega ** 2)

        # Due to the unstable solution tail, we need to sample and extrapolate a bit
        end_cost = []
        for s in list(np.linspace(T * 0.99, T, grid_sz)):
            end_cost.append(C_t(s))
        cost_per_share = max(end_cost) / x0
        return (cost_per_share, "Constant")

    def calculateOptimalCost(self, args):
        # Load parameters to calculate the optimal trading strategy
        x0 = args.totalPos
        T = args.T
        omega = args.omega
        gamma = args.gamma
        eta = args.eta
        grid_sz = args.grid_sz
        # Based on the formula in the paper

        theta = -omega * (omega - gamma / eta)
        sln_type = None
        if theta < 0:
            sln_type = "Hyperbolic"
            k = np.sqrt(-theta)
            kT2 = k * T / 2
            # Calculate C
            C_de = k**3 * x0 * eta * (omega * sy.cosh(kT2) + k * sy.sinh(kT2))
            C_num = k * T * omega * (k**2 * eta + gamma * omega) * sy.cosh(
                kT2) + (k**4 * T * eta + k**2 * T * gamma * omega -
                        2 * gamma * omega**2) * sy.sinh(kT2)
            C = C_de / C_num
            # Calculate C1
            C1_de = C * gamma * omega**2 * sy.cosh(kT2)
            C1_num = k**2 * eta * omega * sy.cosh(kT2) + k**3 * eta * sy.sinh(
                kT2)
            C1 = -C1_de / C1_num
            # Calculate C2
            C2_de = C * gamma * omega**2 * sy.sinh(kT2)
            C2_num = C1_num
            C2 = C2_de / C2_num
            # Calculate C3
            C3 = C * (1 + gamma * omega / (k**2 * eta))

            def E_t(t):
                return C1 * sy.cosh(k * t) + C2 * sy.sinh(k * t) + C3

            def X_t(t):
                return C1 / k * sy.cosh(k * t) + C2 / k * sy.sinh(
                    k * t) + C3 * t - C2 / k

            def C_t(t):
                # Calculate the first part of the cost equation
                Ct_1_de = eta * (
                    -C2 * (C1 + 4 * C3) + (C1**2 - C2**2 + 2 * C3**2) * k * t +
                    (C2 * sy.cosh(k * t) + C1 * sy.sinh(k * t)) *
                    (4 * C3 + C1 * sy.cosh(k * t) + C2 * sy.sinh(k * t)))
                Ct_1_num = 2 * k
                Ct_1 = Ct_1_de / Ct_1_num
                # Calculate the second part of the cost equation
                Ct_2 = 1 / (4 * omega * (-k + omega) * (k + omega)) * \
                    gamma * ( \
                        2*(C1 - C2)*(C1 + C2) * t * omega ** 2 + 4 * C3 ** 2 * t * (-k + omega) * (k + omega) + \
                        omega * (C1 ** 2 + C2 ** 2 - (2 * C1 * C2 * omega) / k) + \
                        (4 * C3 * (C3 * k ** 2 + C2 * k * omega - (C1 + C3) * omega ** 2)) / omega + \
                        (4 * omega * (-C2 ** 2 * k ** 3 - 2 * C2 * C3 * k ** 2 * omega + C1 ** 2 * k * omega ** 2 + 2 * C2 * C3 * omega ** 3)) / (k ** 3 - k * omega ** 2) - \
                        (4 * C3 * (C1 * k * omega + C2 * (k ** 2 - 2 * omega ** 2)) * sy.cosh(k * t)) / k - \
                        (omega * ((C1 ** 2 + C2 ** 2) * k - 2 * C1 * C2 * omega) * sy.cosh(2 * k * t)) / k - \
                        (4 * C3 * (C2 * k * omega + C1 * (k ** 2 - 2 * omega ** 2)) * sy.sinh(k * t)) / k + \
                        (4 * sy.exp(-t * omega) * (-C3 * k ** 2 - C2 * k * omega + (C1 + C3) * omega ** 2) * (C3*(-k ** 2 + omega ** 2) + omega*(C2 * k + C1 * omega) * sy.cosh(k * t) + omega*(C1 * k + C2 * omega) * sy.sinh(k * t))) / (omega*(-k + omega)*(k + omega)) + \
                        (omega * (-2 * C1 * C2 * k + (C1 ** 2 + C2 ** 2) * omega) * sy.sinh(2*k * t)) / k
                    )
                return Ct_1 - Ct_2
        elif theta == 0:
            sln_type = "Parabolic"
            # Calculate C
            C_de = 12 * x0
            C_num = T * (12 + omega * T * (6 + omega * T))
            C = C_de / C_num
            # Calculate C1, C2 and C3
            C1 = T * omega / 2 * C
            C2 = 1 / 2 * omega**2 * T * C

            def E_t(t):
                return C1 + C2 * t + (1 - omega**2 * t**2 / 2) * C

            def X_t(t):
                return C1 * t + C2 * t**2 / 2 + (t - omega**2 * t**3 / 6) * C

            def C_t(t):
                Ct_1 = 1 / 60 * t * eta * (20 *
                                           (3 * (C + C1)**2 + 3 *
                                            (C + C1) * C2 * t + C2**2 * t**2) -
                                           5 * C * t**2 *
                                           (4 *
                                            (C + C1) + 3 * C2 * t) * omega**2 +
                                           3 * C**2 * t**4 * omega**4)

                Ct_2 = (1 / (2 * omega**2)) * gamma * (
                    -2 * C1**2 +
                    (2 * C2**2) / omega**2 + 1 / 10 * C**2 * t**5 * omega**5 -
                    1 / 4 * C * t**4 * omega**3 * (2 * C2 + C * omega) + 2 *
                    (C + C1) * t * (-C2 + C1 * omega) + t**2 *
                    (-C2**2 + (C + 2 * C1) * C2 * omega + C *
                     (C + C1) * omega**2) + 1 / 3 * t**3 * omega *
                    (2 * C2**2 + 3 * C * C2 * omega - C *
                     (C + 2 * C1) * omega**2) -
                    (np.exp(-t * omega) * (C2 - C1 * omega) *
                     (2 * C1 * omega + 2 * C2 *
                      (1 + t * omega) - C * t * omega**2 *
                      (2 + t * omega))) / omega**2)
                return Ct_1 - Ct_2

        else:
            sln_type = "Trigonometric"
            k = np.sqrt(theta)
            kT2 = k * T / 2
            # Calculate C
            C_de = x0
            C_num = T + (T * gamma * omega) / (k**2 * eta) + (
                2 * gamma * omega**2) / (k**4 * eta -
                                         k**3 * eta * omega / sy.tan(kT2))
            C = C_de / C_num
            # Calculate C1
            C1_de = C * gamma * omega**2 * sy.cos(kT2)
            C1_num = -k**2 * eta * omega * sy.cos(kT2) + k**3 * eta * sy.sin(
                kT2)
            C1 = C1_de / C1_num
            # Calculate C2
            C2_de = C * gamma * omega**2 * sy.sin(kT2)
            C2_num = C1_num
            C2 = C2_de / C2_num
            # Calculate C3
            C3 = C * (1 + gamma * omega / (k**2 * eta))

            def E_t(t):
                return C1 * sy.cos(k * t) + C2 * sy.sin(k * t) + C3

            def X_t(t):
                return C1 / k * sy.sin(k * t) - C2 / k * sy.sinh(
                    k * t) + C3 * t + C2 / k

            def C_t(t):
                Ct_1 = (C2 * (C1 + 4 * C3) +
                        (C1**2 + C2**2 + 2 * C3**2) * k * t -
                        (C2 * sy.cos(k * t) - C1 * sy.sin(k * t)) *
                        (4 * C3 + C1 * sy.cos(k * t) + C2 * sy.sin(k * t))) / (
                            2 * k)
                Ct_2 = gamma / (4 * k * omega) * (\
                    - (2 * (C1 ** 2 + C2 ** 2 + 2 * C3 ** 2) * k) / omega + \
                    (2 * (C1 ** 2 + C2 ** 2 + 2 * C3 ** 2) * k * (sy.exp(-t * omega) + t * omega)) / omega + \
                    (8 * C3 * omega*(-C1 * k + C2 * omega)) / (k ** 2 + omega ** 2) + \
                    (2 * omega*(-C1 ** 2 * k + C2 ** 2 * k + C1 * C2 * omega)) / (4 * k ** 2 + omega ** 2) + \
                    8 * C3 * (-C2 + (sy.exp(-t * omega) * k * (C2 * k + C1 * omega)) / (k ** 2 + omega ** 2)) * sy.cos(k * t) + \
                    2 * (-C1 * C2 + (sy.exp(-t * omega) * k * (4 * C1 * C2 * k + C1 ** 2 * omega - C2 ** 2 * omega)) / (4 * k ** 2 + omega ** 2)) * sy.cos(2 * k * t) + \
                    8 * C3 * (C1 + (sy.exp(-t * omega) * k * (-C1 * k + C2 * omega)) / (k ** 2 + omega ** 2)) * sy.sin(k * t) + \
                    (C1**2 - C2**2 + (4*sy.exp(-t*omega)*k*(-C1**2*k + C2**2*k + C1*C2*omega))/(4*k**2 +omega**2))* sy.sin(2*k * t)
                )
                return Ct_1 - Ct_2

        # Due to the unstable solution tail, we need to sample and extrapolate a bit
        end_cost = []
        for s in list(np.linspace(T * 0.999, T, grid_sz)):
            end_cost.append(C_t(s))
        cost_per_share = min(end_cost) / x0
        return (cost_per_share, sln_type)

    def calcCosts(self,
                  omega,
                  lambda_,
                  gamma,
                  eta,
                  x0,
                  T,
                  is_over_critical_expected=False):
        strat_args = OptimalStrategyArgs(omega=omega,
                                         gamma=gamma,
                                         eta=eta,
                                         totalPos=x0,
                                         grid_sz=200,
                                         T=T)
        c_opt = self.calculateOptimalCost(strat_args)
        strat_args = OptimalStrategyArgs(omega=omega,
                                         gamma=gamma,
                                         eta=eta,
                                         totalPos=x0,
                                         grid_sz=200,
                                         T=T)
        c_naive = self.calculateTWAPCost(strat_args)
        c_static = float(self.calculateStaticCost(lambda_, gamma, x0))
        opt_strat_type = c_opt[-1]
        if c_opt is None or ("Tri" in opt_strat_type
                             and not is_over_critical_expected):
            print("Optimal cost is None")
            return None

        if (not is_over_critical_expected
                and (c_static < 0 or c_opt[0] < 0 or c_naive[0] < 0)):
            print("Negativve cost is detected")
            return None

        return (c_static, c_naive[0], c_opt[0], opt_strat_type)

    def scale_data(self, in_df, no_instances):
        scale_columns = [
            # Trade params
            "Hawkes mu",
            "Hawkes alpha",
            "Hawkes beta",
            "Hawkes ratio",
            "Hawkes score",
            "Hawkes average rate",
            "Poisson mu",
            "Poisson score",
            "Hawkes Q-Q intercept",
            "Hawkes Q-Q slope",
            "Hawkes Q-Q R2",
            "Hawkes Q-Q R2",
            "Hawkes Anderson statistic",
            # Quote params
            "Quote coeff",
            "Quote size",
            "Quote p-value",
            "Quote R2",
            "Trade size",
            # OFI params
            "OFI R2",
            "OFI coeff",
            "OFI p-value",
        ]
        in_df[list(set(in_df.columns) & set(scale_columns))] /= no_instances
        return in_df

    def run(self):
        time_blocks = [
            ("10:00", "11:00"),
            ("11:00", "12:00"),
            ("12:00", "13:00"),
            ("13:00", "14:00"),
            ("14:00", "15:00"),
        ]

        def estimateFactors(func):
            """
            General function to estimate params
            """
            currentDt = startDt
            avg_cost_params = pd.DataFrame()
            std_cost_params = pd.DataFrame()
            no_cost_estimations = 0
            while currentDt <= endDt:
                #==========================================================
                # Load daily data
                dt_str = currentDt.strftime("%Y%m%d")
                var_path = dataPath + "/var/" + exchange + "/" + dt_str + "/" + self.sym + ".var"
                currentDt += timedelta(days=1)
                all_df_map = {}
                if os.path.isfile(var_path):
                    all_df_map = self.load_raw_data(all_df_map, var_path)
                else:
                    continue
                hourly_df = []
                for time_block in time_blocks:
                    time_block_df = all_df_map.copy()
                    for df_type in time_block_df:
                        time_block_df[df_type] = time_block_df[
                            df_type].between_time(time_block[0], time_block[1])
                    hourly_df.append(time_block_df)
                #==========================================================
                # Estimating cost of aggressing the book
                for df in hourly_df:
                    params = func(df)
                    if params.empty:
                        continue
                    if avg_cost_params.empty:
                        avg_cost_params = params
                        std_cost_params = params**2
                    else:
                        avg_cost_params += params
                        std_cost_params += params**2
                    no_cost_estimations += 1

            if avg_cost_params.empty:
                return None, None
            avg_cost_params = self.scale_data(avg_cost_params,
                                              no_cost_estimations)
            std_cost_params = self.scale_data(
                std_cost_params, no_cost_estimations) - avg_cost_params**2

            return avg_cost_params, std_cost_params

        #==========================================================
        # ESTIMATE PARAMETERS
        #==========================================================
        # Estimate perm cost from fundamental price
        avg_perm_cost_params, std_perm_cost_params = estimateFactors(
            self.estimateOrderFlowParams)
        if avg_perm_cost_params is None:
            print("{} no permanent cost params".format(self.sym))
            return None
        # Extract cost params to guide instantaneous estimation
        lambda_ = avg_perm_cost_params.iloc[0]["OFI coeff"]

        # Estimate instant cost of book aggressing
        avg_inst_cost_params, std_inst_cost_params = estimateFactors(
            lambda df: self.estimateQuoteParams(df, lambda_))
        if avg_inst_cost_params is None:
            print("{} no instantaneous cost params".format(self.sym))
            return None

        # Extract cost params to guide Hawkes estimation
        eta = avg_inst_cost_params.iloc[0]["Quote coeff"]

        # Estimate Hawkes params
        avg_hawkes_params, std_hawkes_params = estimateFactors(
            lambda df: self.estimateHawkesParams(df, lambda_, eta))
        if avg_hawkes_params is None:
            print("{} no Hawkes cost params".format(self.sym))
            return None

        # Merge data into one dataframe based on symbol
        daily_df = pd.DataFrame()
        # Average of estimates
        avg_cost_params = pd.DataFrame()
        avg_cost_params = pd.concat([avg_inst_cost_params, avg_cost_params],
                                    axis=1)
        avg_cost_params = pd.concat([avg_perm_cost_params, avg_cost_params],
                                    axis=1)
        avg_cost_params = pd.concat([avg_hawkes_params, avg_cost_params],
                                    axis=1)
        avg_cost_params = pd.concat([daily_df, avg_cost_params], axis=1)
        
        # Std of estimates
        std_cost_params = pd.DataFrame()
        std_cost_params = pd.concat([std_inst_cost_params, std_cost_params],
                                    axis=1)
        std_cost_params = pd.concat([std_perm_cost_params, std_cost_params],
                                    axis=1)
        std_cost_params = pd.concat([std_hawkes_params, std_cost_params],
                                    axis=1)
        std_cost_params = pd.concat([daily_df, std_cost_params], axis=1)

        #==========================================================
        # ESTIMATE COSTS
        #==========================================================

        cost_estimates = {}
        cost_estimates['symbol'] = self.sym
        exec_pct = 0.2
        quote_coeff = avg_cost_params.iloc[0]["Quote coeff"]
        quote_coeff_p_val = avg_cost_params.iloc[0]["Quote p-value"]
        ofi_coeff = avg_cost_params.iloc[0]["OFI coeff"]
        ofi_coeff_p_val = avg_cost_params.iloc[0]["OFI p-value"]

        T = 5.5
        x0 = exec_pct * total_mkt_vol
        alpha = avg_cost_params["Hawkes alpha"].iloc[0]
        beta = avg_cost_params["Hawkes beta"].iloc[0]
        hawkes_statistic = avg_cost_params["Hawkes Anderson statistic"].iloc[0]
        omega = (beta - alpha)
        lambda_ = ofi_coeff
        gamma = alpha * lambda_ / omega
        eta = quote_coeff
        cost_ratio = lambda_ / eta

        theta = -omega * (omega - gamma / eta)
        print(theta)
        print(avg_cost_params)

        cost_res = self.calcCosts(omega, lambda_, gamma, eta, x0, T)
        if cost_res is None:
            print("Cannot estimate normal cost", cost_res, cost_ratio)
            return None

        static_cost = cost_res[0]
        interest_cost = static_cost * interest_cost_ratio
        naive_cost = cost_res[1]
        opt_cost = cost_res[2]
        opt_strat_type = cost_res[3]

        cost_estimates['symbol'] = self.sym
        cost_estimates['Alpha'] = alpha
        cost_estimates['Beta'] = beta
        cost_estimates['Hawkes statistic'] = hawkes_statistic
        cost_estimates['Lambda'] = lambda_
        cost_estimates['Lambda p-value'] = ofi_coeff_p_val
        cost_estimates['Gamma'] = gamma
        cost_estimates['Eta'] = eta
        cost_estimates['Eta p-value'] = quote_coeff_p_val
        cost_estimates['Omega'] = omega
        cost_estimates['Cost ratio'] = cost_ratio
        cost_estimates['Oscillation factor'] = theta
        cost_estimates['Optimal cost'] = opt_cost
        cost_estimates['Optimal strat type'] = opt_strat_type
        cost_estimates['Static cost'] = static_cost
        cost_estimates['Interest cost'] = interest_cost
        cost_estimates['TWAP cost'] = naive_cost
        cost_estimates['Cost improvement (Controllable)'] = float(
            (naive_cost - opt_cost) / abs(naive_cost) * 100)
        cost_estimates['Cost improvement (Full)'] = float(
            (naive_cost - opt_cost) / abs(naive_cost + interest_cost) * 100)

        # Push the cost ratio up 50%
        min_eta = alpha * lambda_ / (-0.1 + omega**2)
        eta_high_cost_ratio = max([min_eta, eta * 0.5])
        cost_res_high_cost_ratio = self.calcCosts(omega, lambda_, gamma,
                                                  eta_high_cost_ratio, x0, T)
        if cost_res_high_cost_ratio is None:
            print("Cannot estimate cost in high cost ratio scenario")
            return None

        naive_cost = cost_res_high_cost_ratio[1]
        opt_cost = cost_res_high_cost_ratio[2]
        opt_strat_type = cost_res_high_cost_ratio[3]

        cost_estimates['(A) Gamma'] = gamma
        cost_estimates['(A) Eta'] = eta_high_cost_ratio
        cost_estimates['(A) Omega'] = omega
        cost_estimates['(A) Cost ratio'] = cost_ratio
        cost_estimates['(A) Oscillation factor'] = float(theta)
        cost_estimates['(A) Optimal cost'] = opt_cost
        cost_estimates['(A) Optimal strat type'] = opt_strat_type
        cost_estimates['(A) Static cost'] = static_cost
        cost_estimates['(A) Interest cost'] = interest_cost
        cost_estimates['(A) TWAP cost'] = naive_cost
        cost_estimates['(A) Cost improvement (Controllable)'] = float(
            (naive_cost - opt_cost) / abs(naive_cost) * 100)
        cost_estimates['(A) Cost improvement (Full)'] = float(
            (naive_cost - opt_cost) / abs(naive_cost + interest_cost) * 100)

        print(pd.DataFrame.from_records([cost_estimates], index="symbol"))
        # Push the branching ratio up 50%
        current_hawkes_ratio = alpha / beta
        new_hawkes_ratio = min([current_hawkes_ratio * 1.5, 0.975])
        beta_high_br = alpha / new_hawkes_ratio
        min_omega = np.sqrt(alpha * lambda_ / eta + 0.1)
        omega_high_br = max([min_omega, beta_high_br - alpha])
        cost_res_high_br = self.calcCosts(omega_high_br, lambda_, gamma, eta,
                                          x0, T)
        if cost_res_high_br is None:
            print("Cannot estimate cost in high branching ratio scenario")
            return None

        naive_cost = cost_res_high_br[1]
        opt_cost = cost_res_high_br[2]
        opt_strat_type = cost_res_high_br[3]

        cost_estimates['(B) Gamma'] = gamma
        cost_estimates['(B) Eta'] = eta
        cost_estimates['(B) Omega'] = omega_high_br
        cost_estimates['(B) Cost ratio'] = cost_ratio
        cost_estimates['(B) Oscillation factor'] = float(theta)
        cost_estimates['(B) Optimal cost'] = opt_cost
        cost_estimates['(B) Optimal strat type'] = opt_strat_type
        cost_estimates['(B) Static cost'] = static_cost
        cost_estimates['(B) Interest cost'] = interest_cost
        cost_estimates['(B) TWAP cost'] = naive_cost
        cost_estimates['(B) Cost improvement (Controllable)'] = float(
            (naive_cost - opt_cost) / abs(naive_cost) * 100)
        cost_estimates['(B) Cost improvement (Full)'] = float(
            (naive_cost - opt_cost) / abs(naive_cost + interest_cost) * 100)

        # Set the cost ratio gamma/eta equals to omega for the critical case. We keep eta the same, while changing gamma
        new_omega = gamma / eta
        cost_res_critical = self.calcCosts(new_omega, lambda_, gamma, eta, x0,
                                           T)
        if cost_res_critical is None:
            print("Cannot estimate cost in critical scenario")
            return None

        naive_cost = cost_res_critical[1]
        opt_cost = cost_res_critical[2]
        opt_strat_type = cost_res_critical[3]

        cost_estimates['(C) Gamma'] = gamma
        cost_estimates['(C) Eta'] = eta
        cost_estimates['(C) Omega'] = new_omega
        cost_estimates['(C) Cost ratio'] = cost_ratio
        cost_estimates['(C) Oscillation factor'] = float(theta)
        cost_estimates['(C) Optimal cost'] = opt_cost
        cost_estimates['(C) Optimal strat type'] = opt_strat_type
        cost_estimates['(C) Static cost'] = static_cost
        cost_estimates['(C) Interest cost'] = interest_cost
        cost_estimates['(C) TWAP cost'] = naive_cost
        cost_estimates['(C) Cost improvement (Controllable)'] = float(
            (naive_cost - opt_cost) / abs(naive_cost) * 100)
        cost_estimates['(C) Cost improvement (Full)'] = float(
            (naive_cost - opt_cost) / abs(naive_cost + interest_cost) * 100)

        # Set the cost ratio gamma/eta equals to omega for the underdamped case
        new_omega = gamma / eta * 0.5
        cost_res_critical = self.calcCosts(new_omega, lambda_, gamma, eta, x0,
                                           T, True)

        naive_cost = cost_res_critical[1]
        opt_cost = cost_res_critical[2]
        opt_strat_type = cost_res_critical[3]

        cost_estimates['(D) Gamma'] = gamma
        cost_estimates['(D) Eta'] = eta
        cost_estimates['(D) Omega'] = new_omega
        cost_estimates['(D) Cost ratio'] = cost_ratio
        cost_estimates['(D) Oscillation factor'] = float(theta)
        cost_estimates['(D) Optimal cost'] = min([opt_cost, naive_cost])
        cost_estimates['(D) Optimal strat type'] = opt_strat_type

        cost_estimates = pd.DataFrame.from_records([cost_estimates],
                                                   index="symbol")

        return (avg_cost_params, std_cost_params, cost_estimates)


def runTask(symbol):
    print('Processing: {}'.format(symbol))
    return MarketImpactMetricsSummary(symbol).run()


if __name__ == "__main__":
    symbols = symbol_list
    all_res = []
    for sym in symbols:
        df = runTask(sym)
        all_res.append(df)
    avg_cost_params = pd.DataFrame()
    std_cost_params = pd.DataFrame()
    cost_estimates = pd.DataFrame()
    for df in all_res:
        if df is not None:
            avg_cost_params = avg_cost_params.append(df[0].round(8))
            std_cost_params = std_cost_params.append(df[1].round(8))
            cost_estimates = cost_estimates.append(df[2].round(8))
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter('{}/NASDAQ_Data.xlsx'.format(outputFilePath),
                            engine='xlsxwriter')

    # Write each dataframe to a different worksheet.
    all_average = pd.concat([avg_cost_params, cost_estimates], axis=1)
    all_average.to_excel(writer, sheet_name='AllAverage')
    cost_estimates.to_excel(writer, sheet_name='CostEstimates')
    avg_cost_params.to_excel(writer, sheet_name='AvgParams')
    std_cost_params.to_excel(writer, sheet_name='StdParams')

    # Close the Pandas Excel writer and output the Excel file.
    writer.save()
    print("Done")
