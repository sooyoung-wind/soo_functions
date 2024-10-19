import numpy as np
import pandas as pd
from typing import Union
from soo_functions import trans_to_UV, trans_to_WSWD


def me(F_data: Union[np.ndarray, list], A_data: Union[np.ndarray, list]) -> float:
    """
    평균 오차 (Mean Error) 계산 함수
    """
    return np.mean(np.array(F_data) - np.array(A_data))


def mae(F_data: Union[np.ndarray, list], A_data: Union[np.ndarray, list]) -> float:
    """
    평균 절대 오차 (Mean Absolute Error) 계산 함수
    """
    return np.mean(np.abs(np.array(F_data) - np.array(A_data)))


def rmse(F_data: Union[np.ndarray, list], A_data: Union[np.ndarray, list]) -> float:
    """
    제곱 평균 오차 (Root Mean Square Error) 계산 함수
    """
    return np.sqrt(np.mean((np.array(F_data) - np.array(A_data))**2))


def d(F_data: Union[np.ndarray, list], A_data: Union[np.ndarray, list]) -> float:
    """
    일치 지수 (Index of Agreement) 계산 함수
    """
    obs_mean = np.mean(A_data)
    numerator = np.sum((np.array(F_data) - np.array(A_data))**2)
    denominator = np.sum((np.abs(np.array(F_data) - obs_mean) + np.abs(np.array(A_data) - obs_mean))**2)
    return 1 - numerator / denominator


def soo_statistic(F_data: Union[np.ndarray, list], A_data: Union[np.ndarray, list], based_value: float = 30) -> pd.DataFrame:
    """
    F_data와 A_data 간의 다양한 통계 지표를 계산하는 함수.

    Args:
        F_data (np.ndarray or list): 예측값 데이터
        A_data (np.ndarray or list): 실제값 데이터
        based_value (float, optional): BRPI와 SBRPI를 계산할 때 사용할 기준 값. 기본값은 30.

    Returns:
        pd.DataFrame: 통계 지표 결과를 담은 데이터프레임
    """
    results = pd.DataFrame(np.full((1, 11), -999), columns=["F_mean", "A_mean", "ME", "Pbias", "MAE", "RMSE", "IOA", "R", "bcRMSE", "BRPI", "SBRPI"])

    results["F_mean"] = np.mean(F_data)
    results["A_mean"] = np.mean(A_data)
    results["ME"] = me(F_data, A_data)
    results["Pbias"] = (np.mean(F_data) - np.mean(A_data)) / np.mean(A_data) * 100
    results["MAE"] = mae(F_data, A_data)
    results["RMSE"] = rmse(F_data, A_data)
    results["IOA"] = d(F_data, A_data)
    results["R"] = np.corrcoef(F_data, A_data)[0, 1]
    results["bcRMSE"] = np.sqrt(rmse(F_data, A_data)**2 - me(F_data, A_data)**2)
    results["BRPI"] = results["bcRMSE"] / (results["IOA"] * results["R"])
    results["SBRPI"] = results["BRPI"] / based_value * 100

    return results


def wd_diff_cal(F_wd: Union[np.ndarray, list], A_wd: Union[np.ndarray, list]) -> np.ndarray:
    """
    예측 풍향과 실제 풍향 간의 차이를 라디안 값으로 계산하여 도(degree)로 반환하는 함수.

    Args:
        F_wd (np.ndarray or list): 예측 풍향 데이터
        A_wd (np.ndarray or list): 실제 풍향 데이터

    Returns:
        np.ndarray: 풍향 차이 (도 단위)
    """
    F_uv = trans_to_UV(ws=np.ones(len(F_wd)), wd=F_wd)
    A_uv = trans_to_UV(ws=np.ones(len(A_wd)), wd=A_wd)
    Diff_radian_value = np.arccos(F_uv['u'] * A_uv['u'] + F_uv['v'] * A_uv['v'])
    return Diff_radian_value * 180 / np.pi


def getmode(data: Union[np.ndarray, list]) -> float:
    """
    데이터의 최빈값을 반환하는 함수.
    """
    values, counts = np.unique(data, return_counts=True)
    mode_index = np.argmax(counts)
    return values[mode_index]


def mean_wd(ws: Union[np.ndarray, list], wd: Union[np.ndarray, list], cal_method: str) -> float:
    """
    풍속(ws)과 풍향(wd)을 기반으로 평균, 최빈값 또는 중앙값을 계산하는 함수.

    Args:
        ws (np.ndarray or list): 풍속 데이터
        wd (np.ndarray or list): 풍향 데이터
        cal_method (str): 계산 방법 ("mean", "mode", "median" 중 하나)

    Returns:
        float: 계산된 풍향 값
    """
    if cal_method == "mean":
        uv_values = trans_to_UV(ws=ws, wd=wd).mean()
        return trans_to_WSWD(u=uv_values['u'], v=uv_values['v'])['wd'].values[0]
    elif cal_method == "mode":
        return getmode(wd)
    elif cal_method == "median":
        uv_values = trans_to_UV(ws=ws, wd=wd).median()
        return trans_to_WSWD(u=uv_values['u'], v=uv_values['v'])['wd'].values[0]


def wd_statistic(
    F_data: Union[np.ndarray, list],
    A_data: Union[np.ndarray, list],
    based_value: float = 360,
    cal_method: str = "mode"
) -> pd.DataFrame:
    """
    F_data와 A_data 간의 다양한 풍향 통계 지표를 계산하는 함수.

    Args:
        F_data (np.ndarray or list): 예측값 데이터 (풍향)
        A_data (np.ndarray or list): 실제값 데이터 (풍향)
        based_value (float, optional): BRPI와 SBRPI를 계산할 때 사용할 기준 값. 기본값은 360.
        cal_method (str, optional): 계산 방법. 'mode', 'mean', 'median' 중 하나. 기본값은 'mode'.

    Returns:
        pd.DataFrame: 통계 지표 결과를 담은 데이터프레임
    """
    results = pd.DataFrame(np.full((1, 11), -999), columns=["F_mean", "A_mean", "ME", "Pbias", "MAE", "RMSE", "bcRMSE", "R", "IOA", "BRPI", "SBRPI"])
    F_minus_A = wd_diff_cal(F_data, A_data)

    if cal_method == "mode":
        results["F_mean"] = getmode(F_data)
        results["A_mean"] = getmode(A_data)
    elif cal_method == "mean":
        F_uv_mean = trans_to_UV(ws=np.ones(len(F_data)), wd=F_data).mean()
        results["F_mean"] = trans_to_WSWD(F_uv_mean["u"], F_uv_mean["v"])["wd"].values[0]

        A_uv_mean = trans_to_UV(ws=np.ones(len(A_data)), wd=A_data).mean()
        results["A_mean"] = trans_to_WSWD(A_uv_mean["u"], A_uv_mean["v"])["wd"].values[0]
    elif cal_method == "median":
        F_uv_median = trans_to_UV(ws=np.ones(len(F_data)), wd=F_data).median()
        results["F_mean"] = trans_to_WSWD(F_uv_median["u"], F_uv_median["v"])["wd"].values[0]

        A_uv_median = trans_to_UV(ws=np.ones(len(A_data)), wd=A_data).median()
        results["A_mean"] = trans_to_WSWD(A_uv_median["u"], A_uv_median["v"])["wd"].values[0]

    results["ME"] = results["F_mean"] - results["A_mean"]
    results["Pbias"] = results["ME"] / results["A_mean"] * 100
    results["MAE"] = np.mean(np.abs(F_minus_A))
    results["RMSE"] = np.sqrt(np.mean(F_minus_A**2))
    results["bcRMSE"] = np.sqrt(results["RMSE"]**2 - results["ME"]**2)

    F_minus_A_mean = wd_diff_cal(F_data, results["A_mean"].values[0])
    A_minus_A_mean = wd_diff_cal(A_data, results["A_mean"].values[0])
    results["IOA"] = 1 - (np.sum(F_minus_A**2) / np.sum((np.abs(F_minus_A_mean) + np.abs(F_minus_A_mean))**2))

    e_3 = np.sqrt(np.sum(F_minus_A_mean**2))
    e_4 = np.sqrt(np.sum(A_minus_A_mean**2))
    results["R"] = np.sum(F_minus_A_mean * A_minus_A_mean) / (e_3 * e_4)

    results["BRPI"] = results["bcRMSE"] / (results["R"] * results["IOA"])
    results["SBRPI"] = results["BRPI"] / based_value * 100

    return results
