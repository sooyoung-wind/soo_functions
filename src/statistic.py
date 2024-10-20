import numpy as np
import warnings
import pandas as pd
from typing import Union
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from soo_functions import trans_to_UV, trans_to_WSWD


def _check_for_nan_or_inf(data: Union[np.ndarray, pd.Series]) -> bool:
    """
    하나의 데이터에서 NaN 또는 inf 값이 있는지 확인.
    NaN 또는 inf 값이 있으면 경고를 발생시키고 False 반환.

    Parameters:
    -----------
    data : np.ndarray 또는 pd.Series
        확인할 데이터.

    Returns:
    --------
    bool
        NaN 또는 inf 값이 없으면 True, 있으면 False를 반환.
    """
    if data.isnull().any() or np.isinf(data).any():
        warnings.warn("데이터에 NaN 또는 inf 값이 포함되어 있습니다.")
        return False
    return True


def _check_both_data_valid(F_data: Union[np.ndarray, pd.Series], A_data: Union[np.ndarray, pd.Series]) -> bool:
    """
    F_data와 A_data에서 모두 NaN 및 inf 값이 없는지 확인.
    둘 중 하나라도 NaN 또는 inf 값이 있으면 False 반환.

    Parameters:
    -----------
    F_data : np.ndarray 또는 pd.Series
        예측된 데이터.

    A_data : np.ndarray 또는 pd.Series
        실제 관측된 데이터.

    Returns:
    --------
    bool
        두 데이터 모두 NaN 및 inf 값이 없으면 True, 하나라도 있으면 False 반환.
    """
    return _check_for_nan_or_inf(F_data) and _check_for_nan_or_inf(A_data)


def _get_valid_mask(data: Union[np.ndarray, pd.Series]) -> np.ndarray:
    """
    데이터에서 NaN 및 inf 값을 제외한 유효한 값들의 마스크를 반환.

    Parameters:
    -----------
    data : np.ndarray 또는 pd.Series
        마스크를 생성할 데이터.

    Returns:
    --------
    np.ndarray
        NaN 및 inf 값을 제외한 유효한 값들을 나타내는 마스크 배열.
    """
    return ~np.isnan(data) & ~np.isinf(data)


def _get_combined_valid_mask(F_data: Union[np.ndarray, pd.Series], A_data: Union[np.ndarray, pd.Series]) -> np.ndarray:
    """
    F_data와 A_data 각각의 NaN 및 inf 값을 제외한 유효한 값들의 결합 마스크 반환.

    Parameters:
    -----------
    F_data : np.ndarray 또는 pd.Series
        예측된 데이터.

    A_data : np.ndarray 또는 pd.Series
        실제 관측된 데이터.

    Returns:
    --------
    np.ndarray
        F_data와 A_data에서 NaN 및 inf 값을 제외한 유효한 값들의 결합 마스크 배열.
    """
    valid_mask_F = _get_valid_mask(F_data)
    valid_mask_A = _get_valid_mask(A_data)
    return valid_mask_F & valid_mask_A


def _filter_valid_data(F_data: Union[np.ndarray, pd.Series], A_data: Union[np.ndarray, pd.Series]) -> tuple[np.ndarray, np.ndarray]:
    """
    F_data와 A_data에서 NaN 및 inf 값을 제외한 유효한 데이터만 반환.

    Parameters:
    -----------
    F_data : np.ndarray 또는 pd.Series
        예측된 데이터.

    A_data : np.ndarray 또는 pd.Series
        실제 관측된 데이터.

    Returns:
    --------
    np.ndarray, np.ndarray
        유효한 값만 남긴 F_data와 A_data 배열.
    """
    valid_mask = _get_combined_valid_mask(F_data, A_data)
    return F_data[valid_mask], A_data[valid_mask]


def me(F_data: Union[np.ndarray, pd.Series], A_data: Union[np.ndarray, pd.Series]) -> float:
    """
    평균 오차 (Mean Error) 계산 함수.
    NaN 및 inf 값을 확인하고, 값이 있으면 제외하고 계산.

    Parameters:
    -----------
    F_data : np.ndarray 또는 pd.Series
        예측된 데이터.

    A_data : np.ndarray 또는 pd.Series
        실제 관측된 데이터.

    Returns:
    --------
    float
        예측된 데이터와 실제 데이터 간의 평균 오차 값.
    """
    # NaN 및 inf 값이 있는지 확인
    if not _check_both_data_valid(F_data, A_data):
        # NaN 및 inf 값을 제외한 유효한 데이터만 사용
        F_data, A_data = _filter_valid_data(F_data, A_data)

    return np.mean(F_data - A_data)


def mae(F_data: Union[np.ndarray, pd.Series], A_data: Union[np.ndarray, pd.Series]) -> float:
    """
    평균 절대 오차 (Mean Absolute Error) 계산 함수

    Parameters:
    -----------
    F_data : np.ndarray 또는 pd.Series
        예측된 데이터입니다.

    A_data : np.ndarray 또는 pd.Series
        실제 관측된 데이터입니다.

    Returns:
    --------
    float
        예측된 데이터와 실제 데이터 간의 평균 절대 오차 값.
    """
    # NaN 및 inf 값이 있는지 확인
    if not _check_both_data_valid(F_data, A_data):
        # NaN 및 inf 값을 제외한 유효한 데이터만 사용
        F_data, A_data = _filter_valid_data(F_data, A_data)

    # np.mean(np.abs(np.array(F_data) - np.array(A_data))) 구버전
    return mean_absolute_error(y_pred=F_data, y_true=A_data)


def rmse(F_data: Union[np.ndarray, pd.Series], A_data: Union[np.ndarray, pd.Series]) -> float:
    """
    제곱 평균 오차 (Root Mean Square Error) 계산 함수

    Parameters:
    -----------
    F_data : np.ndarray 또는 pd.Series
        예측된 데이터입니다.

    A_data : np.ndarray 또는 pd.Series
        실제 관측된 데이터입니다.

    Returns:
    --------
    float
        예측된 데이터와 실제 데이터 간의 제곱 평균 오차 값.
    """
    # NaN 및 inf 값이 있는지 확인
    if not _check_both_data_valid(F_data, A_data):
        # NaN 및 inf 값을 제외한 유효한 데이터만 사용
        F_data, A_data = _filter_valid_data(F_data, A_data)

    # np.sqrt(np.mean((np.array(F_data) - np.array(A_data))**2)) 구버전
    return root_mean_squared_error(y_pred=F_data, y_true=A_data)


def d(F_data: Union[np.ndarray, pd.Series], A_data: Union[np.ndarray, pd.Series]) -> float:
    """
    일치 지수 (Index of Agreement) 계산 함수

    Parameters:
    -----------
    F_data : np.ndarray 또는 pd.Series
        예측된 데이터입니다.

    A_data : np.ndarray 또는 pd.Series
        실제 관측된 데이터입니다.

    Returns:
    --------
    float
        예측된 데이터와 실제 데이터 간의 일치 지수.
    """
    # NaN 및 inf 값이 있는지 확인
    if not _check_both_data_valid(F_data, A_data):
        # NaN 및 inf 값을 제외한 유효한 데이터만 사용
        F_data, A_data = _filter_valid_data(F_data, A_data)

    obs_mean = np.mean(A_data)
    numerator = np.sum((np.array(F_data) - np.array(A_data))**2)
    denominator = np.sum((np.abs(np.array(F_data) - obs_mean) + np.abs(np.array(A_data) - obs_mean))**2)
    return 1 - numerator / denominator


def soo_statistic(F_data: Union[np.ndarray, pd.Series], A_data: Union[np.ndarray, pd.Series], based_value: float = 30) -> pd.DataFrame:
    """
    F_data와 A_data 간의 다양한 통계 지표를 계산하는 함수

    Parameters:
    -----------
    F_data : np.ndarray 또는 pd.Series
        예측된 데이터입니다.

    A_data : np.ndarray 또는 pd.Series
        실제 관측된 데이터입니다.

    based_value : float, 선택 사항
        BCPI와 SBCPI를 계산할 때 사용할 기준 값. 기본값은 30입니다.
        BCPI : Bias-Corrected Performance Index
        SBCPI : Scaled Bias-Corrected Performance Index

    Returns:
    --------
    pd.DataFrame
        다양한 통계 지표를 포함한 데이터프레임.
    """
    results = pd.DataFrame(np.full((1, 11), -999), columns=["F_mean", "A_mean", "ME", "Pbias", "MAE", "RMSE", "IOA", "R", "bcRMSE", "BCPI", "SBCPI"])

    results["F_mean"] = np.mean(F_data)
    results["A_mean"] = np.mean(A_data)
    results["ME"] = me(F_data, A_data)
    results["Pbias"] = (np.mean(F_data) - np.mean(A_data)) / np.mean(A_data) * 100
    results["MAE"] = mae(F_data, A_data)
    results["RMSE"] = rmse(F_data, A_data)
    results["IOA"] = d(F_data, A_data)
    results["R"] = np.corrcoef(F_data, A_data)[0, 1]
    results["bcRMSE"] = np.sqrt(rmse(F_data, A_data)**2 - me(F_data, A_data)**2)
    results["BCPI"] = results["bcRMSE"] / (results["IOA"] * results["R"])
    results["SBCPI"] = results["BCPI"] / based_value * 100

    return results


def wd_diff_cal(F_wd: Union[np.ndarray, pd.Series], A_wd: Union[np.ndarray, pd.Series]) -> np.ndarray:
    """
    예측 풍향과 실제 풍향 간의 차이를 라디안 값으로 계산하여 도(degree)로 반환하는 함수

    Parameters:
    -----------
    F_wd : np.ndarray 또는 pd.Series
        예측된 풍향 데이터입니다.

    A_wd : np.ndarray 또는 pd.Series
        실제 관측된 풍향 데이터입니다.

    Returns:
    --------
    np.ndarray
        예측된 풍향과 실제 풍향 간의 차이 (도 단위).
    """
    F_uv = trans_to_UV(ws=np.ones(len(F_wd)), wd=F_wd)
    A_uv = trans_to_UV(ws=np.ones(len(A_wd)), wd=A_wd)
    Diff_radian_value = np.arccos(F_uv['u'] * A_uv['u'] + F_uv['v'] * A_uv['v'])
    return Diff_radian_value * 180 / np.pi


def _getmode(data: Union[np.ndarray, pd.Series]) -> float:
    """
    데이터의 최빈값을 반환하는 함수

    Parameters:
    -----------
    data : np.ndarray 또는 pd.Series
        계산할 데이터입니다.

    Returns:
    --------
    float
        데이터의 최빈값.
    """
    values, counts = np.unique(data, return_counts=True)
    mode_index = np.argmax(counts)
    return values[mode_index]


def mean_wd(ws: Union[np.ndarray, pd.Series], wd: Union[np.ndarray, pd.Series], cal_method: str = "mode") -> float:
    """
    풍속(ws)과 풍향(wd)을 기반으로 평균, 최빈값 또는 중앙값을 계산하는 함수

    Parameters:
    -----------
    ws : np.ndarray 또는 pd.Series
        풍속 데이터입니다.

    wd : np.ndarray 또는 pd.Series
        풍향 데이터입니다.

    cal_method : str
        계산 방법. 'mean', 'mode', 'median' 중 하나.

    Returns:
    --------
    float
        계산된 풍향 값.
    """
    # TODO #6 mean 동작안됨
    if cal_method == "mean":
        uv_values = trans_to_UV(ws=ws, wd=wd).mean()
        return trans_to_WSWD(u=uv_values['u'], v=uv_values['v'])['wd'].values[0]
    elif cal_method == "mode":
        return _getmode(wd)
    # TODO #5 median 동작안됨
    elif cal_method == "median":
        uv_values = trans_to_UV(ws=ws, wd=wd).median()
        return trans_to_WSWD(u=uv_values['u'], v=uv_values['v'])['wd'].values[0]


def wd_statistic(
    F_data: Union[np.ndarray, pd.Series],
    A_data: Union[np.ndarray, pd.Series],
    based_value: float = 360,
    cal_method: str = "mode"
) -> pd.DataFrame:
    """
    F_data와 A_data 간의 다양한 풍향 통계 지표를 계산하는 함수

    Parameters:
    -----------
    F_data : np.ndarray 또는 pd.Series
        예측된 풍향 데이터입니다.

    A_data : np.ndarray 또는 pd.Series
        실제 관측된 풍향 데이터입니다.

    based_value : float, 선택 사항
        BRPI와 SBRPI를 계산할 때 사용할 기준 값. 기본값은 360입니다.

    cal_method : str, 선택 사항
        풍향 값을 계산하는 방법. 'mode', 'mean', 'median' 중 하나. 기본값은 'mode'입니다.

    Returns:
    --------
    pd.DataFrame
        다양한 풍향 통계 지표를 포함한 데이터프레임.
    """
    # TODO #7 wd_statistic 함수 수정하기
    results = pd.DataFrame(np.full((1, 11), -999), columns=["F_mean", "A_mean", "ME", "Pbias", "MAE", "RMSE", "bcRMSE", "R", "IOA", "BRPI", "SBRPI"])
    F_minus_A = wd_diff_cal(F_data, A_data)

    if cal_method == "mode":
        results["F_mean"] = _getmode(F_data)
        results["A_mean"] = _getmode(A_data)
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

    results["BCPI"] = results["bcRMSE"] / (results["R"] * results["IOA"])
    results["SBCPI"] = results["BCPI"] / based_value * 100

    return results
