import pandas as pd
import numpy as np
import random
import os
from typing import Union

# PyTorch 임포트를 시도하고, 실패하면 None으로 설정
try:
    import torch
except ImportError:
    torch = None


def resumetable(df: pd.DataFrame) -> pd.DataFrame:
    """
    데이터프레임의 요약 정보를 제공하는 함수입니다.

    Parameters:
    df (pd.DataFrame): 요약 정보를 확인할 pandas 데이터프레임.

    Returns:
    pd.DataFrame: 각 피처에 대한 데이터 유형, 결측값 수, 결측값 비율 등 요약 정보가 담긴 데이터프레임.
    """
    print(f'데이터셋 형태: {df.shape}')
    summary = pd.DataFrame(df.dtypes, columns=['데이터 유형'])
    summary = summary.reset_index()
    summary = summary.rename(columns={'index': '피처'})
    summary['결측값 수'] = df.isnull().sum().values
    summary['결측값 비율'] = df.isnull().mean().values * 100
    print(f'Dataset shape: {df.shape}')
    summary = pd.DataFrame(df.dtypes, columns=['Data Type'])
    summary = summary.reset_index()
    summary = summary.rename(columns={'index': 'Feature'})
    summary['Missing Values Count'] = df.isnull().sum().values
    summary['Missing Values Percentage'] = df.isnull().mean().values * 100
    summary['Unique Values Count'] = df.nunique().values
    summary['Min Value'] = df.min().values
    summary['Max Value'] = df.max().values
    summary['First Value'] = df.iloc[0].values
    summary['Second Value'] = df.iloc[1].values
    summary['Third Value'] = df.iloc[2].values
    summary['Second to Last Value'] = df.iloc[-2].values
    summary['Last Value'] = df.iloc[-1].values
    return summary


def seed_everything(seed: int = 9234, set_torch_seed: bool = True) -> None:
    """
    랜덤 시드를 설정하여 실험의 재현성을 확보하는 함수입니다.

    Parameters:
    seed (int): 설정할 시드 값. 기본값은 9234입니다.
    set_torch_seed (bool): torch의 랜덤 시드를 설정할지 여부. 기본값은 True입니다.

    Returns:
    None
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if set_torch_seed:
        if torch is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True
        else:
            print("PyTorch가 설치되어 있지 않습니다. PyTorch 시드 설정은 건너뜁니다.")


def trans_to_WSWD(u: Union[np.ndarray, list], v: Union[np.ndarray, list]) -> pd.DataFrame:
    """
    주어진 u, v 성분을 이용해 풍속(ws)과 풍향(wd)을 계산하는 함수.

    풍향(wd)은 270도에서 시작하여 북쪽을 0도로 기준으로 시계방향으로 계산됩니다.
    풍속(ws)은 u, v 성분의 제곱 합에 루트를 씌워 계산됩니다.

    Args:
        u (np.ndarray or list): 동서 방향의 풍속 성분
        v (np.ndarray or list): 남북 방향의 풍속 성분

    Returns:
        pd.DataFrame: 풍속(ws)과 풍향(wd)을 포함하는 데이터프레임
    """
    wd = (270 - np.arctan2(v, u) * 180 / np.pi) % 360
    ws = np.sqrt(u**2 + v**2)
    data_results = pd.DataFrame({'ws': ws, 'wd': wd})
    return data_results


def trans_to_UV(ws: Union[np.ndarray, list], wd: Union[np.ndarray, list]) -> pd.DataFrame:
    """
    주어진 풍속(ws)과 풍향(wd)을 이용해 u, v 성분을 계산하는 함수.

    풍향(wd)은 북쪽을 0도로 기준으로 시계방향으로 측정되며,
    이를 이용해 u(동서 방향)과 v(남북 방향) 성분을 계산합니다.

    Args:
        ws (np.ndarray or list): 풍속
        wd (np.ndarray or list): 풍향 (단위: 도)

    Returns:
        pd.DataFrame: u(동서 방향 성분)과 v(남북 방향 성분)을 포함하는 데이터프레임
    """
    rad = np.pi / 180
    u = -ws * np.sin(rad * wd)
    v = -ws * np.cos(rad * wd)
    data_results = pd.DataFrame({'u': u, 'v': v})
    return data_results
