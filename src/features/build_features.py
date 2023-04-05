import numpy as np


def add_motor_hp(data):
    """
    설비 번호에 따라 모터의 마력을 계산하여 feature로 추가합니다.

    Args:
        data (pandas.DataFrame): 모터 마력을 추가할 데이터프레임

    Returns:
        pandas.DataFrame: 모터 마력이 추가된 데이터프레임
    """
    data["motor_hp"] = 0
    data.loc[data["type"].isin([0, 4, 5, 6, 7]), "motor_hp"] = 30
    data.loc[data["type"] == 1, "motor_hp"] = 20
    data.loc[data["type"] == 2, "motor_hp"] = 10
    data.loc[data["type"] == 3, "motor_hp"] = 50
    return data


def add_air_flow_pressure(data):
    """
    `air_inflow`와 `out_pressure`의 곱으로 이루어진 `air_flow_pressure` feature를 추가합니다.

    Args:
        data (pandas.DataFrame): Feature가 추가될 데이터프레임

    Returns:
        pandas.DataFrame: Feature가 추가된 데이터프레임
    """
    data["air_flow_pressure"] = data["air_inflow"] * data["out_pressure"]
    return data


def add_motor_vibe_freq(data):
    """
    `motor_vibe` feature를 Fourier Transform을 이용하여 분해하여, 
    주파수 도메인에서의 `motor_vibe_freq1`와 `motor_vibe_freq2` feature를 추가합니다.

    Args:
        data (pandas.DataFrame): Feature가 추가될 데이터프레임

    Returns:
        pandas.DataFrame: Feature가 추가된 데이터프레임
    """
    freq = np.fft.fft(data["motor_vibe"])
    data["motor_vibe_freq1"] = freq[1].real
    data["motor_vibe_freq2"] = freq[2].real
    return data
