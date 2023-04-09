import math
import numpy as np


def create_derived_features(df):
    """
    모든 파생 변수를 생성하는 함수
    """
    df = add_air_flow_pressure(df)
    df = create_current_by_airflow(df)
    df = create_temp_diff(df)
    df = create_current_by_vibration(df)
    df = create_current_by_rpm(df)
    df = create_temp_by_vibration(df)
    df = create_rpm_diff_vibration(df)
    # =================================
    df = create_current_temp_diff(df)
    df = create_vibration_rpm_product(df)
    df = create_current_by_airflow_ratio(df)
    df = create_vibration_endtemp_product(df)
    df = create_rpm_airflow_product(df)
    # =================================

    df = compression_ratio(df)
    df = airflow_per_rotation(df)
    df = create_rpm_airflow_product(df)
    df = all_efficiency(df)
    df = air_density(df)
    df = volumetric_efficiency(df)
    df = work_input(df)
    return df


def add_motor_hp(df):
    """
    설비 번호에 따라 모터의 마력을 계산하여 feature로 추가합니다.

    Args:
        df (pandas.DataFrame): 모터 마력을 추가할 데이터프레임

    Returns:
        pandas.DataFrame: 모터 마력이 추가된 데이터프레임
    """
    df["motor_hp"] = 0
    df.loc[df["type"].isin([0, 4, 5, 6, 7]), "motor_hp"] = 30
    df.loc[df["type"] == 1, "motor_hp"] = 20
    df.loc[df["type"] == 2, "motor_hp"] = 10
    df.loc[df["type"] == 3, "motor_hp"] = 50
    return df


def add_air_flow_pressure(df):
    """
    air_inflow`와 `out_pressure`의 곱으로 이루어진 `air_flow_pressure` feature를 추가합니다.
    """

    df["air_flow_pressure"] = df["air_inflow"] * df["out_pressure"]
    return df


def create_current_by_airflow(df):
    """
    공기 흡입 속도에 따른 모터 전류 파생 변수 생성
    """
    df['current_by_airflow'] = df["motor_current"] / df["air_inflow"]
    return df


def create_temp_diff(df):
    """
    공기 말단 온도와 모터 온도 간의 차이 파생 변수 생성
    """
    df['temp_diff'] = df["motor_temp"] - df["air_end_temp"]
    return df


def create_current_by_vibration(df):
    """
    모터 진동에 따른 모터 전류 파생 변수 생성
    """
    df['current_by_vibration'] = df["motor_current"] * df["motor_vibe"]
    return df


def create_current_by_rpm(df):
    """
    모터 회전수에 따른 모터 전류 파생 변수 생성
    """
    df['current_by_rpm'] = df["motor_current"] / df["motor_rpm"]
    return df


def create_temp_by_vibration(df):
    df['temp_by_vibration'] = df["air_end_temp"] * df["motor_vibe"]
    return df


def create_rpm_diff_vibration(df):
    df['rpm_diff_vibration'] = df["motor_rpm"] - df["motor_vibe"]
    return df


"""
Add 23.04.08
"""


def create_current_temp_diff(df):
    """
    모터 전류와 공기 말단 온도 간의 차이 파생 변수 생성
    """
    df['current_temp_diff'] = df['motor_current'] - df['air_end_temp']
    return df


def create_vibration_rpm_product(df):
    """
    모터 진동과 모터 회전수 간의 곱 파생 변수 생성
    """
    df['vibration_rpm_product'] = df['motor_vibe'] * df['motor_rpm']
    return df


def create_current_by_airflow_ratio(df):
    """
    모터 전류와 공기 흡입 속도의 비율 파생 변수 생성
    """
    df['current_by_airflow_ratio'] = df['motor_current'] / df['air_inflow']
    return df


def create_vibration_endtemp_product(df):
    """
    모터 진동과 공기 말단 온도의 곱 파생 변수 생성
    """
    df['vibration_endtemp_product'] = df['motor_vibe'] * df['air_end_temp']
    return df


def create_rpm_airflow_product(df):
    """
    모터 회전수와 공기 흡입 속도의 곱 파생 변수 생성
    """
    df['rpm_airflow_product'] = df['motor_rpm'] * df['air_inflow']
    return df


# 가압비


def compression_ratio(df):
    df['compression_ratio'] = df['out_pressure'] / df['air_inflow']
    return df


# 회전수당 공기량


def airflow_per_rotation(df):
    df['airflow_per_rotation'] = df['air_inflow'] / df['motor_rpm']
    return df


def all_efficiency(df):
    df['efficiency'] = df['air_flow_pressure'] / \
        df['current_by_vibration']  # 전체 효율
    return df


def air_density(df):
    df['air_density'] = df['out_pressure'] / \
        (0.287 * (df['air_end_temp'] + 273.15))  # 공기 밀도
    return df


def volumetric_efficiency(df):
    df['volumetric_efficiency'] = df['air_inflow'] / \
        (df['motor_rpm'] * (df['motor_vibe'] / 1000))  # 용적 효율
    return df


def work_input(df):
    df['work_input'] = df['motor_current'] * \
        df['motor_rpm'] * df['motor_vibe']  # 입력 열
    return df



def add_motor_vibe_freq(df):
    """
    `motor_vibe` feature를 Fourier Transform을 이용하여 분해하여, 
    주파수 도메인에서의 `motor_vibe_freq1`와 `motor_vibe_freq2` feature를 추가합니다.

    Args:
        df (pandas.DataFrame): Feature가 추가될 데이터프레임

    Returns:
        pandas.DataFrame: Feature가 추가된 데이터프레임
    """
    freq = np.fft.fft(df["motor_vibe"])
    df["motor_vibe_freq1"] = freq[1].real
    df["motor_vibe_freq2"] = freq[2].real
    return df
