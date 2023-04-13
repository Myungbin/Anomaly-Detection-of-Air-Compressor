import math
import numpy as np


def create_derived_features(df):
    """
    모든 파생 변수를 생성하는 함수
    """
    # 토출 압력 대 공기 흡입 유량 비율
    df['pressure_flow_ratio'] = df['out_pressure'] / df['air_inflow']

    # 모터 부하
    df['motor_load'] = df.apply(get_motor_load, axis=1)

    # 모터 전압 대 모터 진동 비율
    df['vibe_voltage_ratio'] = df['motor_vibe'] / df['motor_current']

    # 성능 인덱스
    df['performance_index'] = (df['out_pressure'] * df['air_inflow']) / (df['motor_current'] * df['motor_temp'])

    # # 토출 압력 대 모터 진동 비율
    # df['pressure_vibe_ratio'] = df['out_pressure'] / df['motor_vibe']
    #
    # # 모터 전류 대 모터 온도 비율
    # df['current_temp_ratio'] = df['motor_current'] / df['motor_temp']
    #
    # # 모터 진동 평균값
    # df['motor_vibe_mean'] = df.groupby('type')['motor_vibe'].rolling(window=10, min_periods=1).mean().reset_index(
    #     drop=True)
    #
    # # 모터 온도 증가량
    # df['motor_temp_increase'] = df.groupby('type')['motor_temp'].diff().fillna(0)
    return df


def get_motor_load(x):
    if x['type'] in [0, 4, 5, 6, 7]:
        return x['motor_current'] / 30
    elif x['type'] == 1:
        return x['motor_current'] / 20
    elif x['type'] == 2:
        return x['motor_current'] / 10
    else:
        return x['motor_current'] / 50


