import math
import numpy as np
import pandas as pd

def create_derived_features(df):
    """
    모든 파생 변수를 생성하는 함수
    """
    # 토출 압력 대 공기 흡입 유량 비율
    df['pressure_flow_ratio'] = df['out_pressure'] / df['air_inflow']
    
    # 모터 전압 대 모터 진동 비율
    df['vibe_voltage_ratio'] = df['motor_vibe'] / df['motor_current']

    # 성능 인덱스
    df['performance_index'] = (df['out_pressure'] * df['air_inflow']) / (df['motor_current'] * df['motor_temp'])
    
    df['pressure_load_ratio'] = df['out_pressure'] / (df['motor_current'] / df.apply(lambda row: 30 if row['type'] in [0, 4, 5, 6, 7] else (20 if row['type'] == 1 else (10 if row['type'] == 2 else 50)), axis=1))

    # 토출 압력 대 모터 회전수 비율
    df['vibe_rpm_ratio'] = df['motor_rpm'] / df['motor_vibe']

    # 회전 + 진동 / 온도
    df['vibe+rpm_temp'] = (df['motor_vibe'] + df['motor_rpm']) / df['motor_temp']

    # 온도 차이
    df['temp_diff'] = df["motor_temp"] - df["air_end_temp"]
    
    # 들어올때 회전을 얼마나 하는지
    df['rpm*inflow'] = df["air_inflow"] / df["motor_rpm"]
    
    return df
