import math
import numpy as np
import pandas as pd

def create_derived_features(df):
    """
    모든 파생 변수를 생성하는 함수
    """
    
    df = add_motor_hp(df)
    df['air_inflow_hp'] = df['motor_hp'] / df['air_inflow']
    df['motor_current_hp'] = df['motor_hp'] / df['motor_current']
    df['motor_rpm_hp'] = df['motor_hp'] / df['motor_rpm']
    df['motor_temp_hp'] = df['motor_hp'] / df['motor_temp']
    df['motor_vibe_hp'] = df['motor_hp'] / df['motor_vibe']

    # # 토출 압력 대 공기 흡입 유량 비율
    df['pressure_flow_ratio'] = df['out_pressure'] / df['air_inflow']
    
    # # 모터 전압 대 모터 진동 비율
    df['vibe_voltage_ratio'] = df['motor_vibe'] / df['motor_current']

    # # 성능 인덱스
    df['performance_index'] = (df['out_pressure'] * df['air_inflow']) / (df['motor_current'] * df['motor_temp'])
    
    df['pressure_load_ratio'] = df['out_pressure'] / (df['motor_current'] / df['motor_hp'])
    # 토출 압력 대 모터 회전수 비율
    df['vibe_rpm_ratio'] = df['motor_rpm'] / df['motor_vibe']

    # 회전 + 진동 / 온도
    df['vibe+rpm_temp'] = (df['motor_vibe'] + df['motor_rpm']) / df['motor_temp']

    # 온도 차이
    df['temp_diff'] = df["motor_temp"] - df["air_end_temp"]
    
    # 들어올때 회전을 얼마나 하는지
    df['rpm*inflow'] = df["air_inflow"] / df["motor_rpm"]
    

    # compressed Air Flow Rate (m^3/hr)
    df['compressed_air_flow_rate'] = df['air_inflow'] * df['out_pressure'] / ((273.15 + df['air_end_temp']) * 0.1 * 60)
    
    # Vibration Velocity (mm/s): 모터 진동(motor_vibe)에 따른 진동 속도
    df['vibration_velocity'] = (df['motor_vibe'] * (2 * np.pi * df['motor_rpm'])) / 60


    df["air_flow_pressure"] = df["air_inflow"] * df["out_pressure"]
    df['current_by_vibration'] = df["motor_current"] * df["motor_vibe"]
    df['efficiency'] = df['air_flow_pressure'] / df['current_by_vibration'] 
    
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
