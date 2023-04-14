import math
import numpy as np
import pandas as pd

def create_derived_features(df):
    """
    모든 파생 변수를 생성하는 함수
    """
    
    df = add_motor_hp(df)
    df["air_flow_pressure"] = df["air_inflow"] * df["out_pressure"]
    df['current_by_airflow'] = df["motor_current"] / df["air_inflow"]
    df['temp_diff'] = df["motor_temp"] - df["air_end_temp"]
    df['current_by_vibration'] = df["motor_current"] * df["motor_vibe"]
    df['airflow_per_rotation'] = df['air_inflow'] / df['motor_rpm']
    df['efficiency'] = df['air_flow_pressure'] /df['current_by_vibration']  
    df['volumetric_efficiency'] = df['air_inflow'] / (df['motor_rpm'] * (df['motor_vibe'] / 1000))  # 용적 효율
    df['fan_laws_6'] = df['out_pressure'] * (df['motor_temp'] + 273) / (df['air_end_temp'] + 273)
    df['performance_index'] = (df['out_pressure'] * df['air_inflow']) / (df['motor_current'] * df['motor_temp'])
    df['air_to_motor_ratio'] = df['air_inflow'] / df['motor_rpm']  # 공기 유량과 모터 회전수의 비율
    df['air_to_power_ratio'] = df['air_inflow'] / (df['motor_current'] * df['motor_vibe']) # 공기 유량 대비 모터 전력
    df['mechanical_efficiency'] = (df['air_flow_pressure'] * df['motor_rpm']) / (df['motor_current'] * df['motor_vibe']) # 기계 효율
    df['air_density'] = df['air_inflow'] / (df['motor_rpm'] * (df['motor_vibe'] / 1000))  # 공기 밀도
    df['air_mass_flow'] = df['air_inflow'] * df['air_density']  # 공기 질량유량 
    df['air_inflow_efficiency1'] = df['air_inflow'] / (df['motor_rpm'] * (df['motor_temp'] / 1000))  # 용적 효율
    df['air_inflow_efficiency2'] = df['air_inflow'] / (df['motor_temp'] * (df['motor_vibe'] / 1000))  # 용적 효율
    df['motor_current_efficiency'] = df['motor_current'] / (df['motor_rpm'] * (df['motor_vibe'] / 1000))  # 용적 효율
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
