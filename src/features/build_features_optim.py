import math
import numpy as np


def create_derived_features(df):
    """
    모든 파생 변수를 생성하는 함수
    """
    df = add_motor_hp(df)
    df = add_air_flow_pressure(df)
    df = create_current_by_airflow(df)
    df = create_temp_diff(df)
    df = create_current_by_vibration(df)
    df = compression_ratio(df)
    df = airflow_per_rotation(df)
    df = all_efficiency(df)
    df = power_efficiency(df)
    df['air_power'] = df['air_inflow'] * (df['out_pressure'] * 10)
    df['motor_power'] = (df['motor_current'] * df['motor_rpm']) / 1000
    df['power_efficiency'] = df['air_power'] / df['motor_power']
    df['vibration_index'] = (df['motor_vibe'] * 60) / df['motor_rpm']
    # # -------------------------------------
    df['temp_pressure'] = df['air_end_temp'] * df['out_pressure']
    df['temp_current'] = df['air_end_temp'] * df['motor_current']
    df['vibe_pressure'] = df['motor_vibe'] * df['out_pressure']

    df['air_inflow_squared'] = df['air_inflow'] ** 2
    df['air_end_temp_squared'] = df['air_end_temp'] ** 2
    df['out_pressure_squared'] = df['out_pressure'] ** 2
    df['motor_current_squared'] = df['motor_current'] ** 2
    df['motor_rpm_squared'] = df['motor_rpm'] ** 2
    df['motor_temp_squared'] = df['motor_temp'] ** 2
    df['motor_vibe_squared'] = df['motor_vibe'] ** 2

 #   ===========================================================
    df['air_density'] = df['out_pressure'] / (287 * (df['air_end_temp'] + 273))
    df['air_flow_rate'] = df['air_inflow'] / df['air_density']
    df['specific_power'] = df['air_power'] / df['air_flow_rate']
    df['motor_power_factor'] = np.cos(np.arccos(
        df['power_efficiency']) - np.arctan(df['motor_vibe'] / df['motor_current']))
    df['motor_power_density'] = df['motor_power'] / \
        (df['air_flow_rate'] * df['air_density'])
    df['specific_work'] = df['out_pressure'] * \
        1000 / df['air_density'] - 101325
    df['temperature_ratio'] = df['air_end_temp'] / df['motor_temp']
    df['pressure_ratio'] = df['out_pressure'] / \
        (101325 * df['motor_temp'] / df['air_end_temp'])
    df['pump_head'] = df['out_pressure'] * 10.1972
    df['fan_laws_1'] = df['air_flow_rate'] * df['pressure_ratio'] ** 0.5
    df['fan_laws_2'] = df['pump_head'] * df['pressure_ratio']
    df['fan_laws_3'] = df['motor_power'] * df['pressure_ratio'] ** 3

 #   ======================================================

    df['air_moles'] = df['out_pressure'] * \
        10**6 / (8.31 * (df['air_end_temp'] + 273))
    df['air_quality'] = (df['air_inflow'] * 60) / (1000 * df['air_moles'])
    df['electric_power'] = df['motor_current'] * 380 * np.sqrt(3) / 1000
    df['volumetric_efficiency'] = df['air_flow_rate'] / \
        (df['motor_rpm'] * 1000 / 60) * 100
    df['thermal_efficiency'] = (df['air_inflow'] * 60 * 1.4 * 287 *
                                (df['air_end_temp'] + 273)) / (df['motor_power'] * 1000)
    df['motor_torque'] = (9.55 * df['motor_power'] * 1000) / df['motor_rpm']
    df['motor_slip'] = (df['motor_rpm'] - (df['out_pressure'] * 10 / 2.45)
                        * 60 / (2 * np.pi * 50)) / df['motor_rpm'] * 100
    df['motor_frequency'] = df['motor_rpm'] / 60 * 50
    df['motor_impedance'] = (380 / df['motor_current']) / np.sqrt(3)
    df['motor_pf'] = df['power_efficiency'] / np.sqrt(df['power_efficiency']**2 + (
        df['motor_power_factor']**2 - df['power_efficiency']**2)**0.5)
    df['motor_ea'] = (df['motor_current'] / df['motor_impedance']
                      ) * np.sin(np.arccos(df['motor_pf']))
    df['motor_torque_factor'] = (df['motor_torque'] / (df['motor_current']
                                 * df['motor_impedance'])) * np.sin(np.arccos(df['motor_pf']))
    df['motor_efficiency'] = df['motor_power'] / df['electric_power'] * 100

    df['air_specific_heat'] = 1.005 + (df['air_end_temp'] - 25) * 0.00114
    df['delta_enthalpy'] = df['air_inflow'] * 60 * df['air_specific_heat'] * \
        (df['out_pressure'] * 10 / 2.45 - 1.01325) / 1000
    df['air_velocity'] = df['air_inflow'] / (np.pi * (0.15**2) / 4)
    df['fan_laws_4'] = df['air_flow_rate'] * \
        (df['motor_temp'] + 273) / (df['air_end_temp'] + 273)
    df['fan_laws_5'] = df['air_inflow'] * \
        (df['motor_temp'] + 273) / (df['air_end_temp'] + 273)
    df['fan_laws_6'] = df['out_pressure'] * \
        (df['motor_temp'] + 273) / (df['air_end_temp'] + 273)
    df['fan_laws_7'] = df['air_flow_rate'] * \
        (df['out_pressure'] / (101325 * df['motor_temp'] / df['air_end_temp'])) ** 0.5
        
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


def add_air_flow_pressure(df):
    """
    air_inflow`와 `out_pressure`의 곱으로 이루어진 `air_flow_pressure` feature를 추가합니다.
    """

    df["air_flow_pressure"] = df["air_inflow"] * df["out_pressure"]
    return df


def create_current_by_vibration(df):
    """
    모터 진동에 따른 모터 전류 파생 변수 생성
    """
    df['current_by_vibration'] = df["motor_current"] * df["motor_vibe"]
    return df


def create_temp_by_vibration(df):
    df['temp_by_vibration'] = df["air_end_temp"] * df["motor_vibe"]
    return df


def create_vibration_rpm_product(df):
    """
    모터 진동과 모터 회전수 간의 곱 파생 변수 생성
    """
    df['vibration_rpm_product'] = df['motor_vibe'] * df['motor_rpm']
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


def work_input(df):
    df['work_input'] = df['motor_current'] * \
        df['motor_rpm'] * df['motor_vibe']  # 입력 열
    return df


# 나누기 연산


def create_current_by_airflow(df):
    """
    공기 흡입 속도에 따른 모터 전류 파생 변수 생성
    """
    df['current_by_airflow'] = df["motor_current"] / df["air_inflow"]
    return df


def create_current_by_rpm(df):
    """
    모터 회전수에 따른 모터 전류 파생 변수 생성
    """
    df['current_by_rpm'] = df["motor_current"] / df["motor_rpm"]
    return df


def create_current_by_airflow_ratio(df):
    """
    모터 전류와 공기 흡입 속도의 비율 파생 변수 생성
    """
    df['current_by_airflow_ratio'] = df['motor_current'] / df['air_inflow']
    return df


def compression_ratio(df):
    df['compression_ratio'] = df['out_pressure'] / df['air_inflow']
    return df


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


def temp_rise_rate(df):
    df['temp_rise_rate'] = df['air_end_temp'] / df['motor_temp']
    return df


def motor_stability(df):
    df['motor_stability'] = df['motor_vibe'] / df['motor_temp']
    return df


def power_efficiency(df):
    df['power_efficiency'] = df['out_pressure'] * \
        df['air_inflow'] / df['motor_current']
    return df


# 빼기 연산


def create_temp_diff(df):
    """
    공기 말단 온도와 모터 온도 간의 차이 파생 변수 생성
    """
    df['temp_diff'] = df["motor_temp"] - df["air_end_temp"]
    return df


def create_rpm_diff_vibration(df):
    df['rpm_diff_vibration'] = df["motor_rpm"] - df["motor_vibe"]
    return df


def create_current_temp_diff(df):
    """
    모터 전류와 공기 말단 온도 간의 차이 파생 변수 생성
    """
    df['current_temp_diff'] = df['motor_current'] - df['air_end_temp']
    return df


'''
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
'''
