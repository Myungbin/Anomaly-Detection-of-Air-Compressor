import numpy as np
import pandas as pd

def outlier_z_score_filter_df(df, threshold=3):
    """
    데이터프레임의 모든 feature에 대해 Z-score 방법을 이용하여 이상치를 제거하고, NA 값을 가진 행을 제거하는 함수
    :param df: 이상치를 제거할 데이터프레임
    :param threshold: 이상치 판단 기준값 (일반적으로 3 사용)
    :return: 이상치가 제거된 데이터프레임
    """
    # 데이터프레임의 모든 feature에 대해 Z-score 방법을 이용하여 이상치 제거
    filtered_df = df.apply(
        lambda x: x[np.abs((x - x.mean()) / x.std()) < threshold])

    # NA 값을 가진 행 제거
    print("Drop Data \n", filtered_df.isna().sum())
    filtered_df = filtered_df.fillna(method='ffill')
    # filtered_df = filtered_df.dropna()

    # 결과 반환
    return filtered_df


def moving_average(df, period):
    """
    데이터프레임에서 이동평균을 구하는 함수

    :param df: 이동평균을 계산할 데이터프레임
    :param period: 이동평균의 기간
    :return: 이동평균이 추가된 데이터프레임
    """
    ma = df.rolling(period).mean().fillna(method='bfill')  # 이동평균 계산
    ma.columns = [col + '_MA' + str(period) for col in df.columns]  # 컬럼 이름 변경
    return pd.concat([df, ma], axis=1)  # 데이터프레임 합치기
