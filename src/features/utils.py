import numpy as np


def outlier_z_score_filter_df(df, threshold=3.5):
    """
    데이터프레임의 모든 feature에 대해 Z-score 방법을 이용하여 이상치를 제거하고, NA 값을 가진 행을 제거하는 함수
    :param df: 이상치를 제거할 데이터프레임
    :param threshold: 이상치 판단 기준값 (일반적으로 3 사용)
    :return: 이상치가 제거된 데이터프레임
    """
    # 데이터프레임의 모든 feature에 대해 Z-score 방법을 이용하여 이상치 제거
    filtered_df = df.apply(lambda x: x[np.abs((x - x.mean()) / x.std()) < threshold])

    # NA 값을 가진 행 제거
    filtered_df = filtered_df.dropna()

    # 결과 반환
    return filtered_df