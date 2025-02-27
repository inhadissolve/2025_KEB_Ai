# Assignment
# 데이터 로딩 -> 데이터 전처리 -> 타겟 및 독립변수 설정 -> 트레이닝/테스트 셋 설정
# -> 모델 선택 및 학습 -> 예측 수행 -> 성능 평가(mse 등) -> 시각화

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score

mpg = sns.load_dataset('mpg')
# print(mpg.info())
# print(mpg[["horsepower"]].tail(10))
# print(mpg.tail(10))