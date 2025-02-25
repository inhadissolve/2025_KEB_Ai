import numpy as np
import pandas as pd

# pandas 사용
df = pd.DataFrame(
    {
        'A': [1, 2, np.nan, 4],
        'B': [np.nan, 12, 3, 4],
        'C': [1, 2, 3, 4]
    }
)

# 평균값으로 결측치 채우기
df_mean_filled = df.fillna(df.mean())
print(df_mean_filled)


from sklearn.impute import SimpleImputer
df = pd.DataFrame(
    {
        'A':[1, 2, np.nan, 4],
        'B':[np.nan, 12, 3, 4],
        'C':[1, 2, 3, 4]
    }
)
print(df)
i = SimpleImputer(strategy='mean')
df[['A', 'B']] = i.fit_transform(df[['A', 'B']])
print(df)


# apply()와 lambda 사용
df = pd.DataFrame(
    {
        'A': [1, 2, np.nan, 4],
        'B': [np.nan, 12, 3, 4],
        'C': [1, 2, 3, 4]
    }
)

df_lambda_filled = df.apply(lambda x: x.fillna(x.mean()), axis=0)

print(df_lambda_filled)

# transform() 사용
df = pd.DataFrame(
    {
        'A': [1, 2, np.nan, 4],
        'B': [np.nan, 12, 3, 4],
        'C': [1, 2, 3, 4]
    }
)

df_transform_filled = df.copy()
df_transform_filled[df.columns] = df.transform(lambda x: x.fillna(x.mean()))

print(df_transform_filled)