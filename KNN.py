# 도미 길이
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
# 도미 무게
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]

# 빙어
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

import matplotlib.pyplot as plt

fish_length = bream_length + smelt_length
fish_weight = bream_weight + smelt_weight

import numpy as np

# 생선 데이터 벡터 처리
fish_data = np.column_stack((fish_length, fish_weight))

# 생선 정답 데이터 
fish_target = np.concatenate((np.ones(35), np.zeros(14)))

from sklearn.model_selection import train_test_split

# 훈련 데이터셋, 테스트 데이터셋 분류
train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, random_state = 42, stratify = fish_target)

from sklearn.neighbors import KNeighborsClassifier

# kn 모델 생성
kn = KNeighborsClassifier()

# 훈련 데이터셋의 평균, 표준편차 계산
mean = np.mean(train_input, axis = 0)
std = np.std(train_input, axis = 0)

# 훈련 데이터셋 표준점수 계산
train_scaled = (train_input - mean) / std

new_one = ([25, 150] - mean) / std

# 전처리된 데이터셋으로 훈련
kn.fit(train_scaled, train_target)

print('도미' if kn.predict([new_one]) == 1 else '빙어')
# 도미

# 새로운 물고기의 주변 이웃과의 거리와 인덱스 계산
distances, indexes = kn.kneighbors([new_one])

plt.scatter(train_scaled[:,0], train_scaled[:,1], c = 'blue')
plt.scatter(train_scaled[indexes, 0], train_scaled[indexes, 1], marker='D', c='red')
plt.scatter(new_one[0], new_one[1], c = 'yellow', marker= '^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

test_scaled = (test_input - mean) / std

print(kn.score(test_scaled, test_target))
# 1.0