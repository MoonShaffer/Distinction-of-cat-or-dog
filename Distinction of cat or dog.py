# 必要なライブラリのインポート

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.svm import SVC
from sklearn import metrics
import matplotlib.pyplot as plt

%matplotlib inline


# 訓練データ用CSVの読み込み

train_data = pd.read_csv("train/train_data.csv")
train_data.head()


# テストデータ用CSVの読み込み

test_data = pd.read_csv("test/test_data.csv")
test_data.head()

