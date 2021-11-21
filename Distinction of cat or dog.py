# ねこの写真をグレースケール化して表示

sample_img1 = Image.open("train/cat-089.jpg").convert("L")
plt.imshow(sample_img1, cmap="gray")

# いぬの写真をグレースケール化して表示

sample_img2 = Image.open("train/dog-070.jpg").convert("L")
plt.imshow(sample_img2, cmap="gray")

# グレースケール化した写真をndarrayに変換してサイズを確認

sample_img1_array = np.array(sample_img1)
sample_img1_array.shape

# 訓練用データの読み込み

# ndarrayのデータを保管する領域の確保
train_len = len(train_data)

# 左右、上下、180度回転させたものを用意するため、4倍の容量を確保する

X_train = np.empty((train_len * 4, 5625), dtype=np.uint8)                
y_train = np.empty(train_len * 4, dtype=np.uint8)

# 画像ひとつひとつについて繰り返し処理

for i in range(len(train_data)):
    name = train_data.loc[i, "File name"] 
    train_img = Image.open(f"train/{name}.jpg").convert("L")
    train_img = np.array(train_img)  
    train_img_f = train_img.flatten()   
    X_train[i] = train_img_f  
    y_train[i] = train_data.loc[i, "DC"]
    
    # 左右反転させたものを訓練データに追加
    train_img_lr = np.fliplr(train_img)  
    train_img_lr_f = train_img_lr.flatten()
    X_train[i + train_len] = train_img_lr_f
    y_train[i + train_len] = train_data.loc[i, "DC"]
    
    # 上下反転させたものを訓練データに追加
    train_img_ud = np.flipud(train_img_lr) 
    train_img_ud_f = train_img_ud.flatten()
    X_train[i + train_len * 2] = train_img_ud_f
    y_train[i + train_len * 2] = train_data.loc[i, "DC"]

    
    # 180度回転させたものを訓練データに追加
    train_img_180 = np.rot90(train_img_lr, 2)   
    train_img_180_f = train_img_180.flatten()
    X_train[i + train_len * 3] = train_img_180_f
    y_train[i + train_len * 3] = train_data.loc[i, "DC"]


# テスト用データの読み込み

# ndarrayのデータを保管する領域の確保
test_len = len(test_data)
X_test = np.empty((test_len, 5625), dtype=np.uint8)
y_test = np.empty(test_len, dtype=np.uint8)

# 画像ひとつひとつについて繰り返し処理
for i in range(test_len):
    name = test_data.loc[i, "File name"]
    test_img = Image.open(f"test/{name}.jpg").convert("L")
    test_img = np.array(test_img)
    test_img_f = test_img.flatten()
    X_test[i] = test_img_f
    y_test[i] = test_data.loc[i, "DC"]


# 分類器の作成

classifier = SVC(kernel="linear")
classifier.fit(X_train, y_train)

# 分類の実施と結果表示

y_pred = classifier.predict(X_test)
y_pred

# 正解の表示

y_test

# 混同行列で正答数の確認

print(metrics.confusion_matrix(y_test, y_pred))

print(metrics.classification_report(y_test, y_pred))

