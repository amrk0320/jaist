# This program is designed to perform pattern recognition and classification of images. 
# This program is for pattern recognition and classification using iris types (Iris dataset). 
# CLAFIC method is used for the technique.
import sys
stdin = sys.stdin
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
import numpy as np
from collections import defaultdict

teach_percent = int(stdin.readline()[:-1])

# fetch_lfw_people
lfw = load_iris()

# 教師データ、テストデータの振り分け
classed_data = defaultdict(list)
teach_data = []
test_data = []

# 単位は全てcmなので正規化不要？
# 正規化しておく
from sklearn.preprocessing import StandardScaler
ms = StandardScaler()
lfw.data = ms.fit_transform(lfw.data)

# クラス毎にデータを分割する
for i in range(len(lfw.target)):
    # データ、正解
    classed_data[lfw.target[i]].append( (lfw.data[i] ,lfw.target[i] )  )

for c in classed_data:
    # 教師データ
    border = int(len( classed_data[c]  )*(teach_percent/100))
    for i in range( border ):
        teach_data.append( classed_data[c][i]  )

    # テストデータ
    for i in range(border, len(classed_data[c])):
        test_data.append( classed_data[c][i]  )

class_num = len(set(lfw.target))
classed_teach_data = defaultdict(list)
seiki_chokko_vector = defaultdict(list)
ruseki_kiyoritu = [0]*class_num

# クラス毎に分類する
for i in range(len(teach_data)):
    # lfw.data １次元配列(ベクトル)に圧縮したもの
    classed_teach_data[teach_data[i][1]].append( np.array([teach_data[i][0]]))

# クラスkの自己相関行列を求める、d*d^の行列
def make_ziko_sokan_gyoretu(c):
    # 列ベクトルと行ベクトルの内積
    gyoretu = np.dot(   classed_teach_data[c][0].T , classed_teach_data[c][0])
    for j in range(1,len(classed_teach_data[c]) ):
        gyoretu += np.dot( classed_teach_data[c][j].T , classed_teach_data[c][j]  )

    # データ数で割る
    gyoretu /= len(classed_teach_data[c]) 

    return gyoretu

# 累積寄与率の計算
def get_ruseki_kiyoritu(l):
    l1 = []
    for v1,v2 in l:
        l1.append(v1)
    ad = 1
    k = 0.99
    now = 0.0
    s = sum(l1)
    for i in range(len(l1)):
        now += l1[i]
        if (  now/s  ) < k:
            ad = i+1
    return ad

# 自己相関行列の計算
# クラスiに属するパターンx(パターン数ni)のクラス 自己相関行列を求める
for c in classed_teach_data:
    print("-----------------------------------")
    print("クラス"+str(c)+"の正規直行ベクトルを計算")
    ziko_sokan_gyoretu = make_ziko_sokan_gyoretu(c)
    # 固有値、固有ベクトル
    koyu_val, koyu_vector = np.linalg.eigh(ziko_sokan_gyoretu)
    r = sorted(zip(koyu_val, koyu_vector), reverse=True)
    for j in range(len(r)):
        # 固有ベクトルを保存する
        seiki_chokko_vector[c].append( r[j] )

    # 累積寄与率の計算
    ruseki_kiyoritu[c] = get_ruseki_kiyoritu(r)

    for scv in seiki_chokko_vector[c][:ruseki_kiyoritu[c]]:
        # 固有ベクトルを保存する
        print("正規直行ベクトル")
        print(scv[1])
        print("正規直行ベクトルの長さは1を確認")
        print(np.linalg.norm(scv[1], ord=2))
        print("固有値、固有ベクトル")
        print(r[j])


# 学習データの画像パターンを直交展開する
classed = [0]*len(test_data)
print("-----------------------------------")
print("ノルム計算")
for i in range(len(test_data)):
    norms = []
    for c in range(class_num):
        norm = 0.0
        for scv in seiki_chokko_vector[c][:ruseki_kiyoritu[c]]:
            norm += np.dot(test_data[i][0] , scv[1])**2
        norms.append(norm)
    # クラス分類
    classed[i] = np.argmax(norms)
    
from collections import defaultdict
ans = defaultdict(list)

ok = 0
ng = 0

for i in range(len(test_data)):
    seikai = test_data[i][1]
    result = classed[i]
    ans["クラス:"+str(seikai)].append(result)
    if seikai==result:
        ok += 1
    else:
        ng += 1

print("-----------------------------------")
print("結果")
print("-----------------------------------")
print("クラス毎分類", ans)
print("寄与率", ruseki_kiyoritu)
print("データ数", len(test_data))
print("正解率数", ok)
print("不正解数", ng)
print("正解率", ok/len(test_data))
