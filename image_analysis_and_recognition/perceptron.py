# This program simulates the perceptron learning algorithm. Specifically, it addresses the 2-class classification problem, learning weight parameters on linearly separable data.
# We define the discriminant function g(x) =x1*w1 + x2*w2.
# For each data point, the value of g(x) is computed, and if the result does not match the class label (i.e., in the case of misclassification), the weights and thresholds are updated. The update is based on the average of the feature values of the misclassified data points and the negative average of the g(x) values

# ; 入力
# ; 0.5 0.5 0.5
# ; 6
# ; 1 1.2 1
# ; 1 0.2 1
# ; 1 -0.2 1
# ; 1 -0.5 2
# ; 1 -1 2
# ; 1 -1.5 2

# ; 出力
# ; 4


import sys
stdin = sys.stdin

# 重み1、重み2、ρ
w1,w2,r = [float(x) for x in stdin.readline().rstrip().split()]
n = int(stdin.readline()[:-1])

l = []
for i in range(n):
    x1, x2, c = [float(x) for x in stdin.readline().rstrip().split()]
    l.append(( x1,x2,c  ))

def gx(x1,w1,x2,w2):
    return x1*w1 +x2*w2

# 学習回数
ans = 0
while True:
    ans += 1
    # 識別関数の差分
    diff_gx=[]
    # 特徴の差分
    diff_x1=[]
    diff_x2=[]
    for i in range(n):
        g = gx(l[i][0],w1,l[i][1],w2)
        # クラス間違い
        if (0< g and l[i][2]==2) or (g<0 and l[i][2]==1):
            # 差分を保存
            diff_gx.append(-1*g)
            diff_x1.append(  l[i][0] )
            diff_x2.append(  l[i][1] )
    # 差分がなければ終了
    if len(diff_gx)==0:
        break
    else:
        diff_avg = sum(diff_gx)/len(diff_gx)
        diff_x1_avg = sum(diff_x1)/len(diff_x1)
        diff_x2_avg = sum(diff_x2)/len(diff_x2)
        r += diff_avg
        w1 += diff_x1_avg*r
        w2 += diff_x2_avg*r

# 学習回数
print(ans)
