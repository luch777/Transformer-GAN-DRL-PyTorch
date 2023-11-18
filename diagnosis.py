import numpy as np
import scipy.io as sio
import random
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import VotingClassifier, BaggingClassifier
from tqdm import tqdm
import warnings
import pandas as pd
from config import args

warnings.filterwarnings('ignore')

select_number = args.select_number
size = select_number * 6
GAN = args.GAN

# # 生成的数据、标签
# test_data = np.loadtxt('GAN_data_base_' + str(select_number) + '.txt')
# # test_data = np.loadtxt('GAN_data5.txt')
# test_labels = np.loadtxt('labels1500.txt')
#
# # 测试阶段的数据、标签
# x_test = np.loadtxt('test_data.txt')
# y_test = np.loadtxt('test_labels.txt')
# 生成的数据、标签
test_data = np.loadtxt('GAN_data_base_' + str(select_number) + '.txt')
# test_data = np.loadtxt('GAN_data5.txt')
test_labels = np.loadtxt('labels1500.txt')

# 测试阶段的数据、标签
x_test = np.loadtxt('test_data.txt')
y_test = np.loadtxt('test_labels.txt')

# x_test = min_max_scaler.fit_transform(x_test)


def Get_Average(list):
    sum = 0
    for item in list:
        sum += item
    return sum / len(list)


def training(train_sample, train_label, test_sample):
    # SVM
    train_label = train_label.ravel()
    clf = SVC(kernel='rbf', C=75, gamma=8)
    clf.set_params(kernel='rbf', probability=True).fit(train_sample, train_label)
    SVM_predict = clf.predict(test_sample)

    # 随机森林
    rfc1 = RandomForestClassifier(n_estimators=40, max_depth=None, min_samples_split=2, random_state=20)
    rfc1.fit(train_sample, train_label)
    RF_predict = rfc1.predict(test_sample)

    # 多层感知机
    MLP = MLPClassifier(solver='lbfgs', alpha=1e-4,
                        hidden_layer_sizes=(100, 3), random_state=1)
    MLP.fit(train_sample, train_label)
    MLP_predict = MLP.predict(test_sample)

    # # 决策树
    # dtc = DecisionTreeClassifier()
    # dtc.fit(train_sample, train_label)
    # dt_predict = dtc.predict(test_sample)
    return SVM_predict, RF_predict, MLP_predict


x_train = np.loadtxt('train_data' + str(select_number) + '.txt')
y_train = np.loadtxt('train_labels' + str(select_number) + '.txt')
if GAN:
    select_x, select_y = CHOSE()
    x_train = np.concatenate([x_train, select_x])
    y_train = np.concatenate([y_train, select_y])

for i in tqdm(range(10)):
    # RF_AC, SVM_AC, DT_AC, MLP_AC, KNN_AC, f1_rf, f1_SVM, \
    # f1_KNN, f1_MLP, f1_dt = classify_1(x_train, y_train)
    RF_AC, ENS_AC, DT_AC, BG_AC, KNN_AC, f1_rf, f1_ENS, f1_KNN, f1_BG, f1_dt = classify_1(x_train, y_train)
    if i == 0:
        ave_RF = RF_AC
        # ave_SVM = SVM_AC
        ave_ENS = ENS_AC
        ave_DT = DT_AC
        # ave_MLP = MLP_AC
        ave_BG = BG_AC
        ave_KNN = KNN_AC

        ave_f1_RF = f1_rf
        # ave_f1_SVM = f1_SVM
        ave_f1_ENS = f1_ENS
        ave_f1_DT = f1_dt
        # ave_f1_MLP = f1_MLP
        ave_f1_BG = f1_BG
        ave_f1_KNN = f1_KNN
    else:
        ave_RF = np.append(ave_RF, RF_AC)
        # ave_SVM = np.append(ave_SVM, SVM_AC)
        ave_ENS = np.append(ave_ENS, ENS_AC)
        ave_DT = np.append(ave_DT, DT_AC)
        # ave_MLP = np.append(ave_MLP, MLP_AC)
        ave_BG = np.append(ave_BG, BG_AC)
        ave_KNN = np.append(ave_KNN, KNN_AC)

        ave_f1_RF = np.append(ave_f1_RF, f1_rf)
        # ave_f1_SVM = np.append(ave_f1_SVM, f1_SVM)
        ave_f1_ENS = np.append(ave_f1_ENS, f1_ENS)
        ave_f1_DT = np.append(ave_f1_DT, f1_dt)
        # ave_f1_MLP = np.append(ave_f1_MLP, f1_MLP)
        ave_f1_BG = np.append(ave_f1_BG, f1_BG)
        ave_f1_KNN = np.append(ave_f1_KNN, f1_KNN)

print('\n')
print('RF accuracy acc: {:.2f}% '.format(Get_Average(ave_RF) * 100.0))
# print('SVM accuracy acc: {:.2f}% '.format(Get_Average(ave_SVM) * 100.0))
print('ENS accuracy acc: {:.2f}% '.format(Get_Average(ave_ENS) * 100.0))
# print('MLP accuracy acc: {:.2f}% '.format(Get_Average(ave_MLP) * 100.0))
print('BG accuracy acc: {:.2f}% '.format(Get_Average(ave_BG) * 100.0))
print('knn accuracy acc: {:.2f}% '.format(Get_Average(ave_KNN) * 100.0))
print('DT accuracy acc: {:.2f}% '.format(Get_Average(ave_DT) * 100.0))

print('\n')
print('RF accuracy f1: {:.4f} '.format(Get_Average(ave_f1_RF)))
# print('SVM accuracy f1: {:.4f} '.format(Get_Average(ave_f1_SVM)))
print('ENS accuracy f1: {:.4f} '.format(Get_Average(ave_f1_ENS)))
# print('MLP accuracy f1: {:.4f} '.format(Get_Average(ave_f1_MLP)))
print('BG accuracy f1: {:.4f} '.format(Get_Average(ave_f1_BG)))
print('knn accuracy f1: {:.4f} '.format(Get_Average(ave_f1_KNN)))
print('DT accuracy f1: {:.4f} '.format(Get_Average(ave_f1_DT)))
