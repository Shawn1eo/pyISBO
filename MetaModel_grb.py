import copy
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from gurobipy import *
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.optimize import differential_evolution
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.model_selection import train_test_split
from copy import deepcopy
from pyearth import Earth
import re


def find_parents(node_id, tree_):
    nodes_id = [node_id]
    while node_id != 0:
        for i in range(tree_.node_count):
            if tree_.children_left[i] == node_id or tree_.children_right[i] == node_id:
                nodes_id.append(i)
                node_id = i
    return nodes_id


'''
生成初始区间，这里采用每个变量最小值和最大值为区间的形式。也可以根据变量实际意义划定。
'''


def local_section_generator(input):
    max = input.max()
    min = input.min()
    return [min, max]


'''
model中output的value最大的对应区间,以及对应的value
local_section代表输入值的区间,形式为[[min,...,min],[max,...,max]]
'''


def max_section(model, local_section):
    tree_ = model.tree_
    final_section = []
    maxleaf = []
    maxvalue = 0
    leaves = []
    Output = []

    # 通过遍历的方法找到value最大的叶子节点
    for i in range(tree_.node_count):
        if tree_.threshold[i] == -2:
            if tree_.value[i] >= maxvalue - 0.01:
                leaves.append([i, tree_.value[i]])
                maxvalue = tree_.value[i]
    for i in range(len(leaves)):
        if leaves[i][1] >= maxvalue - 0.01:
            maxleaf.append(leaves[i][0])

    # 求出叶子节点的对应区间
    for j in maxleaf:
        section = local_section.copy()
        nodes = find_parents(j, tree_)
        for m in range(len(nodes) - 1):
            index = nodes[m]
            par = nodes[m + 1]
            feature = tree_.feature[par]
            threshold = tree_.threshold[par]
            if tree_.children_left[par] == index:  # 左子树 <=
                if section[1][feature] > threshold:
                    section[1][feature] = threshold
            else:  # 右子树 >=
                if section[0][feature] < threshold:
                    section[0][feature] = threshold
        final_section.append(section.copy())
    for i in final_section:
        for j in range(len(i[0])):
            Output.append(random.uniform(i[0][j],i[1][j]))
    return maxvalue,Output


class DT:
    model = None
    optimizedParameter = None
    MIP = None
    errorMode = 0
    bounds = []
    types = None
    output = None
    mean = 0
    std = 0
    local_section = None

    def __init__(self, errorMode, parameterInfo):
        self.errorMode = errorMode
        for i in range(parameterInfo.shape[0]):
            self.bounds.append((parameterInfo["lb"][i], parameterInfo["ub"][i]))
        self.types = list(parameterInfo.pop("type"))

    def fit(self, X, y):
        model = DecisionTreeRegressor(max_depth=8, min_samples_leaf=5)  # 实例化
        xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)  # 0.8 0.2 划分
        model.fit(xtrain, ytrain)
        ypred = model.predict(xtest)  # 获取模型在验证集上的预测结果
        # 计算RMSE
        rmse = np.sqrt(mean_squared_error(ytest, ypred))
        # 计算MAPE
        mape = mean_absolute_percentage_error(ytest, ypred)
        # 计算MAE
        mpe = mean_absolute_error(ytest, ypred)
        error = 0
        # score mode: 1 RMSE 2 MAPE 3 MAE
        if self.errorMode == 1:
            error = rmse
        elif self.errorMode == 2:
            error = mape
        elif self.errorMode == 3:
            error = mpe
        else:
            print("input error:wrong error mode")

        self.model = model
        self.local_section = local_section_generator(X)
        return error

    def predict(self, X):
        return self.model.predict(X)

    def optimize(self, Goal):
        self.optimizedParameter, self.output = max_section(self.model, self.local_section)



class NN:
    model = None
    optimizedParameter = None
    MIP = None
    errorMode = 0
    bounds = []
    types = None
    output = None
    mean = 0
    std = 0

    def __init__(self, errorMode, parameterInfo):
        self.errorMode = errorMode
        for i in range(parameterInfo.shape[0]):
            self.bounds.append((parameterInfo["lb"][i], parameterInfo["ub"][i]))
        self.types = list(parameterInfo.pop("type"))
        self.types = [GRB.CONTINUOUS if self.types[i] == "Continuous" else GRB.INTEGER for i in range(len(self.types))]

    def norm(self, x):
        return (x - self.mean) / self.std

    def fit(self, input: pd.DataFrame, output: pd.DataFrame):
        '''
        此函数用于进行神经网络训练
        :param input: 输入参数，为DataFrame格式
        :param output:输出结果，为DataFrame格式
        :return: 一个keras神经网络模型，同时输出输入参数的均值和标准差
        '''
        train_dataset = input.sample(frac=0.8, random_state=0)
        test_dataset = input.drop(train_dataset.index)
        input_dim = train_dataset.shape[1]
        test_labels = output.drop(train_dataset.index)
        train_labels = output.drop(test_labels.index)

        train_stats = train_dataset.describe()
        train_stats = train_stats.transpose()

        self.mean = train_stats['mean']
        self.std = train_stats['std']

        normed_train_data = self.norm(train_dataset)
        normed_test_data = self.norm(test_dataset)

        Models = []
        RMSE = []
        MAPE = []
        MAE = []
        for K in range(2, 10):
            print("K=%d" % K)
            model = keras.Sequential()

            model.add(layers.Dense(input_dim, activation='relu', input_shape=[len(train_dataset.keys())]))
            for i in range(1, K - 1):
                model.add(layers.Dense(input_dim, activation='relu'))
            model.add(layers.Dense(1))

            optimizer = tf.keras.optimizers.RMSprop(0.001)
            if self.errorMode == 1:
                lossMode = "mse"
            elif self.errorMode == 2:
                lossMode = "mape"
            else:
                lossMode = "mae"

            model.compile(loss=lossMode, optimizer=optimizer, metrics=['mae', 'mse'])

            train_history = model.fit(x=normed_train_data, y=train_labels, validation_split=0.1, epochs=200)
            predict_y = model.predict(normed_test_data)
            RMSE.append(np.sqrt(mean_squared_error(predict_y, test_labels)))
            MAPE.append(mean_absolute_percentage_error(predict_y, test_labels))
            MAE.append(mean_absolute_error(predict_y, test_labels))
            Models.append(model)

        error = 0
        if self.errorMode == 1:
            model_index = RMSE.index(min(RMSE))
            error = RMSE[model_index]
        elif self.errorMode == 2:
            model_index = MAPE.index(min(MAPE))
            error = MAPE[model_index]
        else:
            model_index = MAE.index(min(MAE))
            error = MAE[model_index]
        self.model = Models[model_index]
        return error

    def predict(self, X):
        return self.model.predict(self.norm(X))

    def MIP_transform(self):
        nk = []
        K = len(self.model.get_config()['layers']) - 1
        input_dim = len(self.types)
        for k in range(1, K + 1):
            nk.append(self.model.get_config()['layers'][k]['config']['units'])
        weights = self.model.get_weights()
        w = []
        b = []
        for k in range(K):
            w.append(weights[2 * k])
            b.append(weights[2 * k + 1])

        inputInfo = {}
        for i in range(len(self.types)):
            inputInfo[(i)] = [self.bounds[i][0], self.bounds[i][1], self.types[i]]
        inp, lb, ub, vtype = multidict(inputInfo)
        self.MIP = Model("Neural Network")
        neuron_index = []
        for i in range(len(inp)):
            neuron_index.append((0, i))
        for k in range(K):
            for i in range(nk[k]):
                neuron_index.append((k + 1, i))

        x = self.MIP.addVars(inp, lb=lb, ub=ub, vtype=vtype, name="x")
        neuron = self.MIP.addVars(neuron_index, lb=0, vtype=GRB.CONTINUOUS, name="neuron")
        s_ki = self.MIP.addVars(neuron_index, lb=0, vtype=GRB.CONTINUOUS, name="s_ki")
        z_ki = self.MIP.addVars(neuron_index, lb=0, vtype=GRB.BINARY, name="z_ki")
        y = self.MIP.addVar(lb=0, vtype=GRB.CONTINUOUS, name="y")
        self.MIP.update()

        M = 1e5

        self.MIP.setObjective(y, GRB.MINIMIZE)

        firstLayer = self.MIP.addConstrs(
            (x[i] - self.mean[i] == self.std[i] * neuron[0, i] for i in range(input_dim)),
            name="firstLayer")

        neuronTransmit = self.MIP.addConstrs(
            (quicksum(neuron[k - 1, i] * w[k - 1][i, j] for i in range(w[k - 1].shape[0]))
             + b[k - 1][j] - neuron[k, j] + s_ki[k, j] == 0 for k in range(1, K + 1) for j
             in range(nk[k - 1])), name="neuronTransmit")
        ReLUx = self.MIP.addConstrs(
            (z_ki[k, i] * M + neuron[k, i] <= M for k in range(1, K + 1) for i in range(nk[k - 1])),
            name="ReLUx")
        ReLUs = self.MIP.addConstrs(
            (z_ki[k, i] * M - s_ki[k, i] >= 0 for k in range(1, K + 1) for i in range(nk[k - 1])),
            name="ReLUs")
        objective = self.MIP.addConstr((y == neuron[K, 0]), name="objective")
        self.MIP.update()

    def optimize(self, Goal):
        Goal = GRB.MAXIMIZE if Goal == "MAXIMIZE" else GRB.MINIMIZE
        self.MIP.optimize()
        self.output = [self.MIP.getVarByName("x[%d]" % i).X for i in range(len(self.types))]


class RSM:
    model = None
    optimizedParameter = None
    errorMode = 0
    degree = 0
    bounds = []
    output = None
    types = None
    MIP = None

    def __init__(self, errorMode, parameterInfo):
        self.errorMode = errorMode
        for i in range(parameterInfo.shape[0]):
            self.bounds.append((parameterInfo["lb"][i], parameterInfo["ub"][i]))

    def fit(self, X, y):
        train_dataset = X.sample(frac=0.8, random_state=0)
        test_dataset = X.drop(train_dataset.index)
        test_labels = y.drop(train_dataset.index)
        train_labels = y.drop(test_labels.index)

        Models = []
        RMSE = []
        MAPE = []
        MAE = []
        for i in range(3):
            polyX = train_dataset if i == 0 else PolynomialFeatures(degree=i+1).fit_transform(train_dataset)
            model = LinearRegression()
            model.fit(polyX, train_labels)
            Models.append(model)
            polyX_test = test_dataset if i == 0 else PolynomialFeatures(degree=i+1).fit_transform(test_dataset)
            predict_y = model.intercept_ + np.dot(polyX_test, model.coef_.T)
            RMSE.append(np.sqrt(mean_squared_error(predict_y, test_labels)))
            MAPE.append(mean_absolute_percentage_error(predict_y, test_labels))
            MAE.append(mean_absolute_error(predict_y, test_labels))
        error = 0
        if self.errorMode == 1:
            model_index = RMSE.index(min(RMSE))
            error = RMSE[model_index]
        elif self.errorMode == 2:
            model_index = MAPE.index(min(MAPE))
            error = MAPE[model_index]
        else:
            model_index = MAE.index(min(MAE))
            error = MAE[model_index]
        self.model = Models[model_index]
        self.degree = model_index + 1
        return error

    def predict(self, X):
        if self.degree >= 2:
            X_poly = PolynomialFeatures(degree=self.degree).fit_transform(np.array([X]))
        else:
            X_poly = X
        return float(self.model.intercept_ + np.dot(X_poly, self.model.coef_.T))

    def __predict_for_optimize__(self, X, mode):
        if mode == "MINIMIZE":
            return self.predict(X)
        else:
            return -self.predict(X)

    def optimize(self, mode):
        if mode == "MINIMIZE":
            result = differential_evolution(self.__predict_for_optimize__, bounds=self.bounds, args=(mode,))
            self.optimizedParameter = result.x
            self.output = result.fun
        else:
            result = differential_evolution(self.__predict_for_optimize__, bounds=self.bounds, args=(mode,))
            self.optimizedParameter = result.x
            self.output = result.fun
        return self.output


class MARS:
    model = None
    optimizedParameter = None
    errorMode = 0
    degree = 0
    bounds = []
    output = None
    MIP = None
    types = None

    def __init__(self, errorMode, parameterInfo):
        self.errorMode = errorMode
        for i in range(parameterInfo.shape[0]):
            self.bounds.append((parameterInfo["lb"][i], parameterInfo["ub"][i]))
        self.types = list(parameterInfo.pop("type"))
        self.types = [GRB.CONTINUOUS if self.types[i] == "Continuous" else GRB.INTEGER for i in range(len(self.types))]

    def fit(self, X, y):
        train_dataset = X.sample(frac=0.8, random_state = 1)
        test_dataset = X.drop(train_dataset.index)
        test_labels = y.drop(train_dataset.index)
        train_labels = y.drop(test_labels.index)
        self.optimizedParameter = test_dataset.drop(index=test_dataset.index)

        self.model = Earth()
        self.model.fit(train_dataset, train_labels)

        predicted_y = self.model.predict(test_dataset)
        RMSE = np.sqrt(mean_squared_error(predicted_y, test_labels))
        MAPE = mean_absolute_percentage_error(predicted_y, test_labels)
        MAE = mean_absolute_error(predicted_y, test_labels)
        if self.errorMode == 1:
            return RMSE
        elif self.errorMode == 2:
            return MAPE
        else:
            return MAE

    def predict(self, X):
        return self.model.predict(X)

    def MIP_transform(self):
        summary = self.model.summary()
        summary = summary.split("\n")[5:]
        summary = summary[:len(summary) - 2]
        for i in range(len(summary)):
            summary[i] = summary[i].split(" ")
            while "" in summary[i]:
                summary[i].remove("")
        for i in range(len(summary)):
            if summary[i][1] == "Yes":
                summary[i] = ""
        while "" in summary:
            summary.remove("")
        branch_index = [i for i in range(len(summary))]
        intercept = self.model.coef_[0][0]

        inputInfo = {}
        for i in range(len(self.types)):
            inputInfo[(i)] = [self.bounds[i][0], self.bounds[i][1], self.types[i]]
        inp, lb, ub, vtype = multidict(inputInfo)
        self.MIP = Model("Multivariate Adaptive Regression Splines")

        x = self.MIP.addVars(inp, lb=lb, ub=ub, vtype=vtype, name="x")
        branch = self.MIP.addVars(branch_index, lb=-1e5 ,vtype=GRB.CONTINUOUS, name="branch")
        y = self.MIP.addVar(vtype=GRB.CONTINUOUS, lb=-1e5, name="y")
        s = self.MIP.addVars(branch_index, lb=0, vtype=GRB.CONTINUOUS, name="s")
        z = self.MIP.addVars(branch_index, lb=0, vtype=GRB.BINARY, name="z")
        self.MIP.update()

        M=1e5
        self.MIP.addConstr((y == intercept + quicksum(branch[i] for i in range(len(summary)))), name="objective")
        for col in range(len(self.optimizedParameter.columns)):
            colname = self.optimizedParameter.columns[col]
            for i in range(len(summary)):
                coef = float(summary[i][2])
                if summary[i][0] == colname:  # 没有分割点
                    self.MIP.addConstr((branch[i] == coef*x[col]), name="branch[%d]"%i)
                if summary[i][0][2:2+len(colname)] == colname:
                    point = summary[i][0][3+len(colname):len(summary[i][0])-1]
                    try:
                        point = float(point)
                        self.MIP.addConstr((x[col]-point-branch[i]+s[i] == 0), name="ReLU1[%d]"%i)
                        self.MIP.addConstr((branch[i] <= M*(1-z[i])), name="ReLU2[%d]"%i)
                        self.MIP.addConstr((s[i] <= M*z[i]), name="ReLU3[%d]"%i)
                    except ValueError:
                        pass

                if summary[i][0][len(summary[i][0])-1-len(colname):len(summary[i][0])-1] == colname:
                    point = summary[i][0][2:len(summary[i][0])-2-len(colname)]
                    try:
                        point = float(point)
                        self.MIP.addConstr((point - x[col] - branch[i] + s[i] == 0), name="ReLU1[%d]"%i)
                        self.MIP.addConstr((branch[i] <= M * (1 - z[i])), name="ReLU2[%d]"%i)
                        self.MIP.addConstr((s[i] <= M * z[i]), name="ReLU3[%d]"%i)
                    except ValueError:
                        pass
        self.MIP.update()

    def optimize(self, mode=GRB.MINIMIZE, objective=None):
        if objective is None:
            y = self.MIP.getVarByName("y")
            self.MIP.setObjective(y, mode)
        else:
            self.MIP.setObjective(objective, mode)
        self.MIP.optimize()
        self.output = [self.MIP.getVarByName("x[%d]" % i).X for i in range(len(self.types))]


class AutoRegression:
    model = None
    optimizedParameter = None
    errorMode = 0
    output = None
    parameterInfo = None
    MIP = None

    def __init__(self, errorMode, parameterInfo):
        self.errorMode = errorMode
        self.parameterInfo = parameterInfo

    def fit(self, X, y):
        RSM_Model = RSM(self.errorMode, deepcopy(self.parameterInfo))
        NN_Model = NN(self.errorMode, deepcopy(self.parameterInfo))
        DT_Model = DT(self.errorMode, deepcopy(self.parameterInfo))
        MARS_Model = MARS(self.errorMode, deepcopy(self.parameterInfo))
        error = [RSM_Model.fit(X, y), NN_Model.fit(X, y), DT_Model.fit(X, y)]
        if error.index(min(error)) == 0:
            self.model = RSM_Model
        elif error.index(min(error)) == 1:
            self.model = NN_Model
        elif error.index(min(error)) == 2:
            self.model = DT_Model
        else:
            self.model = MARS_Model
        print("模型误差为："+str(min(error)))
        return min(error)

    def predict(self, X):
        return self.model.predict(X)

    def MIP_transform(self):
        self.model.MIP_transform()

    def optimize(self, mode=GRB.MINIMIZE, objective=None):
        if objective is None:
            y = self.model.MIP.getVarByName("y")
            self.model.MIP.setObjective(y, mode)
        else:
            self.model.MIP.setObjective(objective, mode)
        self.model.MIP.optimize()
        self.output = [self.model.MIP.getVarByName("x[%d]" % i).X for i in range(len(self.model.types))]


data = pd.read_excel("Example.xlsx")
ParameterInfo = pd.read_excel("Example.xlsx", 1)
Output = data.pop("y")
# model = AutoRegression(2, ParameterInfo)
# model.fit(data, Output)
# model.optimize("MAXIMIZE")

m = MARS(2, ParameterInfo)
m.fit(data, Output)
m.MIP_transform()
x = [m.MIP.getVarByName("x[%d]" % i)for i in range(ParameterInfo.shape[0])]
y = m.MIP.getVarByName("y")
# m.optimize()
# print(m.trace())
# print(model.optimizedParameter)


