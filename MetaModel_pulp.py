import copy
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from pulp import *
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import differential_evolution
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.model_selection import train_test_split
from copy import deepcopy
from pyearth import Earth


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


def local_section_generator(bounds):
    max = []
    min = []
    for i in range(len(bounds)):
        max.append(bounds[i][1])
        min.append(bounds[i][0])
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


def find_leaves(node_id,tree_):
    if tree_.children_left[node_id] == -1:
        return [node_id]
    else:
        return find_leaves(tree_.children_left[node_id], tree_) + find_leaves(tree_.children_right[node_id], tree_)


class DT:
    model = None
    optimizedParameter = None
    errorMode = 0
    bounds = []
    types = []
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
        self.local_section = local_section_generator(self.bounds)
        return error

    def predict(self, X):
        return self.model.predict(X)

    def optimize(self, Goal):
        self.optimizedParameter, self.output = max_section(self.model, self.local_section)


class RF:
    model = None
    optimizedParameter = None
    errorMode = 0
    bounds = []
    types = []
    output = None
    mean = 0
    std = 0

    def __init__(self, errorMode, parameterInfo):
        self.errorMode = errorMode
        for i in range(parameterInfo.shape[0]):
            self.bounds.append((parameterInfo["lb"][i], parameterInfo["ub"][i]))
        self.types = list(parameterInfo.pop("type"))

    def autofit(self, X, y):
        model = RandomForestRegressor(max_depth=8, min_samples_leaf=5)
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

        return error

    def predict(self, X):
        return self.model.predict(X)

    def optimize(self, Goal):
        Goal = LpMaximize if Goal == "MAXIMIZE" else LpMinimize

        model = LpProblem("random_forest", Goal)
        estimators_ = self.model.estimators_

        # 添加变量：每个叶子节点对应零一变量
        leaves_num = []
        for i in range(len(estimators_)):
            leaves_num.append(find_leaves(0, estimators_[i].tree_))

        x_len = len(self.types)
        local_section = local_section_generator(self.bounds)
        x_variable = {}
        for i in range(x_len):
            x_variable[i] = LpVariable('x_{}'.format(i), lowBound=local_section[0][i], upBound=local_section[1][i],
                                          cat=self.types[i])

        y_variable = {}
        for j in range(len(estimators_)):
            y_variable[j] = LpVariable.dict('y_{}'.format(j), range(len(leaves_num[j])), cat=LpBinary)

        # 添加一个字典，效果是通过叶子结点的索引找到对应决策变量。
        y_dict_list = []
        for j in range(len(estimators_)):
            y_dict = {}
            for i in range(len(leaves_num[j])):
                y_dict[leaves_num[j][i]] = y_variable[j][i]
            y_dict_list.append(y_dict.copy())

        # 添加目标函数
        objective = 0
        for j in range(len(estimators_)):
            objective += lpSum(
                y_variable[j][i] * estimators_[j].tree_.value[leaves_num[j][i]] for i in range(len(leaves_num[j])))
        model += objective

        # 添加约束：每个分支节点对应两个约束；所有叶子节点只有一个能被选中。
        M = 10000  # 注意！由于pulp不允许高精度的大数字，float(inf)是不被允许的。
        for m in range(len(estimators_)):
            for i in range(estimators_[m].tree_.node_count):
                if estimators_[m].tree_.children_left[i] != -1:
                    cons_leaves_1 = find_leaves(estimators_[m].tree_.children_left[i], estimators_[m].tree_)
                    model += M * (lpSum(y_dict_list[m][j] for j in cons_leaves_1) - 1) - (
                                estimators_[m].tree_.threshold[i] - x_variable[estimators_[m].tree_.feature[i]]) <= 0
                    cons_leaves_2 = find_leaves(estimators_[m].tree_.children_right[i], estimators_[m].tree_)
                    model += M * (lpSum(y_dict_list[m][j] for j in cons_leaves_2) - 1) + (
                                estimators_[m].tree_.threshold[i] - x_variable[
                            estimators_[m].tree_.feature[i]]) + 1 / M <= 0
            model += lpSum(y_variable[m][i] for i in range(len(y_variable[m]))) == 1
        model.solve()
        solved_variables = {}
        for v in model.variables():
            solved_variables[v.name] = v.varValue
        solved_variables_dict = {}
        for i in range(x_len):
            try:
                solved_variables_dict['x_{}'.format(i)] = solved_variables['x_{}'.format(i)]
            except:
                solved_variables_dict['x_{}'.format(i)] = random.uniform(local_section[0][i], local_section[1][i])
        '''
        for j in range(len(estimators_)):
            for k in range(len(leaves_num[j])):
                solved_variables_dict["y_{}_{}".format(j,k)] = solved_variables["y_{}_{}".format(j,k)]
        '''
        optimizedParameter = []
        for i in solved_variables_dict:
            optimizedParameter.append(solved_variables_dict[i])
        # 随机森林的目标函数为所有树的value求和，return时求平均以保持与y一致。
        self.output = value(model.objective) / len(estimators_)
        self.optimizedParameter = optimizedParameter


class NN:
    model = None
    optimizedParameter = None
    errorMode = 0
    bounds = []
    types = []
    output = None
    mean = 0
    std = 0

    def __init__(self, errorMode, parameterInfo):
        self.errorMode = errorMode
        for i in range(parameterInfo.shape[0]):
            self.bounds.append((parameterInfo["lb"][i], parameterInfo["ub"][i]))
        self.types = list(parameterInfo.pop("type"))
        self.types = [LpContinuous if self.types[i] == "Continuous" else LpInteger for i in range(len(self.types))]

    def norm(self, x):
        return (x - self.mean) / self.std

    def fit(self, input: pd.DataFrame, output: pd.DataFrame, layer=None):
        '''
        此函数用于进行神经网络训练
        :param input: 输入参数，为DataFrame格式
        :param output:输出结果，为DataFrame格式
        :param layer: 层数，为整数，如果不输入代表自动拟合
        :return: 一个keras神经网络模型，同时输出输入参数的均值和标准差
        '''
        train_dataset = input.sample(frac=0.8)
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

        if layer is None:
            Models = []
            RMSE = []
            MAPE = []
            MAE = []
            for K in range(2, 6):
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
            print("The best Neural Network has "+str(model_index+2)+" layers")
            return error

        else:
            model = keras.Sequential()
            model.add(layers.Dense(input_dim, activation='relu', input_shape=[len(train_dataset.keys())]))
            for i in range(1, layer - 1):
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
            self.model = model
            predict_y = model.predict(normed_test_data)
            RMSE = np.sqrt(mean_squared_error(predict_y, test_labels))
            MAPE = mean_absolute_percentage_error(predict_y, test_labels)
            MAE = mean_absolute_error(predict_y, test_labels)
            if self.errorMode == 1:
                return RMSE
            elif self.errorMode == 2:
                return MAPE
            else:
                return MAE

    def predict(self, X):
        return self.model.predict(self.norm(X))

    def optimize(self, Goal):
        Goal = LpMaximize if Goal == "MAXIMIZE" else LpMinimize
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

        lpmodel = LpProblem("神经网络优化", Goal)
        # 建立变量
        neuron_index = []
        for i in range(input_dim):
            neuron_index.append((0, i))
        for k in range(K):
            for i in range(nk[k]):
                neuron_index.append((k + 1, i))

        x_input = {}
        for i in range(len(self.types)):
            x_input[i] = LpVariable("x_input_%d" % i, self.bounds[i][0], self.bounds[i][1], cat=self.types[i])

        neuron = {}
        for t in neuron_index:
            neuron[t] = LpVariable("neuron_%d_%d" % t, 0, cat=LpContinuous)

        s_ki = {}
        for t in neuron_index:
            s_ki[t] = LpVariable("s_ki_%d_%d" % t, 0, cat=LpContinuous)

        z_ki = {}
        for t in neuron_index:
            z_ki[t] = LpVariable("z_ki_%d_%d" % t, 0, cat=LpBinary)

        M = 1e5

        # 目标函数
        lpmodel += neuron[(K, 0)], "输出结果"

        # 设置约束
        # 约束1：标准化还原
        for i in range(input_dim):
            lpmodel += x_input[i] - self.mean[i] == self.std[i] * neuron[0, i], "firstLayer_" + str(i)
        # 约束2：值在神经元间传递
        for k in range(1, K + 1):
            for j in range(nk[k - 1]):
                lpmodel += lpSum(neuron[k - 1, i] * w[k - 1][i, j] for i in range(w[k - 1].shape[0])) \
                           + b[k - 1][j] - neuron[k, j] + s_ki[k, j] == 0, "neuronTransmit_" + str(k) + "_" + str(j)
        # 约束3：ReLUx
        for k in range(1, K + 1):
            for i in range(nk[k - 1]):
                lpmodel += z_ki[k, i] * M + neuron[k, i] <= M, "ReLUx" + str(k) + "_" + str(i)
        # 约束4：ReLUs
        for k in range(1, K + 1):
            for i in range(nk[k - 1]):
                lpmodel += z_ki[k, i] * M - s_ki[k, i] >= 0, "ReLUs" + str(k) + "_" + str(i)

        lpmodel.solve()
        result = []
        for v in lpmodel.variables():
            if "x_input" in v.name:
                result.append(v.varValue)
        self.optimizedParameter = result
        self.output = value(lpmodel.objective)


class RSM:
    model = None
    optimizedParameter = None
    errorMode = 0
    degree = 0
    bounds = []
    output = None

    def __init__(self, errorMode, parameterInfo):
        self.errorMode = errorMode
        for i in range(parameterInfo.shape[0]):
            self.bounds.append((parameterInfo["lb"][i], parameterInfo["ub"][i]))

    def fit(self, X, y, degree = None):
        train_dataset = X.sample(frac=0.8, random_state=0)
        test_dataset = X.drop(train_dataset.index)
        test_labels = y.drop(train_dataset.index)
        train_labels = y.drop(test_labels.index)

        if degree is None:
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

        else:
            self.degree = degree
            polyX = train_dataset if degree == 1 else PolynomialFeatures(degree=degree + 1).fit_transform(train_dataset)
            self.model = LinearRegression()
            self.model.fit(polyX, train_labels)
            polyX_test = test_dataset if degree == 0 else PolynomialFeatures(degree=degree + 1).fit_transform(test_dataset)
            predict_y = self.model.intercept_ + np.dot(polyX_test, self.model.coef_.T)
            if self.errorMode == 1:
                return np.sqrt(mean_squared_error(predict_y, test_labels))
            elif self.errorMode == 2:
                return mean_absolute_percentage_error(predict_y, test_labels)
            else:
                return mean_absolute_error(predict_y, test_labels)

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
            self.output = -result.fun
        return self.output


class MARS:
    model = None
    optimizedParameter = None
    errorMode = 0
    degree = 0
    bounds = []
    output = None

    def __init__(self, errorMode, parameterInfo):
        self.errorMode = errorMode
        for i in range(parameterInfo.shape[0]):
            self.bounds.append((parameterInfo["lb"][i], parameterInfo["ub"][i]))

    def fit(self, X, y):
        train_dataset = X.sample(frac=0.8, random_state = 2)
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

    def optimize(self, mode):
        summary = self.model.summary()
        print(summary)
        summary = summary.split("\n")[5:]
        summary = summary[:len(summary)-2]
        for i in range(len(summary)):
            summary[i] = summary[i].split(" ")
            while "" in summary[i]:
                summary[i].remove("")
        self.optimizedParameter = pd.DataFrame([[None for i in range(len(self.optimizedParameter.columns))]],
                                               columns=self.optimizedParameter.columns)
        for col in range(len(self.optimizedParameter.columns)):
            colflag = 0
            colname = self.optimizedParameter.columns[col]
            dataframeCopy = copy.deepcopy(self.optimizedParameter)
            breakpoint = []
            for branch in summary:
                if branch[1] == "No":
                    if branch[0] == colname:  # 没有分割点
                        colflag = 1
                        breakpoint.append(self.bounds[col][0])
                        breakpoint.append(self.bounds[col][1])
                    if branch[0][2:2+len(colname)] == colname:
                        colflag = 1
                        point = branch[0][3+len(colname):len(branch[0])-1]
                        try:
                            point = float(point)
                            if point not in breakpoint:
                                breakpoint.append(point)
                            if self.bounds[col][0] not in breakpoint:
                                breakpoint.append(self.bounds[col][0])
                                breakpoint.append(self.bounds[col][1])
                        except ValueError:
                            pass

                    if branch[0][len(branch[0])-1-len(colname):len(branch[0])-1] == colname:
                        colflag = 1
                        point = branch[0][2:len(branch[0])-2-len(colname)]
                        try:
                            point = float(point)
                            if point not in breakpoint:
                                breakpoint.append(point)
                            if self.bounds[col][0] not in breakpoint:
                                breakpoint.append(self.bounds[col][0])
                                breakpoint.append(self.bounds[col][1])
                        except ValueError:
                            pass
            if colflag == 1:
                colframe = pd.DataFrame()
                for point in breakpoint:
                    pointCopy = copy.deepcopy(dataframeCopy)
                    for j in range(pointCopy.shape[0]):
                        pointCopy.iloc[j, col] = point
                    colframe = pd.concat([colframe, pointCopy])
                self.optimizedParameter = colframe
        for i in range(self.optimizedParameter.shape[0]):
            for j in range(self.optimizedParameter.shape[1]):
                if self.optimizedParameter.iloc[i, j] is None:
                    self.optimizedParameter.iloc[i, j] = self.bounds[j][0] \
                                                         + np.random.rand()*(self.bounds[j][1]-self.bounds[j][0])
        self.output = self.model.predict(self.optimizedParameter)
        if mode == "MINIMIZE":
            self.optimizedParameter = self.optimizedParameter.iloc[list(self.output).index(min(self.output)), :]
            self.output = min(self.output)
        else:
            self.optimizedParameter = self.optimizedParameter.iloc[list(self.output).index(max(self.output)), :]
            self.output = max(self.output)
        print(self.optimizedParameter)


class AutoRegression:
    model = None
    optimizedParameter = None
    errorMode = 0
    output = None
    parameterInfo = None

    def __init__(self, errorMode, parameterInfo):
        self.errorMode = errorMode
        self.parameterInfo = parameterInfo

    def fit(self, X, y):
        RSM_Model = RSM(self.errorMode, deepcopy(self.parameterInfo))
        NN_Model = NN(self.errorMode, deepcopy(self.parameterInfo))
        DT_Model = DT(self.errorMode, deepcopy(self.parameterInfo))
        MARS_Model = MARS(self.errorMode, deepcopy(self.parameterInfo))
        RF_Model = RF(self.errorMode, deepcopy(self.parameterInfo))
        error = [RSM_Model.fit(X, y), NN_Model.fit(X, y), DT_Model.fit(X, y), MARS_Model.fit(X, y), RF_Model.fit(X, y)]
        if error.index(min(error)) == 0:
            self.model = RSM_Model
        elif error.index(min(error)) == 1:
            self.model = NN_Model
        elif error.index(min(error)) == 2:
            self.model = DT_Model
        elif error.index(min(error)) == 3:
            self.model = MARS_Model
        else:
            self.model = RF_Model
        print("模型误差为："+str(min(error)))
        return min(error)

    def predict(self, X):
        return self.model.predict(X)

    def optimize(self, Goal):
        self.model.optimize(Goal)
        self.optimizedParameter = self.model.optimizedParameter
        self.output = self.model.output

# data = pd.read_excel("Example.xlsx")
# ParameterInfo = pd.read_excel("Example.xlsx", 1)
# Output = data.pop("y")
# # model = AutoRegression(2, ParameterInfo)
# # model.fit(data, Output)
# # model.optimize("MAXIMIZE")
#
# # m = Earth()
# # m.fit(data, Output)
# # print(m.trace())
# # print(m.summary())
# # print(model.optimizedParameter)
#
# model = RF(1,ParameterInfo)
# model.fit(data,Output)
# model.optimize("MAXIMIZE")

