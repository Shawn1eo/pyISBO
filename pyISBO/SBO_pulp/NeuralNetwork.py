from sklearn.metrics import get_scorer
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold
from tensorflow.keras import layers
from pulp import *



class NN:
    """
    Neural Network. Based on tensorflow2.0.
    """
    model = None
    optimizedParameter = None
    scoring = None
    bounds = []
    output = None
    types = None
    MIP = None
    mean = None
    std = None

    def __init__(self, parameterInfo, scoring="neg_mean_squared_error"):
        """
        Initialize a Linear surrogate.
        :param parameterInfo: A Pandas Dataframe. Default = None
            A dataframe containing information of your input variables. It should contain four columns: Name, lb, ub
            and types, which correspond to the names, lower bounds, upper bounds and types of your input variables.
            You can find an example by checking "example.xlsx" in https://github.com/Shawn1eo/pyISBO.
        :param scoring: A string or callable object. Default = "neg_mean_squared_error"
            You can name a specific scoring metric for the surrogate. Use sorted(sklearn.metrics.SCORERS.keys()) to
            get valid options.
        """
        self.scoring = scoring
        for i in range(parameterInfo.shape[0]):
            self.bounds.append((parameterInfo["lb"][i], parameterInfo["ub"][i]))
        self.types = list(parameterInfo.pop("type"))
        self.types = [LpContinuous if self.types[i] == "Continuous" else LpInteger for i in range(len(self.types))]

    def norm(self, x):
        return (x - self.mean) / self.std

    def __create_model(self, k):
        n = len(self.types)
        model = keras.Sequential()
        model.add(layers.Dense(n, activation='relu', input_shape=[n]))
        for i in range(1, k - 1):
            model.add(layers.Dense(n, activation='relu'))
        model.add(layers.Dense(1))
        optimizer = tf.keras.optimizers.RMSprop(0.001)
        model.compile(loss="mse", optimizer=optimizer, metrics=['mae', 'mse'])
        return model

    def fit(self, X, y):
        """
        Fit linear model.
        :param X:{array-like, sparse matrix} of shape (n_samples, n_features)T
            Training data.
        :param y:array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary.
        :return: A float number.
            The cross-validation score of the fitted model based on the scoring metric you choose.
        """
        print("Now fitting neural network.")
        stats = X.describe().transpose()
        self.mean = stats['mean']
        self.std = stats['std']
        normed_X = self.norm(X)
        scorer = get_scorer(self.scoring)
        Scores = []
        for K in range(2, 7):
            score = 0
            kfold = KFold(5, shuffle=True)
            for train_index, test_index in kfold.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                model = self.__create_model(K)
                model.fit(X_train, y_train, epochs=100, verbose=None)
                score += scorer(model, X_test, y_test)
            Scores.append(score/5)

        k = Scores.index(max(Scores)) + 2
        self.model = self.__create_model(k)
        self.model.fit(normed_X, y)
        return max(Scores)

    def predict(self, X):
        """
        Predict using the surrogate.
        :param X:{array-like, sparse matrix} of shape (n_samples, n_features)T
            Training data.
        :return:An array, shape (n_samples,)
            Predicted values.
        """
        assert self.model is not None, "You haven't build a surrogate yet. Try using fit() to create one."
        return self.model.predict(X)

    def MIP_transform(self):
        """
        Transform the surrogate into a pulp linear program
        :return: None.
            You can access the transformed linear model by MIP object.
        """
        assert self.model is not None, "You haven't build a surrogate yet. Try using fit() to create one."

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

        lpmodel = LpProblem("NeuralNetwork", LpMinimize)
        neuron_index = []
        for i in range(input_dim):
            neuron_index.append((0, i))
        for k in range(K):
            for i in range(nk[k]):
                neuron_index.append((k + 1, i))

        x = {}
        for i in range(len(self.types)):
            x[i] = LpVariable("x_%d" % i, self.bounds[i][0], self.bounds[i][1], cat=self.types[i])

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

        lpmodel += neuron[(K, 0)], "objective"

        for i in range(input_dim):
            lpmodel += x[i] - self.mean[i] == self.std[i] * neuron[0, i], "firstLayer_" + str(i)
        for k in range(1, K + 1):
            for j in range(nk[k - 1]):
                lpmodel += lpSum(neuron[k - 1, i] * w[k - 1][i, j] for i in range(w[k - 1].shape[0])) \
                           + b[k - 1][j] - neuron[k, j] + s_ki[k, j] == 0, "neuronTransmit_" + str(k) + "_" + str(j)
        for k in range(1, K + 1):
            for i in range(nk[k - 1]):
                lpmodel += z_ki[k, i] * M + neuron[k, i] <= M, "ReLUx" + str(k) + "_" + str(i)
        for k in range(1, K + 1):
            for i in range(nk[k - 1]):
                lpmodel += z_ki[k, i] * M - s_ki[k, i] >= 0, "ReLUs" + str(k) + "_" + str(i)

        self.MIP = lpmodel

    def optimize(self):
        """
        Optimize over the MIP
        :return: None.
            You can get the optimized value and the optimized parameters by "output" and  "optimizedParameter" object.
        """
        if self.MIP is None:
            self.MIP_transform()
        self.MIP.solve()
        self.optimizedParameter = [self.MIP.variablesDict()["x_%d" % i].varValue for i in range(len(self.types))]
        self.output = value(self.MIP.objective)
