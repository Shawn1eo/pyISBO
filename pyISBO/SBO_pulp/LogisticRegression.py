import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from pulp import *


class LogR:
    """
    Logistic Surrogate. Based on sklearn.linear_model.LogisticRegression.
    """
    model = None
    optimizedParameter = None
    scoring = None
    bounds = []
    output = None
    types = None
    MIP = None

    def __init__(self, parameterInfo, scoring="Accuracy"):
        """
        Initialize a Logistic surrogate.
        :param parameterInfo: A Pandas Dataframe. Default = None
            A dataframe containing information of your input variables. It should contain four columns: parameterName,
            lb, ub and types, which correspond to the names, lower bounds, upper bounds and types of your input
            variables. You can find an example by checking "example.xlsx" in https://github.com/Shawn1eo/pyISBO.
        :param scoring: A string or callable object. Default = "Accuracy"
            You can name a specific scoring metric for the surrogate. Use sorted(sklearn.metrics.SCORERS.keys()) to
            get valid options.
        """
        self.scoring = scoring
        for i in range(parameterInfo.shape[0]):
            self.bounds.append((parameterInfo["lb"][i], parameterInfo["ub"][i]))
        self.types = list(parameterInfo.pop("type"))
        self.types = [LpContinuous if self.types[i] == "Continuous" else LpInteger for i in range(len(self.types))]

    def fit(self, X, y):
        """
        Fit logistic model.
        :param X:{array-like, sparse matrix} of shape (n_samples, n_features)T
            Training data.
        :param y:array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary.
        :return: A float number.
            The cross-validation score of the fitted model based on the scoring metric you choose.
        """
        print("Now fitting Logistic Regression model.")
        self.model = LogisticRegression(max_iter=1e5)
        self.model.fit(X, y)
        try:
            for i in range(len(self.model.classes_)):
                float(self.model.classes_[i])
        except ValueError:
            print("pyISO can only recognize digital classification.")
        score = cross_val_score(self.model, X, y, cv=5, scoring=self.scoring)
        return np.mean(score)

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

        self.MIP = LpProblem("LogisticRegression", LpMinimize)
        n = len(self.types)
        x = {}
        for i in range(len(self.types)):
            x[i] = LpVariable("x_%d" % i, self.bounds[i][0], self.bounds[i][1], cat=self.types[i])
        y = LpVariable("y")
        k = {}
        for i in range(len(self.model.classes_)):
            k[i] = LpVariable("k_%d" % i, cat=LpBinary)

        self.MIP += y
        self.MIP += y == lpSum(k[i] * float(self.model.classes_[i])
                               for i in range(len(self.model.classes_))), "objective"
        self.MIP += lpSum(k[i] for i in range(len(self.model.classes_))) == 1, "integrity"
        M = 1e5

        if len(self.model.classes_) == 2:
            self.MIP += M * k[1] >= lpSum(self.model.coef_[0][i] * x[i] for i in range(n)) + self.model.intercept_, "k1"
            self.MIP += M * (1 - k[0]) >= -lpSum(
                self.model.coef_[0][i] * x[i] for i in range(n)) - self.model.intercept_, "k0"
        else:
            for l in range(len(self.model.classes_)):
                for j in range(len(self.model.classes_)):
                    if l != j:
                        self.MIP += M * (1 - k[j]) >= lpSum(
                            (self.model.coef_[l] - self.model.coef_[j])[i] * x[i] for i in range(n)) + \
                                    self.model.intercept_[l] - self.model.intercept_[
                                        j], "probabilityComparison_%d_%d" % (l, j)

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
