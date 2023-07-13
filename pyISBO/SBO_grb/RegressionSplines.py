from gurobipy import *
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from pyearth import Earth


class MARS:
    """
    Linear Surrogate. Based on sklearn.linear_model.LinearRegression.
    """
    model = None
    optimizedParameter = None
    scoring = None
    bounds = []
    output = None
    types = None
    MIP = None
    colName = None

    def __init__(self, parameterInfo, scoring="neg_mean_squared_error"):
        """
        Initialize a MARS surrogate.
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
        self.types = [GRB.CONTINUOUS if self.types[i] == "Continuous" else GRB.INTEGER for i in range(len(self.types))]
        self.colName = list(parameterInfo.pop("parameterName"))

    def fit(self, X, y):
        """
        Fit the MARS model.
        :param X:{array-like, sparse matrix} of shape (n_samples, n_features)T
            Training data.
        :param y:array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary.
        :return: A float number.
            The cross-validation score of the fitted model based on the scoring metric you choose.
        """
        print("Now fitting MARS model.")
        self.model = Earth()
        self.model.fit(X, y)
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
        Transform the surrogate into a Gurobi linear program
        :return: None.
            You can access the transformed linear model by MIP object.
        """
        assert self.model is not None, "You haven't build a surrogate yet. Try using fit() to create one."

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

        self.MIP = Model("MARS")
        inputInfo = {}
        for i in range(len(self.types)):
            inputInfo[(i)] = [self.bounds[i][0], self.bounds[i][1], self.types[i]]
        inp, lb, ub, vtype = multidict(inputInfo)

        x = self.MIP.addVars(inp, lb=lb, ub=ub, vtype=vtype, name="x")
        y = self.MIP.addVar(lb=-1e5, vtype=GRB.CONTINUOUS, name="y")
        branch = self.MIP.addVars(branch_index, lb=-1e5, vtype=GRB.CONTINUOUS, name="branch")
        z = self.MIP.addVars(branch_index, lb=0, vtype=GRB.BINARY, name="z")
        self.MIP.update()

        M = 1e5
        self.MIP.addConstr((y == intercept + quicksum(branch[i] for i in range(len(summary)))), name="objective")
        for col in range(len(self.colName)):
            colname = self.colName[col]
            for i in range(len(summary)):
                coef = float(summary[i][2])
                if summary[i][0] == colname:  # 没有分割点
                    self.MIP.addConstr((branch[i] == coef * x[col]), name="branch[%d]" % i)
                if summary[i][0][2:2 + len(colname)] == colname:
                    point = summary[i][0][3 + len(colname):len(summary[i][0]) - 1]
                    try:
                        point = float(point)
                        self.MIP.addConstr((branch[i] >= x[col] - point), name="h1[%d]" % i)
                        self.MIP.addConstr((branch[i] >= 0), name="h2[%d]" % i)
                        self.MIP.addConstr((branch[i] <= x[col] - point + M*(1-z[i])), name="h3[%d]" % i)
                        self.MIP.addConstr((branch[i] <= M * z[i]), name="h4[%d]" % i)
                    except ValueError:
                        pass

                if summary[i][0][len(summary[i][0]) - 1 - len(colname):len(summary[i][0]) - 1] == colname:
                    point = summary[i][0][2:len(summary[i][0]) - 2 - len(colname)]
                    try:
                        point = float(point)
                        self.MIP.addConstr((branch[i] >= point - x[col]), name="h1[%d]" % i)
                        self.MIP.addConstr((branch[i] >= 0), name="h2[%d]" % i)
                        self.MIP.addConstr((branch[i] <= point - x[col] + M * (1 - z[i])), name="h3[%d]" % i)
                        self.MIP.addConstr((branch[i] <= M * z[i]), name="h4[%d]" % i)
                    except ValueError:
                        pass
        self.MIP.update()

    def optimize(self):
        """
        Optimize over the MIP
        :return: None.
            You can get the optimized value and the optimized parameters by "output" and  "optimizedParameter" object.
        """
        if self.MIP is None:
            self.MIP_transform()
        self.MIP.optimize()
        self.optimizedParameter = [self.MIP.getVarByName("x[%d]" % i).X for i in range(len(self.types))]
        self.output = self.MIP.getVarByName("y").X