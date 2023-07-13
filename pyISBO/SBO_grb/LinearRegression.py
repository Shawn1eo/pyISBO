from gurobipy import *
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score


class LR:
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
        self.types = [GRB.CONTINUOUS if self.types[i] == "Continuous" else GRB.INTEGER for i in range(len(self.types))]

    def fit(self, X, y):
        """
        Fit the linear model.
        :param X:{array-like, sparse matrix} of shape (n_samples, n_features)T
            Training data.
        :param y:array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary.
        :return: A float number.
            The cross-validation score of the fitted model based on the scoring metric you choose.
        """
        print("Now fitting Linear Regression model.")
        self.model = LinearRegression()
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

        self.MIP = Model("LinearRegression")
        inputInfo = {}
        for i in range(len(self.types)):
            inputInfo[(i)] = [self.bounds[i][0], self.bounds[i][1], self.types[i]]
        inp, lb, ub, vtype = multidict(inputInfo)

        x = self.MIP.addVars(inp, lb=lb, ub=ub, vtype=vtype, name="x")
        y = self.MIP.addVar(lb=-1e5, vtype=GRB.CONTINUOUS, name="y")
        self.MIP.setObjective(y, GRB.MINIMIZE)
        self.MIP.addConstr(y == self.model.intercept_ + quicksum(x[i] * self.model.coef_[i]
                                                                 for i in range(len(self.types))), "LinearTransform")
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