from copy import deepcopy
from .DecisionTree import DT
from .LinearRegression import LR
from .LogisticRegression import LogR
from .NeuralNetwork import NN
from .RandomForest import RF
from .RegressionSplines import MARS


class AR:
    """
       AutoRegression Tool. It can automatically choose a surrogate with the best performance.
    """
    parameterInfo = None
    model = None
    optimizedParameter = None
    scoring = None
    output = None
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
        self.parameterInfo = parameterInfo

    def fit(self, X, y, Classification = False):
        """
        Fit this surrogate with your data. A surrogate with the best performance will be chosen.
        :param Classification: Boolean Variable
            Whether this is a classification problem.
        :param X:{array-like, sparse matrix} of shape (n_samples, n_features)T
            Training data.
        :param y:array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary.
        :return: A float number.
            The cross-validation score of the fitted model based on the scoring metric you choose.
        """
        if Classification:
            print("Using Logistic model as the surrogate.")
            self.model = LogR(self.parameterInfo, self.scoring)
            self.model.fit(X, y)
        else:
            LRmodel = LR(deepcopy(self.parameterInfo), deepcopy(self.scoring))
            NNmodel = NN(deepcopy(self.parameterInfo), deepcopy(self.scoring))
            DTmodel = DT(deepcopy(self.parameterInfo), deepcopy(self.scoring))
            RFmodel = RF(deepcopy(self.parameterInfo), deepcopy(self.scoring))
            RSmodel = MARS(deepcopy(self.parameterInfo), deepcopy(self.scoring))
            score = [LRmodel.fit(X,y), NNmodel.fit(X,y), DTmodel.fit(X,y), RFmodel.fit(X,y),
                     RSmodel.fit(X,y)]
            index = score.index(max(score))
            if index == 0:
                print("\nLinear Model has the best overall performance.\n")
                self.model = LRmodel
            elif index == 1:
                print("\nNeural Network has the best overall performance.\n")
                self.model = NNmodel
            elif index == 2:
                print("\nDecision Tree has the best overall performance.\n")
                self.model = DTmodel
            elif index == 3:
                print("\nRandom Forest has the best overall performance.\n")
                self.model = RFmodel
            else:
                print("\nMARS Model has the best overall performance.\n")
                self.model = RSmodel

    def predict(self, X):
        """
        Predict using the linear model.
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

        self.model.MIP_transform()
        self.MIP = self.model.MIP

    def optimize(self):
        """
        Optimize over the MIP
        :return: None.
            You can get the optimized value and the optimized parameters by "output" and  "optimizedParameter" object.
        """
        if self.MIP is None:
            self.MIP_transform()
        self.model.optimize()
        self.optimizedParameter = self.model.optimizedParameter
        self.output = self.model.output


