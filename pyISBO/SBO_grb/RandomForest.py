import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from gurobipy import *


def local_section_generator(bounds):
    '''
    Transform the structure of bounds for later algorithm implementation.
    '''
    max = []
    min = []
    for i in range(len(bounds)):
        max.append(bounds[i][1])
        min.append(bounds[i][0])
    return [min, max]


def find_leaves(node_id, tree_):
    '''
    return all leaves of a specific node in list form.
    '''
    if tree_.children_left[node_id] == -1:
        return [node_id]
    else:
        return find_leaves(tree_.children_left[node_id], tree_) + find_leaves(tree_.children_right[node_id], tree_)


class RF:
    """
    Random Forest Surrogate. Based on sklearn.ensemble.RandomForestRegressor
    """
    model = None
    optimizedParameter = None
    scoring = None
    bounds = []
    types = None
    output = None
    MIP = None

    def __init__(self, parameterInfo, scoring="neg_mean_squared_error"):
        """
        Initialize a Random Forest surrogate.
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
        Fit Random Forest model.
        :param X:{array-like, sparse matrix} of shape (n_samples, n_features)T
            Training data.
        :param y:array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary.
        :return: A float number.
            The cross-validation score of the fitted model based on the scoring metric you choose.
        """
        print("Now fitting Random Forest.")
        M_score = []
        M_model = []
        for i in range(4, 9):
            model = RandomForestRegressor(max_depth=i, min_samples_leaf=5)  # 实例化
            model.fit(X, y)
            score = np.mean(cross_val_score(model, X, y, cv=5, scoring=self.scoring))
            M_score.append(score)
            M_model.append(model)
        model_index = M_score.index(max(M_score))
        self.model = M_model[model_index]
        return max(M_score)

    def predict(self, X):
        """
        Predict using the surrogate.
        :param X:{array-like, sparse matrix} of shape (n_samples, n_features)T
            Training data.
        :return:An array, shape (n_samples,)
            Predicted values.
        """
        return self.model.predict(X)

    def MIP_transform(self):
        """
        Transform the surrogate into a Gurobi mix integer program
        :return: None.
        """
        MIP = Model("random_forest")
        estimators_ = self.model.estimators_

        leaves_num = []
        for i in range(len(estimators_)):
            leaves_num.append(find_leaves(0, estimators_[i].tree_))
        x_len = len(self.types)
        local_section = local_section_generator(self.bounds)

        '''
        Adding variables:
            Each input parameter corresponds to a variable (x).
            Each leaf node corresponds to a binary variable (y).
        '''
        x = MIP.addVars(list(range(x_len)), lb=local_section[0],
                                 ub=local_section[1], vtype=self.types, name="x")
        y = {}
        for j in range(len(estimators_)):
            y[j] = MIP.addVars(list(range(len(leaves_num[j]))), vtype=GRB.BINARY, name="y_{}".format(j))

        '''
        Setting a dictionary: with the effect of finding the corresponding decision
        variable by the index of the leaf node.
        '''
        y_dict_list = []
        for j in range(len(estimators_)):
            y_dict = {}
            for i in range(len(leaves_num[j])):
                y_dict[leaves_num[j][i]] = y[j][i]
            y_dict_list.append(y_dict.copy())

        '''
        Set objective function.
        '''
        obj = LinExpr(0)
        for j in range(len(estimators_)):
            tree_ = estimators_[j].tree_
            for i in range(len(leaves_num[j])):
                obj.addTerms(tree_.value[leaves_num[j][i]], y[j][i])
        MIP.setObjective(obj, GRB.MINIMIZE)

        '''
        Adding constraints: 
            Each branch node corresponds to two constraints;
            Only one of all leaf nodes can be selected.
        '''
        M = 10000
        for m in range(len(estimators_)):
            tree_ = estimators_[m].tree_
            for i in range(tree_.node_count):
                if tree_.children_left[i] != -1:
                    cons_leaves_1 = find_leaves(tree_.children_left[i], tree_)
                    cons_lhs_1 = quicksum(y_dict_list[m][j] * M for j in cons_leaves_1) - M - (
                            tree_.threshold[i] - x[tree_.feature[i]])
                    MIP.addConstr(cons_lhs_1 <= 0, "C1")

                    cons_leaves_2 = find_leaves(tree_.children_right[i], tree_)
                    cons_lhs_2 = quicksum(y_dict_list[m][j] * M for j in cons_leaves_2) - M + (
                            tree_.threshold[i] - x[tree_.feature[i]]) + 1 / float(M)
                    MIP.addConstr(cons_lhs_2 <= 0, "C2")
        for m in range(len(estimators_)):
            y_lhs = quicksum(y[m][j] for j in range(len(y[m])))
            MIP.addConstr(y_lhs == 1, "C3")
        self.MIP = MIP
        
    def optimize(self):
        """
        Optimize over the MIP
        :return: None.
            You can get the optimized value and the optimized parameters by "output" and  "optimizedParameter" object.
        """
        if self.MIP is None:
            self.MIP_transform()

        if self.MIP is None:
            self.MIP_transform()
        self.MIP.optimize()
        self.optimizedParameter = [self.MIP.getVarByName("x[%d]" % i).X for i in range(len(self.types))]
        self.output = self.MIP.ObjVal/len(self.model.estimators_)
