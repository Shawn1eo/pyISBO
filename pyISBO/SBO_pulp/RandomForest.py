import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from pulp import *
import random


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
        self.types = [LpContinuous if self.types[i] == "Continuous" else LpInteger for i in range(len(self.types))]

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
        Transform the surrogate into a Pulp mix integer program
        :return: None.
        """
        MIP = LpProblem("random_forest", LpMinimize)
        estimators_ = self.model.estimators_

        leaves_num = []
        for i in range(len(estimators_)):
            leaves_num.append(find_leaves(0, estimators_[i].tree_))
        x_len = len(self.types)
        local_section = local_section_generator(self.bounds)
        x_variable = {}

        '''
        Adding variables:
            Each input parameter corresponds to a variable (x_variable).
            Each leaf node corresponds to a binary variable (y_variable).
        '''
        for i in range(x_len):
            x_variable[i] = LpVariable('x_{}'.format(i), lowBound=local_section[0][i], upBound=local_section[1][i],
                                          cat=self.types[i])

        y_variable = {}
        for j in range(len(estimators_)):
            y_variable[j] = LpVariable.dict('y_{}'.format(j), range(len(leaves_num[j])), cat=LpBinary)

        '''
        Setting a dictionary: with the effect of finding the corresponding decision
        variable by the index of the leaf node.
        '''
        y_dict_list = []
        for j in range(len(estimators_)):
            y_dict = {}
            for i in range(len(leaves_num[j])):
                y_dict[leaves_num[j][i]] = y_variable[j][i]
            y_dict_list.append(y_dict.copy())

        '''
        Set objective function.
        '''
        objective = 0
        for j in range(len(estimators_)):
            objective += lpSum(
                y_variable[j][i] * estimators_[j].tree_.value[leaves_num[j][i]] for i in range(len(leaves_num[j])))
        MIP += objective

        '''
        Adding constraints: 
            Each branch node corresponds to two constraints;
            Only one of all leaf nodes can be selected.
        '''
        # Note that since pulp does not allow large numbers with high
        # precision, float(inf) is not allowed.
        M = 10000
        for m in range(len(estimators_)):
            for i in range(estimators_[m].tree_.node_count):
                if estimators_[m].tree_.children_left[i] != -1:
                    cons_leaves_1 = find_leaves(estimators_[m].tree_.children_left[i], estimators_[m].tree_)
                    MIP += M * (lpSum(y_dict_list[m][j] for j in cons_leaves_1) - 1) - (
                                estimators_[m].tree_.threshold[i] - x_variable[estimators_[m].tree_.feature[i]]) <= 0
                    cons_leaves_2 = find_leaves(estimators_[m].tree_.children_right[i], estimators_[m].tree_)
                    MIP += M * (lpSum(y_dict_list[m][j] for j in cons_leaves_2) - 1) + (
                                estimators_[m].tree_.threshold[i] - x_variable[
                            estimators_[m].tree_.feature[i]]) + 1 / M <= 0
            MIP += lpSum(y_variable[m][i] for i in range(len(y_variable[m]))) == 1

        self.MIP = MIP
        
    def optimize(self):
        """
        solve the Pulp mix integer program
        :return: None.
        """
        if self.MIP is None:
            self.MIP_transform()

        local_section = local_section_generator(self.bounds)
        MIP = self.MIP
        MIP.solve()
        solved_variables = {}
        for v in MIP.variables():
            solved_variables[v.name] = v.varValue
        solved_variables_dict = {}
        for i in range(len(self.types)):
            try:
                solved_variables_dict['x_{}'.format(i)] = solved_variables['x_{}'.format(i)]
            except:
                solved_variables_dict['x_{}'.format(i)] = random.uniform(local_section[0][i], local_section[1][i])
                if self.types[i] == LpInteger:
                    solved_variables_dict['x_{}'.format(i)] = int(solved_variables_dict['x_{}'.format(i)] + 0.5)
        optimizedParameter = []
        for i in solved_variables_dict:
            optimizedParameter.append(solved_variables_dict[i])
        # The objective function of the random forest is to sum the values of
        # all trees, averaging on return to maintain consistency with y.
        self.output = value(MIP.objective) / len(self.model.estimators_)
        self.optimizedParameter = optimizedParameter
