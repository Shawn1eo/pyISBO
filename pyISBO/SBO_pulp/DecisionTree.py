from pulp import *
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
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


class DT:
    """
    Decision Tree Surrogate. Based on sklearn.tree.DecisionTreeRegressor.
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
        Initialize a Decision Tree surrogate.
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
        self.types = list(parameterInfo.pop('type'))
        self.types = [LpContinuous if self.types[i] == "Continuous" else LpInteger for i in range(len(self.types))]

    def fit(self, X, y):
        """
        Fit Decision Tree model.
        :param X:{array-like, sparse matrix} of shape (n_samples, n_features)T
            Training data.
        :param y:array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary.
        :return: A float number.
            The cross-validation score of the fitted model based on the scoring metric you choose.
        """
        print("Now fitting Decision Tree.")
        M_score = []
        M_model = []
        for i in range(4, 9):
            model = DecisionTreeRegressor(max_depth=i, min_samples_leaf=5)  # 实例化
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
        assert self.model is not None, "You haven't build a surrogate yet. Try using fit() to create one."
        return self.model.predict(X)

    def MIP_transform(self):
        """
        Transform the surrogate into a Pulp mix integer program.
        :return: None.
        """
        MIP = LpProblem("decision_tree", LpMinimize)
        tree_ = self.model.tree_

        leaves_num = find_leaves(0, tree_)
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

        y_variable = LpVariable.dict('y', range(len(leaves_num)), cat=LpBinary)

        '''
        Setting a dictionary: with the effect of finding the corresponding decision
        variable by the index of the leaf node.
        '''
        y_dict = {}
        for i in range(len(leaves_num)):
            y_dict[leaves_num[i]] = y_variable[i]

        '''
        Set objective function.
        '''
        objective = 0
        objective += lpSum(y_variable[i] * tree_.value[leaves_num[i]] for i in range(len(leaves_num)))
        MIP += objective

        '''
        Adding constraints: 
            Each branch node corresponds to two constraints;
            Only one of all leaf nodes can be selected.
        '''
        # Note that since pulp does not allow large numbers with high
        # precision, float(inf) is not allowed.
        M = 10000
        for i in range(tree_.node_count):
            if tree_.children_left[i] != -1:
                cons_leaves_1 = find_leaves(tree_.children_left[i], tree_)
                MIP += M * (lpSum(y_dict[j] for j in cons_leaves_1) - 1) - (
                    tree_.threshold[i] - x_variable[tree_.feature[i]]) <= 0
                cons_leaves_2 = find_leaves(tree_.children_right[i], tree_)
                MIP += M * (lpSum(y_dict[j] for j in cons_leaves_2) - 1) + (
                    tree_.threshold[i] - x_variable[
                        tree_.feature[i]]) + 1 / M <= 0
        MIP += lpSum(y_variable[i] for i in range(len(y_variable))) == 1

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
        self.output = value(MIP.objective)
        self.optimizedParameter = optimizedParameter
