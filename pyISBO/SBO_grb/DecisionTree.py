from gurobipy import *
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
        self.types = [GRB.CONTINUOUS if self.types[i] == "Continuous" else GRB.INTEGER for i in range(len(self.types))]

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
        Transform the surrogate into a Gurobi mix integer program
        :return: None.
        """
        assert self.model is not None, "You haven't build a surrogate yet. Try using fit() to create one."

        MIP = Model("decision_tree")
        tree_ = self.model.tree_


        leaves_num = find_leaves(0, tree_)
        x_len = len(self.types)
        local_section = local_section_generator(self.bounds)

        '''
        Adding variables:
            Each input parameter corresponds to a variable (x_variable).
            Each leaf node corresponds to a binary variable (y_variable).
        '''
        x_variable = MIP.addVars(list(range(x_len)),lb=local_section[0],
                                        ub=local_section[1],vtype = self.types, name = "x")
        y_variable = MIP.addVars(list(range(len(leaves_num))),vtype = GRB.BINARY, name = "y")

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
        obj = LinExpr(0)
        for i in range(len(leaves_num)):
            obj.addTerms(tree_.value[leaves_num[i]], y_variable[i])
        MIP.setObjective(obj, GRB.MINIMIZE)

        '''
        Adding constraints: 
            Each branch node corresponds to two constraints;
            Only one of all leaf nodes can be selected.
        '''
        M = 10000
        for i in range(tree_.node_count):
            if tree_.children_left[i] != -1:
                cons_leaves_1 = find_leaves(tree_.children_left[i], tree_)
                cons_lhs_1 = LinExpr(0)
                cons_lhs_1 = quicksum(y_dict[j] * M for j in cons_leaves_1)- M - (
                    tree_.threshold[i] - x_variable[tree_.feature[i]])
                MIP.addConstr(cons_lhs_1  <= 0)
                cons_leaves_2 = find_leaves(tree_.children_right[i], tree_)
                cons_lhs_2 = LinExpr(0)
                cons_lhs_2 = quicksum(y_dict[j] * M for j in cons_leaves_2)- M + (
                    tree_.threshold[i] - x_variable[tree_.feature[i]])+1/float(M)
                MIP.addConstr(cons_lhs_2  <= 0)
        y_lhs = quicksum(y_variable[j] for j in range(len(y_variable)))
        MIP.addConstr(y_lhs == 1)
        self.MIP = MIP

    def optimize(self):
        """
        solve the Gurobi mix integer program
        :return: None.
        """
        if self.MIP is None:
            self.MIP_transform()

        local_section = local_section_generator(self.bounds)
        MIP = self.MIP
        x_len = len(self.types)
        MIP.optimize()
        solved_variables = {}

        for v in MIP.getVars():
            solved_variables[v.Varname] = v.X
        solved_variables_dict = {}
        for i in range(x_len):
            try:
                solved_variables_dict['x[{}]'.format(i)] = solved_variables['x[{}]'.format(i)]

            except:
                solved_variables_dict['x[{}]'.format(i)] = random.uniform(local_section[0][i], local_section[1][i])
        optimizedParameter = []

        for i in solved_variables_dict:

            optimizedParameter.append(solved_variables_dict[i])

        self.output = MIP.ObjVal
        self.optimizedParameter = optimizedParameter
