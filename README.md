# pyISBO
**pyISBO** is a Python based integrated surrogate-based optimization toolbox that integrates surrogate **construction, transformation, customization and optimization**. To the best of our knowledge, it's the first integrated whole-process surrogate-based optimization toolbox in Python. **pyISO** can be used to automatically choose and construct a surrogate with minimized cross-validation error, and minimize/maximize the response variable based on it. **pyISO** also allows users to specify and train a certain surrogate, hereafter they can integrate it into their own optimization models with other variables, constraints and new objective. **pyISO** currently supports predictive models of the following categories: Linear Regression, Quadratic Regression, Logistic Regression, Decision Tree Regression, Random Forest, Multivariate Adaptive Regression Splines (MARS)  and Deep Neural Network with rectified linear activation function (DNN). 

## setup：  
To fully utilize all the functions in this package, your Python environment need to satisfy all following requirement (which we highly recommend).
1. 3.5 <= python <= 3.7;  
2. Gurobi installed or PuLP installed;  
3. TensorFlow>=2.0.0;    
4. numpy >= 1.19.0;  
5. pandas >= 1.3.0;  
6. scikit-learn >= 1.0.2;  
7. sklearn-contrib-py-earth >= 0.1.0;

You can still use this package if some requirement above is not satistied. Within these packages, numpy and pandas are neccessary, and the relationship between surrogates and packages are listed in the following table.

| Surrogate Type | **pyISBO** class | Related Package |
|----|----|----|
| Linear Regression | LR | scikit-learn |
| Quadratic Regression | QR | scikit-learn |
| Logistic Regression | LogR | scikit-learn |
| Decision Tree | DT | scikit-learn |
| Random Forest | RF | scikit-learn |
| Neural Network | NN | tensorflow |
| Multiple Adaptive Regression Spline  | MARS | sklearn-contrib-py-earth |

Surrogate is not available if the corresponding package is not installed. And if any of the requirement above is not satisfied, then the automatical regression function *AutoRegression* is not available either. Due to version differences, many users might encounter difficulty when installing py-earth package (sklearn-contrib-py-earth). After our practice, installing this package by its wheels is most stable. Users may refer to https://pypi.org/project/sklearn-contrib-py-earth/#files to find the corresponding version of your computer.

## Installation
The **pyISBO** package can be installed by using <code>pip</code> command:  
For windows/Linux CentOS/MacOS users, execute the following line in cmd/shell/terminal:  
**`pip install pyISBO`**  

For Linux Ubuntu users, execute the following line in the shell:  
**`pip install pyISBO --break-system-packages`**  

This also works for other packages you might need.

## User Guidance
The framework of modeling in **pyISBO** is as follows.  
<img src="images/steps%20of%20pyISO%20implementation.png" width = "388" height = "355.5" alt="" align=center />

### Choose and instantiate a surrogate
In the first step, users need to choose a surrogate model category and instantiate. As aforementioned, pyISO provides seven kinds of surrogate models. Users can choose a specific surrogate model according to their wills, or choose *AutoRegression* which can automatically choose a surrogate model with the best performance in the second step. In this step, you need to give a information table of your parameters and specify a scoring metric. The parameter information table should contain four columns: Name, lb, ub and types, which correspond to the names, lower bounds, upper bounds and types of your input variables. You can find an example by checking the second sheet of "example.xlsx", which is in the same root directory as this file. As for scoring metric, **pyISBO** share the same scoring system as sklearn, use <code>sorted(sklearn.metrics.SCORERS.keys())</code> to get valid options. For example:  

<code>from pyISBO.SO_grb.LinearRegression import LR
model = LR(parameterInfo, "neg_mean_squared_error")
</code>

By this code, user instantiate a linear regression surrogate, input the parameter information table and specify "neg_mean_squared_error" as the scoring metric.

### Fit the surrogate with your data
Next, users can train the model with their data by *fit* function. In this step, the requirement of data is the same as other sklearn machine learning model. The training data should be matrix of shape (n_sample, n_features), the target value should be array of shape  (n_samples, 1). Currently, pyISBO don't support multiple targets. If you choose *AutoRegression* class, then you need to specify whether this is a classification problem by a boolean parameter *classification* whose default value is False.  
Hyper-parameters of some surrogates in **pyISBO** are adjustable. In this step, **pyISBO** will automatically choose hyper-parameters with the best cross-validation score.

### Transform the surrogate model into a MIP model (optional)
After the surrogate model is fitted, users can transform it into a mix integer program by *MIP_transform* function. You can access the transformed linear model by object *MIP*.

### Make customization to the MIP model (optional)
If you want to specify your own variables, constraints and objective, you can edit the transformed MIP by adjusting the *MIP* object in the model. The operations of editing is consistent with the solver.  
For example, in *SO_grb*, you can get variables in the MIP and make cusmization in the following way:  
<code>x = [model.MIP.getvarbyname("x[%d]" % i) for i in range(numOfParameter)]
y = model.MIP.getvarbyname("y")

model.MIP.setObjective(y, GRB.MINIMIZE)
model.MIP.addConstr(quicksum(a[i]*x[i] for i in range(numOfParameter)) <= b)
</code>

In *SO_pulp*, you can get variables in the MIP and make cusmization in the following way:  
<code>x = [model.MIP.variablesDict()["x_%d" % i] for i in range(numOfParameter)]
y = model.MIP.variablesDict()["y"]

model.MIP.setObjective(y)
model.MIP.sense = LpMinimize
model.MIP.addConstr(lpSum(a[i]*x[i] for i in range(numOfParameter)) <= b)
</code>

### Solve the model
Finally, you can solve the MIP by *optimize* method. The solution and optimized objective value can be found in *optimizedParameter* and *output* object.

# Contact Us!
If you find any problem when using this package, feel free to contact us by email: liu-xt22@mails.tsinghua.edu.cn
