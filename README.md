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
For windows users, execute the following line in cmd:  
**`pip install pyISBO`**  

For Linux Ubuntu users, execute the following line in the shell:  
**`apt install python3-pyISBO`**  

For Linux CentOS users, execute the following line in the shell:  
**`python3 install pyISBO`**  

For MacOS users, execute the following line in the terminal:  

**`sudo apt install python3-pyISBO`**  

This also works for other packages you might need.

## Getting Started
The framework of modeling in **pyISBO** is as follows.  
<img src="images/steps%20of%20pyISO%20implementation.png" width = "388" height = "355.5" alt="" align=center />
