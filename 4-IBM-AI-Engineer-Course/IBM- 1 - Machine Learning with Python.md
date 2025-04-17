
## 🌟Course Introduction 🌟


### Pre-requisites
- Pandas
- NumPy
- Data preparation and data analysis with python

### What you will learn?
![[Pasted image 20250411140839.png]]

- Multiple linear regression
- Logistic regression
- Prediction
- Fraud Detection
- KNN and SVM

## M1:  ML Concepts

> Foundational machine learning concepts to delve deeper into applied machine learning modeling.


### Machine Learning

#### ML and DL 

Machine Learning :
- Using Algorithms
- Require Feature Engineering

Deep Learning:
- Using Multi-Layer Neural Network
- Automatically extract features

#### How machine learning work?

Teach computers to :
- learn from data
- identify patterns
- make decisions

Algorithms:
- Use computational methods for learning
- doesn't rely on a fixed algorithm

#### ML Paradigms

- **supervised learning models** :  train on labeled data 
- **unsupervised learning**:  works without labels.
- **semi-supervised learning**: works iteratively
- **reinforcement learning**: learns from feedback


### ML Techniques

![[Pasted image 20250411174710.png]]

- **classification**： predicts class or category of a case.
- **regression**:  predicts *continuous* values
- **clustering**： groups similar data
- **association**： finds items or events that co-occur
- **anomaly detection**: discovers abnormal and unusual cases
- **sequence mining**: predicts next event from ordered data
- **dimension reduction**: reduces size of data
- **recommendation systems** : associate people's preferences.

![[classsification-regression-clustering.png]]

### Applications of ML

![[Pasted image 20250411181921.png]]

#### Image Recognition with ML

> **Data**: Images of cats and dogs
> **Traditional Programming**: create rules to detect the animals
> **ML**: build a model to infer the animal type

![[Pasted image 20250411182233.png]]

#### Pairing ML with human intelligence

![[Pasted image 20250411182553.png]]

- chatbot
- face recognition
- computer games

### Machine Learning Model Lifecycle

#### Processes of the lifecycle

- **🔍 Problem Definition**  
- **📊 Data Collection**  
- **🧹 Data Preparation**  
	- clean data
	- **Explore data analysis**
	- **Train-Test Split**
- **🤖 Model Development**  
	- explore existing frameworks
	- **content-based filtering** : focus on similarities between **product features**
	- **collaborative filtering**: recommend -->
		- **similar users** target item
		- **similar items** to users who buy target item
		- 查看详细内容： [[2.6 推荐系统]] 
- ✅ **Model Evaluation**
- **🚀 Model Deployment**

### Tools for ML

![[Pasted image 20250411200005.png]]

#### Data

Data is a collection of:
-  raw facts
- figures
- information
used to draw insights, inform decisions, and fuel advanced technologies.
#### Common Languages for ML

- Python
	- analyzing and processing data
	- developing ML models
- R:
	- statistical learning
	- data exploration and ML
- Julia: parallel and distributed numerical computing support
- Scala： Processing big data and building ML pipelines
- Java： Support Scalable ML applications
- JavaScript: Running ML models in web browsers

#### Diff. types of ML tools

- Data processing and analytics
	- PostgreSQL 
	- Hadoop
	- Spark
	- Kafka
	- **Pandas**: Data manipulation and analysis
	- **NumPy**: Fast numerical computation on arrays and matrices.
- Data Visualization
	- Matplotlib
	- Seaborn
	- ggplot2
	- Tableau
- Shallow Machine Learning
	- **scipy**: computing for optimization, integration and linear regression
	- **scikit-learn**: suite of classification, regression, clustering and dimensionality reduction
- Deep Learning
	- TensorFlow
	- Keras
	- Theano
	- PyTorch
- Computer Vision Application
	- OpenCV: real-time computer vision applications
	- Scikit-image: image processing algorithms
	- TorchVision: 
- Natural Language Processing
	- **NLTK**: text processing, tokenization, and stemming
	- TextBlob: part-of-speech tagging, noun phrase extraction, sentiment analysis, and translation
	- Stanza: pre-trained models for tasks such as NER and dependency parsing
- Generative AI tools
	- Hugging face transformers
	- DALL-E
	- PyTorch

### Sciki-Learn Library

- Free ML library for python
- Classification, Regression, Clustering, and Dimensionality reduction algorithms
- Designed to work with NumPy and SciPy
- Excellent documentation and community support
- Constantly evolving
- Enables easy implementation of ML models

#### Machine Learning pipeline tasks
![[Pasted image 20250411200910.png]]

#### Sk Learn Demo

- 标准化（Standardization）：
把特征转换成 **均值为 0，标准差为 1** 的分布，也叫 **Z-score 标准化**。
```python
import numpy as np  
from sklearn.preprocessing import StandardScaler

X = np.array([[1], [2], [3], [4], [5]])
X_scaled = StandardScaler().fit_transform(X)  
```
计算公式如下：
$$
X_{\text{scaled}} = \frac{X - \mu}{\sigma} , \mu 平均值，\sigma 为方差
$$
- 模型训练
线性回归 预测y=2x
```python
from sklearn.linear_model import LinearRegression  
import numpy as np  
  
# 训练数据  
X = np.array([[1], [2], [3], [4], [5]])  # 输入特征，必须是二维数组  
y = np.array([2, 4, 6, 8, 10])           # 标签（目标值）  
  
# 创建模型并训练  
model = LinearRegression()  
model.fit(X, y)  
# 使用模型进行预测  
y_pred = model.predict(np.array([[6], [7]])  )
```

## M2: Linear and Logistic Regression

>2 classical statistical methods:
> - linear regression
> - logistic regression

[[2.5 回归与分类]]

Regression is a statistical method used to model the **relationship** between a continuous input **variable**  and explanatory **features**.

a type of supervised learning model
![[Pasted image 20250414190604.png]]
### Regression algorithms

- Linear and polynomial 
- nonlinear regression
	- Random forest
	- Extreme Gradient Boosting (XGBoost)
	- K-nearest neighbors (KNN)
	- Support Vector machines (SVM)
	- Neural network


### Applications of Regression

- sales forecasting
- price estimating
- predictive maintenance
- employment income
- rainfall estimation
- wildfire probability and severity
- spread of infectious disease
- risk of chronic disease




### Type of Regression
- simple regression
	- simple linear regression
	- simple nonlinear regression
- multiple regression
	- multiple linear regression
	- multiple nonlinear regression

#### Simple Linear Regression
[SimpleLinearRegression-Jupyter Lab](../jupyter-notes/IBM-1-2-1-SimpleLinearRegression.ipynb)

predict a continuous value
![[Pasted image 20250414194030.png]]
- the Best Fit
find the <font style="background-color:tomato; color:black">regression line</font> or hyperplane that best describe the relationship between X and Y.  

#### OLS Regression 

 **Sum of Squared Residuals, SSR**
$$\text{SSR} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

- easy to understand and interpret
- the method doesn't require any tuning
- solution just a calculation
- accuracy can be greatly reduced by outliers

#### Multiple Linear Regression
[MultipleLinearRegression-Jupyter Lab](../jupyter-notes/IBM-1-2-2-MultipleLinearRegression.ipynb)


Features:
- better than simple linear regression
- too many variables can cause overfitting
- To improve prediction, convert categorical independent vars into numerical vars

Application:
- used in education to predict outcomes and explain relationships
- used to predict the impact of changes in "what-if" scenarios

<font style="background-color:orange; color:black">correlation pitfalls</font>
- remove redundant variables
- select vars which are :
	- most understood
	- controllable
	- most correlated with target

Fitting a <font style="background-color:tomato; color:black">hyperplane</font>
![[Pasted image 20250415093020.png]]

<font style="background-color:tomato; color:black">Least Squares Solution</font>
is usually the best solution for standard linear regression.

#### Nonlinear Regression
![[Pasted image 20250415163345.png]]
Features：
- Represented by a nonlinear equation
	- polynomial
	- exponential
	- logarithmic
	- nonlinear function
	- periodicity

### logistic regression

- sigmoid 
- cross-entropy
- decision boundary
- stochastic gradient descent SGD
- log-loss
- threshold probability

![[Pasted image 20250415202645.png]]



### Cheat Sheet
#### Comparing different regression types

|Model Name|Description|Code Syntax|
|---|---|---|
|Simple linear regression|**Purpose:** To predict a dependent variable based on one independent variable.  <br>**Pros:** Easy to implement, interpret, and efficient for small datasets.  <br>**Cons:** Not suitable for complex relationships; prone to underfitting.  <br>**Modeling equation:** y = b0 + b1x|1. `from sklearn.linear_model import LinearRegression`<br>2. `model = LinearRegression()`<br>3. `model.fit(X, y)`|
|Polynomial regression|**Purpose:** To capture nonlinear relationships between variables.  <br>**Pros:** Better at fitting nonlinear data compared to linear regression.  <br>**Cons:** Prone to overfitting with high-degree polynomials.  <br>**Modeling equation:** y = b0 + b1x + b2x2 + ...|1. `from sklearn.preprocessing import PolynomialFeatures`<br>2. `from sklearn.linear_model import LinearRegression`<br>3. `poly = PolynomialFeatures(degree=2)`<br>4. `X_poly = poly.fit_transform(X)`<br>5. `model = LinearRegression().fit(X_poly, y)`|
|Multiple linear regression|**Purpose:** To predict a dependent variable based on multiple independent variables.  <br>**Pros:** Accounts for multiple factors influencing the outcome.  <br>**Cons:** Assumes a linear relationship between predictors and target.  <br>**Modeling equation:** y = b0 + b1x1 + b2x2 + ...|1. `from sklearn.linear_model import LinearRegression`<br>2. `model = LinearRegression()`<br>3. `model.fit(X, y)`|
|Logistic regression|**Purpose:** To predict probabilities of categorical outcomes.  <br>**Pros:** Efficient for binary classification problems.  <br>**Cons:** Assumes a linear relationship between independent variables and log-odds.  <br>**Modeling equation:** log(p/(1-p)) = b0 + b1x1 + ...|1. `from sklearn.linear_model import LogisticRegression`<br>2. `model = LogisticRegression()`<br>3. `model.fit(X, y)`|

#### Associated functions commonly used

|Function/Method Name|Brief Description|Code Syntax|
|---|---|---|
|train_test_split|Splits the dataset into training and testing subsets to evaluate the model's performance.|1. `from sklearn.model_selection import train_test_split`<br>2. `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)`|
|StandardScaler|Standardizes features by removing the mean and scaling to unit variance.|1. `from sklearn.preprocessing import StandardScaler`<br>2. `scaler = StandardScaler()`<br>3. `X_scaled = scaler.fit_transform(X)`|
|log_loss|Calculates the logarithmic loss, a performance metric for classification models.|1. `from sklearn.metrics import log_loss`<br>2. `loss = log_loss(y_true, y_pred_proba)`|
|mean_absolute_error|Calculates the mean absolute error between actual and predicted values.|1. `from sklearn.metrics import mean_absolute_error`<br>2. `mae = mean_absolute_error(y_true, y_pred)`|
|mean_squared_error|Computes the mean squared error between actual and predicted values.|1. `from sklearn.metrics import mean_squared_error`<br>2. `mse = mean_squared_error(y_true, y_pred)`|
|root_mean_squared_error|Calculates the root mean squared error (RMSE), a commonly used metric for regression tasks.|1. `from sklearn.metrics import mean_squared_error`<br>2. `import numpy as np`<br>3. `rmse = np.sqrt(mean_squared_error(y_true, y_pred))`|
|r2_score|Computes the R-squared value, indicating how well the model explains the variability of the target variable.|1. `from sklearn.metrics import r2_score`<br>2. `r2 = r2_score(y_true, y_pred)`|

## M3: Building Supervised Learning Models

### Basics
#### Classification
![[Pasted image 20250416091910.png]]


#### Application of Classification
- problems expressed as associations between feature and target variables
- used to build apps for 
	- email filtering
	- speech-to-text
	- handwriting recognition
	- biometric identification
- Customer Service
	- churn prediction
	- customer segmentation
	- advertising ： predict if a customer will respond to a campign

#### Classification Algorithms
- Naive Bayes
- Logistic Regression
- Decision Trees
- K-nearest neighbors
- Support Vector Machines (SVM)
- Neural networks

#### Multi-class classification

- One-Versus-All 
- one-versus-one


### Type of Classification Algorithm

#### Decision Trees
- Classification Tree
- Regression Tree






## M4: Building Unsupervised Learning Models


## M5:  Evaluating and Validating Machine Learning Models

## M6: Final Exam and Project
