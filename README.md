# EX-NO.-4a-MACHINE-LEARNING-MODEL-LINEAR-REGRESSION
## AIM
To predict house prices using regression models and compare the performance of different machine learning regression models based on RMSE, MAE, and R².
1.Machine Learning:Machine Learning is used to learn patterns from existing data and make predictions. 
•	Regression is a supervised learning technique used to predict continuous numerical values. 
•	In this experiment, regression models are used to predict the price of a house. 
•	The dataset contains house-related features such as: 
o	square_feet 
o	num_rooms 
o	age 
o	distance_to_city(km) 
•	The target variable is: 
o	price 
## DATASET DESCRIPTION
•	Dataset: House Price Dataset 
•	Problem: Predict house price. 
•	Features (X): 
o	square_feet – size of the house. 
o	num_rooms – number of rooms. 
o	age – age of the house in years. 
o	distance_to_city(km) – distance from the city centre.  
•	Target (y): 
o	price – continuous house price. 
## PROBLEM STATEMENT
•	Develop a machine learning model to predict house prices. 
•	Use house characteristics as input. 
•	Train different regression models. 
•	Compare their prediction performance. 
•	Select the better-performing model based on evaluation metrics. 
## REGRESSION MODELS USED
The uploaded notebook compares the following models:
1.	Linear Regression 
2.	Ridge Regression 
3.	Lasso Regression 
4.	ElasticNet Regression 
5.	Polynomial Regression 
6.	Decision Tree Regressor 
7.	Random Forest Regressor 
8.	Gradient Boosting Regressor 
9.	Support Vector Regressor (SVR) 
10.	K-Nearest Neighbors (KNN) Regressor 
### LIBRARIES USED
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
### LOAD THE DATASET
import pandas as pd

df = pd.read_csv(
    '/content/drive/MyDrive/Datasets/house_prices_dataset.csv'
)

df.head()
### DATA OVERVIEW
Display the dataset
df
Display information
df.info()
Display shape
df.shape
Summary statistics
df.describe()
Check missing values
df.isnull().sum()
## EXPLORATORY DATA ANALYSIS
Distribution of Features
The notebook examines the distribution of:
•	square_feet 
•	num_rooms 
•	age 
•	distance_to_city(km) 
•	price 
numeric_features = [
    'square_feet',
    'num_rooms',
    'age',
    'distance_to_city(km)',
    'price'
]

for col in numeric_features:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
    plt.show()
9. CORRELATION ANALYSIS
•	Correlation shows the relationship between numerical variables. 
•	A correlation heatmap is used to visualize these relationships. 
plt.figure(figsize=(8,6))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap='coolwarm',
    fmt=".2f"
)

plt.title("Feature Correlation Matrix")
plt.show()
10. SCATTER PLOTS
Scatter plots are used to study the relationship between individual features and house price.
for col in [
    'square_feet',
    'num_rooms',
    'age',
    'distance_to_city(km)'
]:
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=df[col], y=df['price'])
    plt.title(f'{col} vs Price')
    plt.show()
## OUTLIER DETECTION
•	Boxplots are used to identify extreme values. 
•	Outliers may negatively affect regression models. 
for col in [
    'square_feet',
    'num_rooms',
    'age',
    'distance_to_city(km)',
    'price'
]:
    plt.figure(figsize=(6,4))
    sns.boxplot(df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()
## OUTLIER TREATMENT
The notebook removes extremely low and extremely high house prices using the 1st and 99th percentiles.
Q1 = df['price'].quantile(0.01)
Q99 = df['price'].quantile(0.99)

df = df[
    (df['price'] >= Q1) &
    (df['price'] <= Q99)
]
•	This reduces the effect of extreme house prices. 
•	It helps the models learn from more typical observations. 
13. DEFINE FEATURES AND TARGET
X = df[
    [
        'square_feet',
        'num_rooms',
        'age',
        'distance_to_city(km)'
    ]
]

y = df['price']
•	X → Input features. 
•	y → Target house price. 
14. TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
•	80% → Training data 
•	20% → Testing data 
## FEATURE SCALING
The notebook uses StandardScaler.
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)
•	Scaling puts the features on a comparable scale. 
•	The scaler is fitted only on training data. 
•	The same transformation is applied to test data. 

## BASELINE MODEL
The baseline predicts the mean house price for every test observation.
y_pred_baseline = (
    np.mean(y_train) *
    np.ones_like(y_test)
)

rmse_baseline = np.sqrt(
    mean_squared_error(y_test, y_pred_baseline)
)

mae_baseline = mean_absolute_error(
    y_test,
    y_pred_baseline
)

print(
    f"Baseline RMSE: {rmse_baseline:.2f}, "
    f"MAE: {mae_baseline:.2f}"
)

## LINEAR REGRESSION
•	Linear Regression finds a linear relationship between input features and house price. 
•	It is used as the main baseline regression model. 
lr = LinearRegression()

lr.fit(X_train_scaled, y_train)

y_pred_lr = lr.predict(X_test_scaled)

## RIDGE REGRESSION
•	Ridge Regression is a regularized version of Linear Regression. 
•	It helps control large model coefficients. 
ridge = Ridge(alpha=1.0)

ridge.fit(X_train_scaled, y_train)

y_pred_ridge = ridge.predict(X_test_scaled)

## LASSO REGRESSION
•	Lasso Regression uses L1 regularization. 
•	It can reduce some feature coefficients toward zero. 
lasso = Lasso(alpha=0.1)

lasso.fit(X_train_scaled, y_train)

y_pred_lasso = lasso.predict(X_test_scaled)

##  ELASTIC NET REGRESSION
•	ElasticNet combines L1 and L2 regularization. 
•	It is useful when several features may contribute to the prediction. 
elastic = ElasticNet(
    alpha=0.1,
    l1_ratio=0.5
)

elastic.fit(X_train_scaled, y_train)

y_pred_elastic = elastic.predict(X_test_scaled)

##  POLYNOMIAL REGRESSION
•	Polynomial Regression extends Linear Regression by creating polynomial features. 
•	The notebook uses degree 2. 
poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

poly_lr = LinearRegression()

poly_lr.fit(X_train_poly, y_train)

y_pred_poly = poly_lr.predict(X_test_poly)

##  DECISION TREE REGRESSOR
•	A Decision Tree divides the data into different regions based on feature values. 
•	It can model nonlinear relationships. 
dt = DecisionTreeRegressor(
    random_state=42
)

dt.fit(X_train_scaled, y_train)

y_pred_dt = dt.predict(X_test_scaled)

##  RANDOM FOREST REGRESSOR
•	Random Forest combines multiple decision trees. 
•	It generally provides more robust predictions than a single tree. 
rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train_scaled, y_train)

y_pred_rf = rf.predict(X_test_scaled)

##  GRADIENT BOOSTING REGRESSOR
•	Gradient Boosting builds models sequentially. 
•	Each new model attempts to improve the errors of previous models. 
gbr = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

gbr.fit(X_train_scaled, y_train)

y_pred_gbr = gbr.predict(X_test_scaled)

## SUPPORT VECTOR REGRESSOR
•	SVR uses Support Vector Machine principles for regression. 
•	The notebook uses an RBF kernel. 
svr = SVR(
    kernel='rbf',
    C=100,
    gamma=0.1,
    epsilon=.1
)

svr.fit(X_train_scaled, y_train)

y_pred_svr = svr.predict(X_test_scaled)

##  KNN REGRESSOR
•	KNN predicts a value based on nearby observations. 
•	The notebook uses 5 neighbors. 
knn = KNeighborsRegressor(
    n_neighbors=5
)

knn.fit(X_train_scaled, y_train)

y_pred_knn = knn.predict(X_test_scaled)

## MODEL EVALUATION
The notebook uses three metrics:
RMSE
•	Root Mean Squared Error. 
•	Lower value indicates better performance. 
MAE
•	Mean Absolute Error. 
•	Lower value indicates better performance. 
R²
•	Measures how well the model explains variation in house prices. 
•	Higher value generally indicates better performance. 
Code
models = {
    "Linear Regression": y_pred_lr,
    "Ridge": y_pred_ridge,
    "Lasso": y_pred_lasso,
    "ElasticNet": y_pred_elastic,
    "Polynomial Regression": y_pred_poly,
    "Decision Tree": y_pred_dt,
    "Random Forest": y_pred_rf,
    "Gradient Boosting": y_pred_gbr,
    "SVR": y_pred_svr,
    "KNN": y_pred_knn
}

results = []

for name, y_pred in models.items():

    rmse = np.sqrt(
        mean_squared_error(y_test, y_pred)
    )

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    r2 = r2_score(
        y_test,
        y_pred
    )

    results.append([
        name,
        rmse,
        mae,
        r2
    ])

results_df = pd.DataFrame(
    results,
    columns=["Model", "RMSE", "MAE", "R2"]
)

results_df.sort_values(
    by="RMSE"
)

##  MODEL COMPARISON
<img width="601" height="250" alt="image" src="https://github.com/user-attachments/assets/bca8dfef-1052-4e92-90e1-0f6356888be5" />
		

### Comparison Criteria
•	Lower RMSE → Better model. 
•	Lower MAE → Better model. 
•	Higher R² → Better model. 
The notebook sorts the models according to RMSE to compare their performance.
## ACTUAL VS PREDICTED PRICE
plt.figure(figsize=(15,12))

for i, (name, y_pred) in enumerate(models.items()):

    plt.subplot(5,2,i+1)

    plt.scatter(
        y_test,
        y_pred,
        alpha=0.5
    )

    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        'r--'
    )

    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(f"{name}: Actual vs Predicted")

plt.tight_layout()
plt.show()
•	The plot compares actual house prices with predicted prices. 
•	Points closer to the diagonal line indicate better predictions. 
30. RESIDUAL ANALYSIS
•	A residual is the difference between actual and predicted values. 
Residual = Actual Price − Predicted Price
for name, y_pred in models.items():

    residuals = y_test - y_pred

    plt.figure(figsize=(6,4))

    sns.scatterplot(
        x=y_pred,
        y=residuals,
        alpha=0.5
    )

    plt.axhline(
        0,
        color='r',
        linestyle='--'
    )

    plt.xlabel("Predicted Price")
    plt.ylabel("Residuals")
    plt.title(f"{name}: Residual Plot")

    plt.show()
•	Residuals close to zero indicate smaller prediction errors. 
•	Residual plots help identify unusual prediction patterns. 
##  RANDOM FOREST FEATURE IMPORTANCE
The notebook calculates the importance of each feature using Random Forest.
importances = rf.feature_importances_

feat_names = X.columns

plt.figure(figsize=(6,4))

sns.barplot(
    x=importances,
    y=feat_names
)

plt.title(
    "Random Forest Feature Importance"
)

plt.show()
This helps identify which house features contribute more to the Random Forest prediction.
##  GRADIENT BOOSTING FEATURE IMPORTANCE
importances_gbr = gbr.feature_importances_

plt.figure(figsize=(6,4))

sns.barplot(
    x=importances_gbr,
    y=feat_names
)

plt.title(  "Gradient Boosting Feature Importance")

plt.show()
This shows the relative contribution of the house features to the Gradient Boosting model.

## RMSE COMPARISON GRAPH
plt.figure(figsize=(10,6))

sns.barplot(
    x="RMSE",
    y="Model",
    data=results_df.sort_values("RMSE")
)

plt.title(
    "RMSE Comparison Across Regression Models"
)

plt.show()
•	The graph provides a visual comparison of model errors. 
•	The model with the lowest RMSE performs best according to this metric.
## CONCLUSION
Thus, Linear Regression and other regression models were successfully applied for house price prediction, and their performance was compared using standard regression evaluation metrics.

