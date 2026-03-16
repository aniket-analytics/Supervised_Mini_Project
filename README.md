# 🏠 California House Price Prediction using Linear Regression

## 📌 Project Overview

This project demonstrates an **end-to-end Supervised Machine Learning workflow** for predicting housing prices using the **California Housing Dataset**.

The objective is to analyze housing data and build a **Linear Regression model** capable of predicting the **median house value** based on key housing and demographic features.

This project covers the full machine learning pipeline including:

- Data loading
- Data inspection
- Exploratory Data Analysis (EDA)
- Feature selection
- Model training
- Model evaluation
- Model visualization

---

# 📊 Dataset

The dataset used in this project is the **California Housing Dataset**, which contains housing information collected from the **1990 California Census**.

Each row represents a **California district**, and the dataset includes several numerical features describing housing and population characteristics.

## Dataset Features

| Feature | Description |
|-------|-------------|
| MedInc | Median income in the district |
| HouseAge | Median house age |
| AveRooms | Average number of rooms |
| AveBedrms | Average number of bedrooms |
| Population | Population in the district |
| AveOccup | Average house occupancy |
| Latitude | Latitude coordinate |
| Longitude | Longitude coordinate |

### Target Variable

**MedHouseVal**

The median house value for California districts.

---

# ⚙️ Project Workflow

## 1️⃣ Import Required Libraries

The following Python libraries are used in this project:

- **Pandas** – Data manipulation
- **NumPy** – Numerical operations
- **Matplotlib** – Data visualization
- **Seaborn** – Statistical visualization
- **Scikit-learn** – Machine learning model building

---

## 2️⃣ Load the Dataset

The dataset is loaded using Scikit-learn and converted into a Pandas DataFrame for easier analysis.

```python
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing(as_frame=True)
df = data.frame
```

---

## 3️⃣ Data Inspection

Initial inspection helps understand the structure and characteristics of the dataset.

```python
df.head()
df.info()
df.describe()
```

This step allows us to examine:

- Data types
- Number of observations
- Summary statistics

---

## 4️⃣ Missing Value Analysis

Before building a model, the dataset is checked for missing values.

```python
df.isnull().sum()
```

Result:  
The dataset **does not contain missing values**, so no data cleaning was required.

---

# 📊 Exploratory Data Analysis (EDA)

## Distribution of House Prices

A histogram is used to visualize the distribution of the target variable.

```python
sns.histplot(df['MedHouseVal'], bins=50)
```

This helps understand how house prices are distributed across districts.

---

## Correlation Heatmap

A correlation heatmap is used to analyze relationships between variables.

```python
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
```

Insights from this analysis help identify features that strongly influence house prices.

---

# 🧠 Feature Selection

For this project, three important features were selected:

- **Median Income (MedInc)**
- **Average Rooms (AveRooms)**
- **House Age (HouseAge)**

```python
X = df[['MedInc','AveRooms','HouseAge']]
y = df['MedHouseVal']
```

These features were chosen because they show meaningful relationships with housing prices.

---

# 🔀 Train-Test Split

The dataset is divided into training and testing sets to evaluate model performance.

- **80% Training Data**
- **20% Testing Data**

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

# 🤖 Model Training

A **Linear Regression** model is used to learn the relationship between the selected features and house prices.

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
```

---

# 🔮 Model Prediction

After training the model, predictions are made using the testing dataset.

```python
y_pred = model.predict(X_test)
```

---

# 📈 Model Evaluation

The model performance is evaluated using two regression metrics.

## Mean Squared Error (MSE)

Measures the average squared difference between actual and predicted values.

```python
from sklearn.metrics import mean_squared_error

mean_squared_error(y_test, y_pred)
```

Lower values indicate better model performance.

---

## R² Score

Measures how well the model explains the variance in the data.

```python
from sklearn.metrics import r2_score

r2_score(y_test, y_pred)
```

Higher values indicate better predictive performance.

---

# 🔍 Feature Importance

Model coefficients help understand the impact of each feature on house prices.

```python
import pandas as pd

coeff_df = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
coeff_df
```

This helps interpret how different factors affect housing prices.

---

# 📊 Model Visualization

## Actual vs Predicted House Prices

A scatter plot is used to compare predicted values with actual values.

```python
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
```

This visualization helps evaluate the prediction accuracy of the model.

---

# 📂 Project Structure

```
California-House-Prediction
│
├── California_House_Prediction_Mini_Project.ipynb
├── README.md
└── requirements.txt
```

---

# 🚀 How to Run the Project

## 1️⃣ Clone the repository

```
git clone https://github.com/yourusername/california-house-prediction.git
```

## 2️⃣ Navigate to the project folder

```
cd california-house-prediction
```

## 3️⃣ Install required libraries

```
pip install -r requirements.txt
```

## 4️⃣ Run the Jupyter Notebook

Open the notebook and run all cells to reproduce the results.

---

# 📚 Key Learnings

Through this project I gained experience in:

- Performing **Exploratory Data Analysis**
- Visualizing data patterns
- Building regression models
- Evaluating machine learning models
- Interpreting model results

---

# 🔮 Future Improvements

Possible improvements include:

- Using all available dataset features
- Trying advanced models such as Random Forest or Gradient Boosting
- Hyperparameter tuning
- Deploying the model as an API

---

# 👨‍💻 Author

**Aniket Yadav**

If you found this project useful, feel free to ⭐ the repository.
