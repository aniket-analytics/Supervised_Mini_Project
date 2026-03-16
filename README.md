# 🏠 California Housing Price Prediction using Linear Regression

## 📌 Project Overview
This project is an **end-to-end Supervised Machine Learning project** where a **Linear Regression model** is built to predict housing prices.

The objective of this project is to analyze housing data and build a predictive model that estimates the **median house value** based on various features such as income, housing age, population, and geographical factors.

This project demonstrates the **complete machine learning workflow**, including data exploration, preprocessing, model training, and evaluation.

---

## 📊 Dataset

The dataset used in this project is the **California Housing Dataset**, which contains housing data collected from the **1990 California census**.

### Features in the dataset

| Feature | Description |
|------|-------------|
| MedInc | Median income in block group |
| HouseAge | Median house age in block group |
| AveRooms | Average number of rooms per household |
| AveBedrms | Average number of bedrooms per household |
| Population | Block group population |
| AveOccup | Average number of household members |
| Latitude | Latitude coordinate |
| Longitude | Longitude coordinate |

### Target Variable
- **MedHouseVal** → Median house value for California districts

---

## ⚙️ Project Workflow

### 1️⃣ Data Loading
- Import the dataset using Python libraries.

### 2️⃣ Exploratory Data Analysis (EDA)
- Understand dataset structure
- Check missing values
- Analyze statistical summary of features

### 3️⃣ Data Visualization
Visualizations help understand patterns and relationships between variables.

Examples include:
- Histograms
- Scatter plots
- Correlation heatmaps

### 4️⃣ Data Preprocessing
- Feature selection
- Data cleaning
- Feature scaling (if required)

### 5️⃣ Model Building
A **Linear Regression** model is used to learn the relationship between independent variables and house prices.

### 6️⃣ Model Training
The dataset is split into:
- **Training Data**
- **Testing Data**

The model is trained on the training dataset.

### 7️⃣ Model Evaluation

The model performance is evaluated using regression metrics:

- **Mean Squared Error (MSE)**

These metrics help measure how well the model predicts housing prices.

---

## 🛠️ Technologies Used

- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  
- Jupyter Notebook  

---
