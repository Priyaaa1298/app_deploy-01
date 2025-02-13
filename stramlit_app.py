import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Streamlit UI
st.title("Temperature Data Analysis and Prediction")

# Upload CSV File
uploaded_file = st.file_uploader("https://drive.google.com/file/d/1xSVx4AGebE0C_2K9ZP9aZ2Kq9df_qBnn/view?usp=drive_link", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.write(df.head())
    st.log("Data preview displayed")
    
    # Data Info
    st.write("### Data Description")
    st.write(df.describe())
    st.write(df.info())
    st.log("Data description displayed")
    st.write("Missing Values:")
    st.write(df.isnull().sum())
    st.write("Duplicate Rows:", df.duplicated().sum())
    
    # Boxplot
    st.write("### Boxplot")
    fig, ax = plt.subplots(figsize=(16, 8))
    sns.boxplot(data=df, ax=ax)
    st.pyplot(fig)
    st.log("Boxplot displayed")
    
    # Min-Max Scaling
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(df)
    df_normalized = pd.DataFrame(normalized_data, columns=df.columns)
    st.write("### Normalized Data")
    st.write(df_normalized.head())
    st.log("Data normalized")
    
    # Outlier Removal using IQR
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_cleaned = df[(df >= lower_bound) & (df <= upper_bound)]
    st.write("### Data after Outlier Removal")
    st.write(df_cleaned.head())
    st.log("Outliers removed")
    
    # Handling Missing Values
    df_filled = df_cleaned.fillna(df_cleaned.mean())
    st.write("### Data after Handling Missing Values")
    st.write(df_filled.isna().sum())
    st.log("Missing values handled")
    
    # Correlation Heatmap
    corr_matrix = df_filled.corr()
    st.write("### Correlation Matrix")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
    st.pyplot(fig)
    st.log("Correlation matrix displayed")
    
    # Model Training
    features = ['ambient', 'u_d', 'u_q', 'i_d', 'i_q', 'pm', 'stator_winding']
    target = 'motor_speed'
    X = df_filled[features]
    y = df_filled[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    st.write("### Linear Regression Results")
    st.write(f"Mean Squared Error: {mse:.4f}")
    st.write(f"Root Mean Squared Error: {rmse:.4f}")
    st.write(f"R² Score: {r2:.4f}")
    st.log("Linear regression model trained and evaluated")
    
    # Model Selection (RandomForest & GradientBoosting)
    models = {
        "RandomForest": RandomForestRegressor(random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42)
    }
    
    model_results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        model_results[name] = {"RMSE": rmse, "R²": r2}
        st.log(f"{name} model trained and evaluated")
    
    st.write("### Model Comparison")
    st.write(pd.DataFrame(model_results))
    st.log("Model comparison displayed")
