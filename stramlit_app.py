import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Install required packages
st.write("Installing required packages...")
import os
os.system("pip install streamlit pandas matplotlib seaborn numpy scikit-learn")

# Streamlit UI
st.title("Electric motor Speed Prediction")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Overview")
    st.write(df.head())
    
    # Data Description
    st.write("### Dataset Description")
    st.write(df.describe())
    
    # Data Info
    st.write("### Dataset Info")
    st.write(df.info())
    
    # Missing Values
    st.write("### Missing Values")
    st.write(df.isnull().sum())
    
    # Boxplot
    st.write("### Boxplot of Data")
    fig, ax = plt.subplots(figsize=(16, 8))
    sns.boxplot(data=df, ax=ax)
    st.pyplot(fig)
    
    # Min-Max Normalization
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(df.select_dtypes(include=[np.number]))
    normalized_df = pd.DataFrame(normalized_data, columns=df.select_dtypes(include=[np.number]).columns)
    
    # Handling Missing Values
    df_cleaned = df.fillna(df.mean())
    
    # Correlation Matrix
    corr_matrix = df_cleaned.corr()
    st.write("### Correlation Matrix")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
    st.pyplot(fig)
    
    # Feature Selection
    X = df[['u_d','u_q', 'i_d', 'pm','stator_winding','torque','coolant']]
    y = df['motor_speed']

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model Selection
    model_choice = st.selectbox("Select a Model", ["Linear Regression", "Ridge", "Lasso", "RandomForest", "GradientBoosting"])
    
    if model_choice == "Linear Regression":
        model = LinearRegression()
    elif model_choice == "Ridge":
        model = Ridge(alpha=1.0)
    elif model_choice == "Lasso":
        model = Lasso(alpha=0.1)
    elif model_choice == "RandomForest":
        model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    else:
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    
    # Train Model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Model Performance
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    st.write(f"**Mean Squared Error:** {mse:.4f}")
    st.write(f"**Root Mean Squared Error:** {rmse:.4f}")
    st.write(f"**R² Score:** {r2:.4f}")
    
    # Predictions
    st.write("### Actual vs Predicted")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r', lw=2)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    st.pyplot(fig)


    # Convert y_test to DataFrame for comparison
    df_results = pd.DataFrame({'Actual Motor Speed': y_test, 'Predicted Motor Speed': y_pred})

    # Display predicted motor speed
    st.write("### Predicted Motor Speed")
    st.write(df_results.head(20))  # Display first 20 rows

    


    
