import streamlit as st
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

option = st.sidebar.selectbox("Choose Analysis:", ["Predict Production", "Model Performance"])
st.title(" Crop Production Predictor")
st.markdown("Predict total crop production using area harvested, yield, and year.")
area = st.number_input("Area Harvested (ha)", min_value=0.0, value=47832000.0, step=10000.0)
yield_t = st.number_input("Yield (kg/ha)", min_value=0.0, value=254367.0, step=1000.0)
year = st.number_input("Year", min_value=1950, max_value=2025, value=2022)
df = pd.read_csv("C:\\Users\\Admin\\Desktop\\vs\\croppro.xls") # .xls should use read_excel
X = df[['Area harvested', 'Yield', 'Year']]
y = df[['Production']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)
if option == "Predict Production":
    if st.button("Predict"):
        input_data = pd.DataFrame([[area, yield_t, year]], columns=['Area harvested', 'Yield', 'Year'])
        prediction = model.predict(input_data)[0]
        st.success(f" Estimated Crop Production: **{prediction:.2f} tons**")

elif option == "Model Performance":
    with st.expander(" Model Evaluation"):
        y_pred = model.predict(X_test)
        st.write(f"**R² Score:** {r2_score(y_test, y_pred):.4f}")
        st.write(f"**RMSE:** {mean_squared_error(y_test, y_pred):.2f} tons")
