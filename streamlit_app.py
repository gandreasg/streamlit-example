
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static

import streamlit as st
import pandas as pd
import joblib, os
import plotly.express as px
import numpy as np
from train_model import get_x_columns

# Load trained model
model = joblib.load('/Users/andreasgeorgiou/Desktop/BAZARAKI/random_forest_model.pkl')

file_path = os.path.join('/Users/andreasgeorgiou/Desktop/BAZARAKI/', 'bazaraki.csv')
df = pd.read_csv(file_path)

# Drop rows with missing values
df.dropna(inplace=True)


st.set_page_config(page_title="Bazar",
                   page_icon=":bar_chart",
                   layout="wide")

st.sidebar.header("Please filter here:")

# Create expandable sections for each filter
with st.sidebar.expander("Brand"):
    Brand = st.multiselect(
        "Select the Brand:", 
        options=df["Brand"].unique(),
        default=df["Brand"].unique()
    )

with st.sidebar.expander("Model"):
    Model = st.multiselect(
        "Select the Model:", 
        options=df["Model"].unique(),
        default=df["Model"].unique()
    )

with st.sidebar.expander("Transmission"):
    Transmission = st.multiselect(
        "Select the Transmission:", 
        options=df["Transmission"].unique(),
        default=df["Transmission"].unique()
    )

with st.sidebar.expander("Fuel"):
    Fuel = st.multiselect(
        "Select the Fuel:", 
        options=df["Fuel"].unique(),
        default=df["Fuel"].unique()
    )


# Calculate the initial min and max price and year values
min_price = df["Price"].min()
max_price = df["Price"].max()
min_year = df["Year"].min()
max_year = df["Year"].max()

# Create slider widgets for price and year
selected_min_price, selected_max_price = st.sidebar.slider(
    "Select Price Range",
    min_value=min_price,
    max_value=max_price,
    value=(min_price, max_price)
)

selected_min_year, selected_max_year = st.sidebar.slider(
    "Select Year",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year)
)

# Apply filters to create df_selection
df_selection = df.query(
    "Brand.isin(@Brand) & Model.isin(@Model) & Transmission.isin(@Transmission) & "
    "Fuel.isin(@Fuel) & Price >= @selected_min_price & Price <= @selected_max_price & "
    "Year >= @selected_min_year & Year <= @selected_max_year"
)

st.title(":bar_chart: BazarHacki")
st.markdown("##")

average_price = round(df_selection["Price"].mean(), 2)

left_column, middle_column, right_column = st.columns(3)
with right_column:
    st.subheader("Average Price")
    st.subheader(f"EUR {average_price}")

st.dataframe(df_selection)















# Create a scatter plot for Year vs Price with a trend line
fig = px.scatter(df_selection, x="Year", y="Price", title="Year vs Price",
                 trendline="ols")  # Add trendline="ols" for linear regression trendline
st.plotly_chart(fig)















# Perform OLS prediction
st.subheader("Car Specification for Prediction")
input_year = st.number_input("Enter Year:")
input_kilometers = st.number_input("Enter Kilometers:")

# Create dropdown menus for Brand, Model, Transmission, and Fuel
brands = df["Brand"].unique()
input_brand = st.selectbox("Select Brand:", brands)

models = df[df["Brand"] == input_brand]["Model"].unique()
input_model = st.selectbox("Select Model:", models)

transmissions = df[(df["Brand"] == input_brand) & (df["Model"] == input_model)]["Transmission"].unique()
input_transmission = st.selectbox("Select Transmission:", transmissions)

fuels = df[(df["Brand"] == input_brand) & (df["Model"] == input_model) &
           (df["Transmission"] == input_transmission)]["Fuel"].unique()
input_fuel = st.selectbox("Select Fuel:", fuels)

# Ensure input data matches structure used for training
input_data = pd.DataFrame({
    "const": [1],
    "Year": [input_year],
    "Km": [input_kilometers],
    "Brand_" + input_brand: [1],
    "Model_" + str(input_model): [1],
    "Transmission_" + str(input_transmission): [1],
    "Fuel_" + str(input_fuel): [1]
})

# Get X_columns
X_train_columns = get_x_columns()

# Make sure input_data columns match X_train columns
input_data = input_data.reindex(columns=X_train_columns, fill_value=0)

# Handle potential missing or infinite values
if input_data.isnull().any().any() or not np.isfinite(input_data).all().all():
    st.write("Warning: Missing or infinite values detected in input data!")

predicted_price = model.predict(input_data)

st.subheader("Predicted Price:")
st.markdown(f"<h1 style='text-align: center; color: red;'>€ {format(predicted_price[0], ',.0f')}</h1>", unsafe_allow_html=True)



















# Preprocess the entire dataset to match the training data structure
X_all = df[["Year", "Km", "Brand", "Model", "Transmission", "Fuel"]]
X_all = pd.get_dummies(X_all, columns=["Brand", "Model", "Transmission", "Fuel"], drop_first=True)
X_all = X_all.reindex(columns=X_train_columns, fill_value=0)
df['Predicted_Price'] = model.predict(X_all)
df['Difference'] = df['Predicted_Price'] - df['Price']


st.subheader("Filter Undervalued Vehicles by Price")
min_selection_price = st.number_input("Enter Minimum Price", min_value=int(df["Price"].min()), max_value=int(df["Price"].max()), value=int(df["Price"].min()))
max_selection_price = st.number_input("Enter Maximum Price", min_value=int(df["Price"].min()), max_value=int(df["Price"].max()), value=int(df["Price"].max()))


filtered_df = df[(df['Price'] >= min_selection_price) & (df['Price'] <= max_selection_price)]
top_10_undervalued = filtered_df[filtered_df['Difference'] > 0].nlargest(10, 'Difference').reset_index(drop=True)


# Display top 10 undervalued vehicles
top_10_undervalued.index = top_10_undervalued.index + 1
st.dataframe(top_10_undervalued)


# Load the area coordinates for the heatmap
area_coordinates_df = pd.read_csv('/Users/andreasgeorgiou/Desktop/BAZARAKI/area_coordinates.csv')  
print(type(area_coordinates_df))
print(area_coordinates_df.head())

heatmap_data = area_coordinates_df[["Latitude", "Longitude"]].values.tolist()

# Generate the heatmap
m = folium.Map(location=[35.1264, 33.4299], zoom_start=9)
HeatMap(heatmap_data).add_to(m)


# Display the heatmap in Streamlit
st.header("Listings Heatmap")
st.write("This heatmap represents the density of listings based on the areas in the dataset.")
folium_static(m, width=1000, height=600)
