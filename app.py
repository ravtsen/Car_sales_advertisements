#####################
# Import libraries
#####################

import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
from PIL import Image

#####################
# Page Title 
#####################

st.title ('Second Hand Cars Marketing')

image = Image.open('used_cars.jpeg')
st.image(image, use_column_width=True)

st.markdown("""
This app performs simple webscraping of vehicles marketing data!<br>
User Input Features allows to compare different models, compare car characteristics.<br>     
Below you can find some useful graphs with data analyses.
* **Python libraries:** pandas, plotly, mumpy, streamlit, pillow
""")

#####################
# Data preprocessing
#####################

# Define a function to load data from a file.
@st.cache_data
def load_data(file_path_1, file_path_2):
    #Reading the file and storing it to df
    try:
        df = pd.read_csv(file_path_1)
    except:
        df = pd.read_csv(file_path_2)
    return df

df = load_data('C:/Users/count/Project_sprint_6/vehicles_us.csv', 'https://practicum-content.s3.us-west-1.amazonaws.com/datasets/vehicles_us.csv')

# Replace the missing values in `'model_year'`, `'cylinders'`, 'paint_color'` and `'is_4wd'` with the string `'unknown'`. 
columns_to_replace = ['model_year', 'cylinders', 'paint_color', 'is_4wd']
for column in columns_to_replace:
    df[column] = df[column].fillna('unknown') 

# For futher analyses required to keep 'odometer' with float dtype.
# Replace the missing values in `'odometer'` with `'-1'`. 
data_to_replace = ['odometer']
for column in data_to_replace:
    df[column] = df[column].fillna('-1')  

# Implicit duplicates
# Functions for replacing implicit duplicates in 'model' and 'type' columns
def replace_wrong_models (wrong_models, correct_models):
    for wrong_model in wrong_models:
        df['model'] = df ['model'].replace(wrong_model, correct_model)

def replace_wrong_type (wrong_types, correct_type):
    for wrong_type in wrong_types:
        df['type'] = df ['type'].replace(wrong_types, correct_type) 

# Removing implicit duplicates. 
# Removing specific details (e.g., trim level) and keeping only the base model name. 

wrong_models = ['chevrolet camaro', 'chevrolet camaro lt coupe 2d']
correct_model = 'chevrolet camaro' 
replace_wrong_models (wrong_models, correct_model) 

wrong_models = ['ford f150', 'ford f150 supercrew cab xlt']
correct_model = 'ford f-150'
replace_wrong_models (wrong_models, correct_model) 

wrong_models = ['chevrolet silverado 1500', 'chevrolet silverado 1500 crew', 'chevrolet silverado 2500hd', 
                'chevrolet silverado 3500hd']
correct_model = 'chevrolet silverado' 
replace_wrong_models (wrong_models, correct_model)

wrong_models = ['ford f-250 sd', 'ford f-250 super duty', 'ford f250', 'ford f250 super duty']
correct_model = 'ford f-250'
replace_wrong_models (wrong_models, correct_model) 

wrong_models = ['ford f-350 sd', 'ford f350', 'ford f350 super duty']
correct_model = 'ford f-350'
replace_wrong_models (wrong_models, correct_model) 

wrong_models = ['ford focus', 'ford focus se']
correct_model = 'ford focus'
replace_wrong_models (wrong_models, correct_model) 

wrong_models = ['ford fusion', 'ford fusion se']
correct_model = 'ford fusion'
replace_wrong_models (wrong_models, correct_model) 

wrong_models = ['ford mustang', 'ford mustang gt coupe 2d']
correct_model = 'ford mustang'
replace_wrong_models (wrong_models, correct_model) 

wrong_models = ['gmc sierra 1500', 'gmc sierra 2500hd']
correct_model = 'gmc sierra'
replace_wrong_models (wrong_models, correct_model) 

wrong_models = ['honda civic', 'honda civic lx']
correct_model = 'honda civic'
replace_wrong_models (wrong_models, correct_model) 

wrong_models = ['jeep grand cherokee', 'jeep grand cherokee laredo']
correct_model = 'jeep grand cherokee'
replace_wrong_models (wrong_models, correct_model) 

wrong_models = ['jeep wrangler', 'jeep wrangler unlimited']
correct_model = 'jeep wrangler'
replace_wrong_models (wrong_models, correct_model) 

wrong_models = ['nissan frontier', 'nissan frontier crew cab sv']
correct_model = 'nissan frontier'
replace_wrong_models (wrong_models, correct_model) 

wrong_models = ['toyota camry', 'toyota camry le']
correct_model = 'toyota camry'
replace_wrong_models (wrong_models, correct_model)

wrong_types = ['truck', 'pickup']
correct_type = 'pickup' 
replace_wrong_type (wrong_types, correct_type) 

# For more convinient table structure `'model'` column should be moved to the beginning of the table
df = df[['model', 'price', 'model_year', 'condition', 'cylinders', 'fuel',
       'odometer', 'transmission', 'type', 'paint_color', 'is_4wd',
       'date_posted', 'days_listed']]

# Car age calculation
# Convert 'date_posted' column to datetime format
df['date_posted'] = pd.to_datetime(df['date_posted'])

# Extract publication year
df['year_posted'] = df['date_posted'].dt.year

# Convert 'year_posted' and 'model_year' columns to numeric type
df['year_posted'] = pd.to_numeric(df['year_posted'], errors='coerce')
df['model_year'] = pd.to_numeric(df['model_year'], errors='coerce')

# Calculate the age of the car and adjust for days listed
df['car_age'] = df['year_posted'] - df['model_year']
df['car_age'] -= df['days_listed'] / 365
df['car_age'] = df['car_age'].round(2)

# Define the function to replace negative numbers with 0
def replace_negative_with_zero(number):
    if number < 0:
        return 0
    else:
        return number

# Apply the function to the column 'car_age' in the DataFrame
df['car_age'] = df['car_age'].apply(replace_negative_with_zero)
# Drop rows with NaN values only in the 'car_age' column
df = df.dropna(subset=['car_age'])

# Sort cars by the age
df = df.sort_values(by='car_age', ascending=True)

# divide the cars into groups by odometer readings : low - medium - high - unknown
# Convert `'odometer'` to numeric data type. 
df['odometer'] = pd.to_numeric(df['odometer'])

# Define bins
bins = [-1, 0, 50000, 100000, float("inf")]

# Define labels for each group
labels = ['unknown', 'low', 'medium', 'high']

# Create a new column 'mileage' with bin labels based on the 'odometer' values
df['mileage'] = pd.cut(df['odometer'], bins=bins, labels=labels, right=False)

# Add column manufacturer to the table
# Extract the first word from 'model' column to the 'manufacturer' column
df['manufacturer'] = df['model'].str.split().str[0]
df = df.reset_index(drop=True)

# Replace 'unknown' with NaN
df['cylinders'] = df['cylinders'].replace('unknown', np.nan)

# Function to filter df based on selected columns and excludes certain values
def filter_df(df, column, excluded_values):
    filtered_df = df[~df[column].isin(excluded_values)]
    return filtered_df

#####################
# Sidebar settings
#####################

# Sidebar header
st.sidebar.header('User Input Features')

# Choose model
model = sorted(df.model.unique())
selected_model = st.sidebar.multiselect('Select model', model, model)

# Choose car characteristics
selected_columns = [ 'cylinders', 'transmission', 'paint_color', 'is_4wd', 'odometer',
       'car_age', 'days_listed']
always_selected_columns = ['model', 'manufacturer', 'price', 'mileage','condition', 'fuel', 'type']
characteristics = list(df[selected_columns])
selected_characteristics = st.sidebar.multiselect('Select car characteristics', characteristics, characteristics)

# Choose condition
condition = sorted(df['condition'].unique().tolist())
selected_condition = st.sidebar.multiselect('Select condition', condition, condition)

# Choose type
type = sorted(df['type'].unique().tolist())
selected_type = st.sidebar.multiselect('Select type', type, type)

# Choose fuel
fuel = sorted(df['fuel'].unique().tolist())
selected_fuel = st.sidebar.multiselect('Select fuel', fuel, fuel)

# Choose mileage
mileage_data = sorted(df['mileage'].unique().tolist())
selected_group = st.sidebar.multiselect('Select mileage', mileage_data, mileage_data)

st.sidebar.markdown("""
Mileage groups:
* **Unknown** - no odometer readings ('-1' in the rable)
* **Low** - between 0 and 50 
* **Medium** - between 50 and 100 
* **High** - higher then 100
""")

#####################
# Filtering data
#####################

# Define a function to filter data based on selected columns
def filter_data(df, selected_columns):
    # Filter the DataFrame to include only selected columns
    filtered_df = df[selected_columns]
    return filtered_df

filtered_df = filter_data(df, always_selected_columns + selected_characteristics)

selected_df = filtered_df[filtered_df['model'].isin(selected_model) 
                          & filtered_df['mileage'].isin(selected_group)
                          & filtered_df['condition'].isin(selected_condition) 
                          & filtered_df['type'].isin(selected_type)
                          & filtered_df['fuel'].isin(selected_fuel)] 

# Dataframe show up  
st.header('Display car advertising data')
st.write('Data Dimension: ' + str(selected_df.shape[0]) + ' rows and ' + str(selected_df.shape[1]) + ' columns.')
st.dataframe(selected_df)

#####################
# Data analyses
#####################

if 'section_state' not in st.session_state:
    st.session_state['section_state'] = [False, False, False, False]

if st.button('**Car price dependence on odometer readings and car age**'):
    st.session_state['section_state'][0] = not st.session_state['section_state'][0]

if st.session_state['section_state'][0]:

    st.header('Car price dependence on odometer readings and car age')

    # Plot scatter plot using Plotly Express
    fig = px.scatter(df, x=df['car_age'], y=df['price'], color=df['mileage'], title='Car price dependence on odometer readings and car age')
    # Display the plot in Streamlit
    st.plotly_chart(fig)

    st.header('Car price range and age range in the groups')
       
    # Median price by mileage
    df.groupby('mileage')['price'].median()
    # Average price by mileage
    df.groupby('mileage')['price'].mean().round(2)
    # Median car age by mileage
    df.groupby('mileage')['car_age'].median()
    # Average car age by mileage
    df.groupby('mileage')['car_age'].mean().round(2)

    # Mileage dataframe
    # Group by 'mileage' and count occurrences
    mileage_df1 = df.groupby('mileage')['price'].median().reset_index(name='median_price')
    mileage_df2 = df.groupby('mileage')['price'].mean().round(2).reset_index(name='average_price')
    mileage_df3 = df.groupby('mileage')['car_age'].median().reset_index(name='median_car_age')
    mileage_df4 = df.groupby('mileage')['car_age'].mean().round(2).reset_index(name='average_car_age')

    # New df creation and merge
    df_merged = pd.merge(mileage_df1, mileage_df2[['mileage', 'average_price']], on='mileage', how='left')
    df_merged = pd.merge(df_merged, mileage_df3[['mileage', 'median_car_age']], on='mileage', how='left')
    df_merged = pd.merge(df_merged, mileage_df4[['mileage', 'average_car_age']], on='mileage', how='left')

    show_nan_data = st.checkbox('Show unknown value')

    if show_nan_data: 
        st.dataframe(df_merged)
        # Create a histogram using Plotly Express
        fig = px.histogram(df_merged, x='mileage', y='average_car_age', color='median_price', title='Average car age by mileage', labels={'mileage': 'mileage', 'average_car_age': 'average_car_age'})
        # Update y-axis name
        fig.update_yaxes(title_text='average_car_age')
        # Display the plot in Streamlit
        st.plotly_chart(fig)

    else:
        df_merged_filtered = filter_df(df_merged, 'mileage', ['unknown'])
        st.dataframe(df_merged_filtered)
        # Create a histogram using Plotly Express
        fig = px.histogram(df_merged_filtered, x='mileage', y='average_car_age', color='median_price', title='Average car age by mileage', labels={'mileage': 'mileage', 'average_car_age': 'average_car_age'})
        # Update y-axis name
        fig.update_yaxes(title_text='average_car_age')
        # Display the plot in Streamlit
        st.plotly_chart(fig)
            
    st.write("""
    Having compared cars by mileage, car age and price, we can draw the following conclusions:
            
    1. Second hand cars with higher mileage are cheaper then the same cars with lower odometer readings. Car age is impoirtant characteristic that cannot be excluded when considering car price and mileage. Older cars mostly have higher mileage and are cheeper.
    2. There are several expensive old cars with in medium milleage group. It is collectible cars.  
    3. Market tendencion is that there are more older second hand cars with higher mileage. 
                    
    Important to note that there is a big 'unknown' group with unknown mileage. It is suggested to the marketing group to add this field as 'required' and fill in all the missed data.
    """)

if st.button('**Car types popularity**'):
    st.session_state['section_state'][1] = not st.session_state['section_state'][1]

if st.session_state['section_state'][1]:

    st.header('Cars quantity of each type')

    # Group by type
    quantity_by_type = df.groupby(['type'])['price'].size().reset_index(name='quantity')
    # Calculate percentage and add it as a new column
    # Total count of rows in the DataFrame
    total_count = df.shape[0] 
    quantity_by_type['percentage'] = (quantity_by_type['quantity'] / total_count) * 100
    quantity_by_type['percentage'] = quantity_by_type['percentage'].round(2)
    # Sort values by percentage
    quantity_by_type = quantity_by_type.sort_values(by='percentage', ascending=False).reset_index(drop=True)

    st.dataframe(quantity_by_type)

    st.write("""
    Based on group by type table the offer of 3 car types takes up more than 80% of the table. For futher analyses we continue to check and compare only these 3 car types: SUV, pickup, sedan. 

    Pickup car market share is the highest - 37.58%
    """)

    # Updated df
    # Specify car types we are going to analyse
    included_types = ['SUV', 'pickup', 'sedan']
    # Filter df.sorted based on the included types
    df_by_selected_type = df[df['type'].isin(included_types)]

    # Group df by type and manufacturer
    grouped_df_by_type = df_by_selected_type.groupby(['type', 'manufacturer'])['price'].count().reset_index(name='quantity')
    # Sort df by type and quantity in descending order
    grouped_df_by_type = grouped_df_by_type.sort_values(by=['type', 'quantity'], ascending=[True,False])
    
    # Bar chart - Car distribution by car type
    fig = px.bar(grouped_df_by_type, x='type', y='quantity', color='manufacturer', title='Car distribution by car type and manufacturer')
    # Show the plot
    st.plotly_chart(fig)

    # Group df by 'manufacturer' and 'type' -> average price
    df_grouped_by_avg_price = df_by_selected_type.groupby(['type', 'manufacturer'])['price'].mean().round(2).reset_index(name = 'average_price')
    # Sort df by average price within each group
    df_sorted_by_avg_price = df_grouped_by_avg_price.sort_values(by=['type', 'average_price'], ascending=[True,False])

    # Bar chart - Car price distribution by car type and manufacturer
    fig = px.bar(df_sorted_by_avg_price, x='type', y='average_price', color='manufacturer', title='Car price distribution by car type and manufacturer', barmode='group')
    # Show the plot
    st.plotly_chart(fig)

    st.write("""
    Pickup cars has wider range of prices depending on the manufacturer. Also most of the average prices are high. It is nessessary to see it in a separate chart.
    """)

    # Updated df for pickup cars only
    # Specify car types we are going to analyse
    included_type = ['pickup']
    # Filter df.sorted based on the included type
    df_pickup_only = df_grouped_by_avg_price[df_grouped_by_avg_price['type'].isin(included_type)]
    # Sort df_pickup_only by average price
    df_sorted = df_pickup_only.sort_values(by=['average_price'], ascending=[False]).reset_index(drop=True)

    # Bar chart - Pickup cars price distribution by manufacturer
    fig = px.bar(df_sorted, x='type', y='average_price', color='manufacturer', text='manufacturer', title='Pickup type car price distribution by car type and manufacturer', barmode='group')
    # Show the plot
    st.plotly_chart(fig)

    st.write("""
    Having compared car quantity and car average price by type and manufacturer, we can draw the following conclusions:

    1. Based on group by type table the offer of 3 car types takes up more than 80% of the table - SUV, pickup and sedan types. 
    Pickup car market share is the highest - 37.58%

    2. Each car type has it's own most popular manufacturers:
    * Jeep, Chevrolet and Ford in SUV car type
    * Chevrolet, Ford, RAM in pickup car type
    * Ford, Toyota, Chevrolet, Honda and Nissan in sedan car type
    
    3. Price distribution by manufacturer in SUV and sedan types mostly flat with some more expensive (prestigious) brands. 
    Pickup type cars price distribution is different. There are 8 manufacturers with high prices - RAM, Nissan, Chevrolet, GMC, Cadillac, Ford, Buick and Toyota. 2 manufacturers with mid price and 4 with a low one.
    Pickup cars are more expensive then other 2 popular types.
    """)

if st.button('**Car price dependence on condition**'):
    st.session_state['section_state'][2] = not st.session_state['section_state'][2]

if st.session_state['section_state'][2]:

    st.header('Car price dependence on condition')

    # Updated df for 3 selected types
    # Specify car types we are going to analyse
    included_types = ['SUV', 'pickup', 'sedan']
    # Filter df.sorted based on the included types
    df_by_selected_type = df[df['type'].isin(included_types)].reset_index()
    #Check and compare car distribution by car type and condition
    # Group df
    grouped_df_by_condition = df_by_selected_type.groupby(['type', 'condition'])['price'].count().reset_index(name='quantity')
    # Sort df
    grouped_df_by_condition = grouped_df_by_condition.sort_values(by=['type', 'quantity'], ascending=(True, False))
    # Calculate percentages
    total = grouped_df_by_condition['quantity'].sum()
    grouped_df_by_condition['percentage'] = (grouped_df_by_condition['quantity'] / total) * 100
    grouped_df_by_condition = grouped_df_by_condition.round(2).reset_index(drop=True)

    # Bar chart - Car distribution by car type and condition
    fig = px.bar(grouped_df_by_condition, x='type', y='quantity', text='percentage', color='condition', title='Car distribution by car type and condition', barmode='group')
    # Update layout to display percentages
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='inside')
    # Show the plot
    st.plotly_chart(fig)

    # Check and compare car price distribution by car type and condition.
    # Group df by 'condition' and 'type' -> average price
    df_grouped_by_cond = df_by_selected_type.groupby(['type', 'condition'])['price'].mean().round(2).reset_index(name = 'average_price')
    # Sort df by average price within each group
    df_sorted_by_cond = df_grouped_by_cond.sort_values(by=['type', 'average_price'], ascending=[True,False]).reset_index(drop=True)

    # Define a function to calculate the percentage within each group
    def calculate_percentage(group):
        group['percentage_within_group'] = (group['average_price'] / group['average_price'].sum()) * 100
        return group

    # Apply the function to each group and combine the results
    df_sorted_by_cond_percentage = df_sorted_by_cond.groupby('type').apply(calculate_percentage).round(2)
   
    # Bar chart 'Price distribution by car type and condition'
    fig = px.bar(df_sorted_by_cond_percentage, x='type', y='average_price', color='condition', text='percentage_within_group', title='Price distribution by car type and condition', barmode='group')
    # Update layout to display percentages
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='inside')
    # Show the plot
    st.plotly_chart(fig)

    st.write("""
    Following the comparison of car quantity and car average price by type and condition, we can draw the following conclusions:

    1. 87% of cars in the second hand market are in exellent and good condition. 

    2. According to price distribution by car type and condition bar chart average price decreases with condition deterioration. 
    * About 15-20% decrease for each degree.
    * For pickup - decrease from 'new condition' to 'like new condition' is about 40%
    """)

if st.button('**Transmission impact on market distribution and prices**'):
    st.session_state['section_state'][3] = not st.session_state['section_state'][3]

if st.session_state['section_state'][3]:

    st.header('Transmission impact on market distribution and prices')

    # Group df by transmission and manufacturer
    grouped_df_by_transmission_count = df.groupby(['transmission', 'manufacturer'])['price'].count().reset_index(name='quantity')
    # Sort df
    grouped_df_by_transmission_count = grouped_df_by_transmission_count.sort_values(by=['transmission', 'quantity'], ascending=(True, False))
    
    # Percentage of cars by transmission
    # Group df by transmission and manufacturer
    grouped_df_by_transmission_count_percentage = grouped_df_by_transmission_count.groupby(['transmission'])['quantity'].sum().reset_index(name='quantity')
    # Calculate the percentage within each group
    grouped_df_by_transmission_count_percentage['percentage'] = grouped_df_by_transmission_count_percentage['quantity'] / grouped_df_by_transmission_count_percentage['quantity'].sum() * 100
    # Reset index
    grouped_df_by_transmission_count_percentage = grouped_df_by_transmission_count_percentage.round(2).reset_index(drop=True)
    
    # Group df by transmission and manufacturer
    grouped_df_by_transmission_mean = df.groupby(['transmission', 'manufacturer'])['price'].mean().round(2).reset_index(name='average_price')
    # Sort df
    grouped_df_by_transmission_mean = grouped_df_by_transmission_mean.sort_values(by=['transmission', 'average_price'], ascending=(True, False))
    
    show_other = st.checkbox('Show other')

    if show_other: 
        st.dataframe(grouped_df_by_transmission_count_percentage)
        
        # Bar chart 'Car distribution by transmission and manufacturer'
        fig = px.bar(grouped_df_by_transmission_count, x='transmission', y='quantity', color='manufacturer', title='Car distribution by transmission and manufacturer', barmode='group')
        # Display the plot in Streamlit
        st.plotly_chart(fig)
        
        # Bar chart - 'Price distribution by car type and condition'
        fig = px.bar(grouped_df_by_transmission_mean, x='transmission', y='average_price', color='manufacturer', title='Price distribution by transmission and manufacturer', barmode='group')
        # Show the plot
        st.plotly_chart(fig)

    else:
        grouped_df_by_transmission_count_percentage_filtered = filter_df(grouped_df_by_transmission_count_percentage, 'transmission', ['other'])
        grouped_df_by_transmission_count_filtered = filter_df(grouped_df_by_transmission_count, 'transmission', ['other'])
        grouped_df_by_transmission_mean_filtered = filter_df(grouped_df_by_transmission_mean, 'transmission', ['other'])
        
        st.dataframe(grouped_df_by_transmission_count_percentage_filtered)
        
        # Bar chart 'Car distribution by transmission and manufacturer'
        fig = px.bar(grouped_df_by_transmission_count_filtered, x='transmission', y='quantity', color='manufacturer', title='Car distribution by transmission and manufacturer', barmode='group')
        # Display the plot in Streamlit
        st.plotly_chart(fig)

        # Bar chart - 'Price distribution by car type and condition'
        fig = px.bar(grouped_df_by_transmission_mean_filtered, x='transmission', y='average_price', color='manufacturer', title='Price distribution by transmission and manufacturer', barmode='group')
        # Show the plot
        st.plotly_chart(fig)

    st.write("""
    Once compared car quantity and car average price by transmission and manufacturer, we can gather:

    1. The majority of cars of second hand car market with automatic transmission. It's almost 91%.

    2. Pickup cars distribution by transmission is the same as full car types selection. The majority of cars are with automatic transmission.
  
    3. Cars with automatic transmission mostly more expensive then manual transmission (About 10% difference).
    We should not take into account 'other' data, since there are no many cars there and mostly it was not filled in correctly.
    """)
