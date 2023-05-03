import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression




# Header
st.header("NBA Playoff Stats")



# Importing dataset
@st.cache_data()
def data():
    data1 = pd.read_excel("Playoffs1.xlsx")

    return data1

df = data()

# Isolating int and float types within dataset
Columns = df.select_dtypes(['float64', 'int64']).columns

# EDA
print(df.head())
print(df.tail())
print(df.shape)
print(df.columns)
print(df.nunique())
print(df.info())
print(df.describe())


# check box wigdet

checkbox = st.sidebar.checkbox('Reveal main Data')

st.sidebar.subheader('EDA')
checkbox1 = st.sidebar.checkbox('Reveal data head')
checkbox2 = st.sidebar.checkbox('Reveal data tail')
checkbox3 = st.sidebar.checkbox('Reveal data info')


if checkbox:
    # Displaying Dataset
    st.dataframe(df)

if checkbox1:
    # Displaying Dataset - Head
    st.dataframe(df.head())


if checkbox2:
    # Displaying Dataset - Tails
    st.dataframe(df.tail())

if checkbox3:
    # Displaying dataset details
    st.dataframe(df.describe())


# Scatter Plot header
st.sidebar.subheader('Scatter Plot')

# Sidebar widget
select1 = st.sidebar.selectbox(label='Y axis', options=Columns)
select5 = st.sidebar.selectbox(label='X axis', options=Columns)
# Scatter Plot data
graph = sns.relplot(data=df, x=select5, y=select1).set(title='Players Stat Data')

# Loop function, assigning column 'Ranks' their respective string(Players)
for i in range(df.shape[0]):
    plt.text(x=df[select5][i]+0.3, y=df[select1][i], s=df.Player[i], fontdict=dict(size=5))

# Display data
st.pyplot(graph)


# Regression Plot data
st.sidebar.subheader('Regression Plot')
fig0 = plt.figure(figsize=(9, 7))
select3 = st.sidebar.selectbox(label='Y1 axis', options=Columns)
select4 = st.sidebar.selectbox(label='X1 axis', options=Columns)
sns.regplot(data=df, x=select4, y=select3).set(title='Players (Reg) Stat Data')
st.pyplot(fig0)


# Histogram data
st.sidebar.subheader('Histogram Plot')
fig1 = plt.figure(figsize=(9, 7))
select2 = st.sidebar.selectbox(label='Columns', options=Columns)
sns.histplot(data=df, x=select2, binwidth=1, kde=True)
plt.title('Histogram Data')
st.pyplot(fig1)


# Box Plot data
st.sidebar.subheader('Box Plot')
fig2 = plt.figure(figsize=(9, 7))
select3 = st.sidebar.selectbox(label='Features', options=Columns)
sns.boxplot(y=df[select3])
plt.title('Box Plot Data')
st.pyplot(fig2)


# Heatmap data
st.subheader('Heatmap')
fig3 = plt.figure(figsize=(20, 18))
sns.heatmap(df[Columns].corr(), annot=True, linewidths=.5, cmap='Reds')
st.pyplot(fig3)





# PDA
st.header("PDA")

# Data to predict with
X = np.array(df[['FGA', 'MP']])
# Data that is being predicted
Y = np.array(df['PTS'])

# Sampling data
model = LinearRegression().fit(X, Y)


# Prediction  Function
def predict_points(fga, min):
    new_data = np.array([[fga, min]])  # new data values
    predicted_points = model.predict(new_data)  # defining new values with prediction statement based on sample data
    return round(predicted_points[0], 1)  # returning values

# inputs
st.write("Enter the total field goal attempts and minutes played to predict points per game:")
fga = st.number_input("Field goal attempts")
min = st.number_input("Minutes played")


# Output
if st.button("Predict"):
    predicted_points = predict_points(fga, min)
    st.write("Predicted points per game: ", predicted_points)




















