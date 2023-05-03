import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression



# df = pd.ExcelFile("STL1.zls")
# data=df.parse("STL1")
# print(data.head(10))
#
# for col in data.head:
#     print(col)
#     print('Skew :', round(data[col].skew(), 2))
#     plt.figure(figsize = (15, 4))
#     plt.subplot(1, 2, 1)
#     data[col].hist(grid=False)
#     plt.ylabel('count')
#     plt.subplot(1, 2, 2)
#     plt.show()


# Header
st.header("NBA Playoff Stats")

# pd.set_option("display.precision", 2)
#
@st.cache_data()
def data():
    data1 = pd.read_excel("Playoffs1.xlsx")

    return data1
# importing data
df = data()
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
    st.dataframe(df.describe())


# Scatter Plot
st.sidebar.subheader('Scatter Plot')
# Select widget

select1 = st.sidebar.selectbox(label='Y axis', options=Columns)
graph = sns.relplot(data=df, x='Rk', y=select1).set(title='Players Stat Data')
st.pyplot(graph)


fig0 = plt.figure(figsize=(9, 7))
select3 = st.sidebar.selectbox(label='Y1 axis', options=Columns)
sns.regplot(data=df, x='Rk', y=select3).set(title='Players Stat Data')
st.pyplot(fig0)



st.sidebar.subheader('Histogram Plot')
fig1 = plt.figure(figsize=(9, 7))
select2 = st.sidebar.selectbox(label='Columns', options=Columns)
sns.histplot(data=df, x=select2, binwidth=1)
plt.title('Histogram Data')
st.pyplot(fig1)


st.sidebar.subheader('Box Plot')
fig2 = plt.figure(figsize=(9, 7))
select3 = st.sidebar.selectbox(label='Features', options=Columns)
sns.boxplot(y=df[select3])
plt.title('Box Plot Data')
st.pyplot(fig2)


st.subheader('Heatmap')
fig3 = plt.figure(figsize=(28, 26))
sns.heatmap(df[Columns], annot=True, linewidths=.00000005)
st.pyplot(fig3)


st.header("PDA")


X = np.array(df[['FGA', 'MP']])
Y = np.array(df['PTS'])

model = LinearRegression().fit(X, Y)

def predict_points(fga, minutes):
    new_data = np.array([[fga, minutes]])
    predicted_points = model.predict(new_data)
    return predicted_points[0]


st.write("Enter the total field goal attempts and minutes played to predict points per game:")
fga = st.number_input("Total field goal attempts")
minutes = st.number_input("Minutes played")

if st.button("Predict"):
    predicted_points = predict_points(fga, minutes)
    st.write("Predicted points per game: ", predicted_points)




















