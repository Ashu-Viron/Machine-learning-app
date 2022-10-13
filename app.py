from re import S
import streamlit as st
from utils import PrepProcesor,columns

import numpy as np
import pandas as pd
import joblib 

model=joblib.load('xgbpipe.joblib')

st.title('Did they survive :ship:')
passengerid=st.text_input('Input passenger id','12345')
passengereclass=st.select_slider('Choose passenger Class',[1,2,3])

name=st.text_input('Input the passenger name ','John Smith')
gender=st.select_slider('Select gender',['male','female'])
age=st.slider('input age',0,100)
sibsp=st.slider('input siblings',0,10)
parch=st.slider('Input parents/children',0,2)
ticketid=st.number_input('Ticket number',123345)
fare=st.number_input('Fare amount',0,1000)
cabin=st.text_input('Enter cabin','CS2')
embarked=st.selectbox('Choose embarkation point',['S','C','Q'])

def predict():
    row=np.array([passengerid,passengereclass,name,gender,age,sibsp,parch,ticketid,fare,cabin,embarked])
    x=pd.DataFrame([row],columns=columns)
    prediction=model.predict(x)[0]

    if prediction==1:
        st.succes('Passenger survived :thumbsup:')
    else:
        st.error('Passenger did not survive :thumbsdown:')

st.button('Predict',on_click=predict)