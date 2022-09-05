import streamlit as st
import datetime

def feed():
    file1 = open("settings.txt", 'a')
    currenttime = datetime.datetime.today()
    val = st.select_slider("Rating of the App", options=[1,2,3,4,5])

    if st.button("Submit"):
        file1.write(str(currenttime))
        file1.write("\n")
        file1.write(str(val))
        file1.write("\n")
        file1.close()