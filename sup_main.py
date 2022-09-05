import streamlit as st
import datetime
import calendar
from sup_analysis import ana
from sup_predection import pred
from sup_feedback import feed
from streamlit_option_menu import option_menu
st.set_page_config(page_title='Superstore', layout='wide')

st.header("Superstore Sales Predictions & Analytics")
currenttime = datetime.datetime.today()
onlydate = currenttime.day
onlymonth = currenttime.strftime("%b")
onlyyear = currenttime.year
onlyday = calendar.day_name[currenttime.weekday()]
st.sidebar.info(f"""{onlyday} , {onlydate} {onlymonth} {onlyyear}""")

with st.sidebar:
    sel = option_menu(
        menu_title="Main Menu",
        options=["Analysis", "Predection", "Feedback1"],
        icons=["tv", "gear", "gear"],
        menu_icon="cast",
        #default_index=0,
    )

# sel = st.sidebar.radio("Select a Page",('Analysis', 'Predection'))

if sel=='Analysis':
    ana()
elif sel=='Predection':
    pred()
elif sel=='Feedback':
    feed()
