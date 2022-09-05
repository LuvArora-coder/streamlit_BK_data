import streamlit as st
import pandas as pd
import statsmodels.api as sm
from keras.models import load_model

def pred():
    col = st.columns(2)
    with col[0]:
        st_date = st.date_input("Select Start Date",pd.to_datetime('2022-05-01'))
    with col[1]:
        ed_date = st.date_input("Select End Date",pd.to_datetime('2022-05-28'))

    data = pd.read_csv("only_sales.csv")
    df_sup = data[['date','sales']]
    df_sup['date'] = pd.to_datetime(df_sup['date'],format='%Y/%m/%d')
    df_sup = df_sup.groupby('date').sum({
        'sales': lambda price: price.sum()})
    df_sup = df_sup['sales'].resample('D').sum()
    # st.write(df_sup)

    ## ARIMA for time series.
    # fit model
    #mod = sm.tsa.statespace.SARIMAX(df_sup,
    #                            order=(1,1,1),
     #                           seasonal_order= (1,1,0,12),
     #                          enforce_invertibility=False)

    mod= load_model("streamlit_BK_data/model1.h5")
    model_fit = mod.fit()

    # pred_uc = model_fit.forecast(steps=20)
    pred_uc_w = model_fit.get_prediction(start = pd.to_datetime(st_date), end = pd.to_datetime(ed_date), dynamic = False)
    pred_ci = pred_uc_w.conf_int()

    # st.line_chart(pred_ci)
    st.write("Predection form ", str(st_date), " to ", str(ed_date))
    st.line_chart(pred_uc_w.predicted_mean)
    st.write("Predection for Upcoming 14 days for which Data is not present")
    st.line_chart(model_fit.forecast(steps=14))