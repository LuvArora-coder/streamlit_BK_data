import streamlit as st
import pandas as pd
import statsmodels.api as sm

def pred():
    col = st.columns(2)
    with col[0]:
        st_date = st.date_input("Select Start Date",pd.to_datetime('2018-01-01'))
    with col[1]:
        ed_date = st.date_input("Select End Date",pd.to_datetime('2019-03-28'))

    data = pd.read_csv("Superstore_sales_Data.csv")
    df_sup = data[['Order Date','Sales']]
    df_sup['Order Date'] = pd.to_datetime(df_sup['Order Date'],format='%d/%m/%Y')
    df_sup = df_sup.groupby('Order Date').sum({
        'Sales': lambda price: price.sum()})
    df_sup = df_sup['Sales'].resample('D').sum()
    # st.write(df_sup)

    ## ARIMA for time series.
    # fit model
    mod = sm.tsa.statespace.SARIMAX(df_sup,
                                order=(1,1,1),
                                seasonal_order= (1,1,0,12),
                                enforce_invertibility=False)
    model_fit = mod.fit()

    # pred_uc = model_fit.forecast(steps=20)
    pred_uc_w = model_fit.get_prediction(start = pd.to_datetime(st_date), end = pd.to_datetime(ed_date), dynamic = False)
    pred_ci = pred_uc_w.conf_int()

    # st.line_chart(pred_ci)
    st.write("Predection form ", str(st_date), " to ", str(ed_date))
    st.line_chart(pred_uc_w.predicted_mean)
    st.write("Predection for Upcoming 10 days for which Data is not present")
    st.line_chart(model_fit.forecast(steps=10))
    # col = st.columns(2)
    # with col[0]:
    #     sel_name = st.selectbox("Select Customer Name", data["Customer Name"].unique())
    # with col[1]:
    #     st.write("Bar Graph")
    # def sale_tot(Customer_name):
    #     df = data[data["Customer Name"] == Customer_name]
    #     value = sum(df["Sales"])
    #     return value

    # def cat_list(Customer_name):
    #     df = data[data["Customer Name"] == Customer_name]
    #     value = list(df["Category"].unique())
    #     return value

    # st.write(sale_tot(sel_name))
    # st.write(pd.DataFrame(cat_list(sel_name)))