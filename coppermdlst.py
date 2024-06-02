import streamlit as st
import pickle
import numpy as np
import datetime

# Load the trained models and encoders
with open(r"C:\Users\tpsna\OneDrive\Desktop\VSCode\Copper_Modeling\country.pkl","rb") as file:
    encode_country = pickle.load(file, encoding='latin1')
with open(r"C:\Users\tpsna\OneDrive\Desktop\VSCode\Copper_Modeling\status.pkl","rb") as file:
    encode_status = pickle.load(file, encoding='latin1')
with open(r"C:\Users\tpsna\OneDrive\Desktop\VSCode\Copper_Modeling\item type.pkl","rb") as file:
    encode_item = pickle.load(file, encoding='latin1')
with open(r"C:\Users\tpsna\OneDrive\Desktop\VSCode\Copper_Modeling\scaling.pkl","rb") as file:
    scaled_data = pickle.load(file, encoding='latin1')


# Function to predict the Selling_Price
def predict_price(input_data):
    input_df = np.array(input_data).reshape(1, -1)
    pred_model = scaled_data.transform(input_df)
    price_predict = price_model.predict(pred_model)
    return price_predict[0]

# Function to predict the Status
def predict_status(input_data):
    input_df = np.array(input_data).reshape(1, -1)
    scaling_model_cls = scaled_data_cls.transform(input_df)
    pred_status = status_model.predict(scaling_model_cls)
    return 'Won' if pred_status == 6 else 'Lost'

# Streamlit app
def app():
    st.title("Copper Selling Price Prediction and Status")

    # Sidebar for selecting prediction task
    task = st.sidebar.selectbox('Select Task', ['Price Prediction', 'Status Prediction'])

    if task == 'Price Prediction':
        st.write('##### ***<span style="color:yellow">Fill all the fields and Press the below button to view the :red[predicted price] of copper</span>***', unsafe_allow_html=True)

        c1, c2, c3 = st.columns([2, 2, 2])
        with c1:
            quantity = st.text_input('Enter Quantity (Min:611728 & Max:1722207579) in tons')
            thickness = st.text_input('Enter Thickness (Min:0.18 & Max:400)')
            width = st.text_input('Enter Width (Min:1, Max:2990)')

        with c2:
            country_code = st.selectbox('Country Code', encode_country.classes_)
            status = st.selectbox('Status', encode_status.classes_)
            item_type = st.selectbox('Item Type', encode_item.classes_)

        with c3:
            application = st.number_input('Application Type', min_value=2, max_value=99)
            product = st.number_input('Product Reference', min_value=611728, max_value=1722207579)
            item_order_date = st.date_input("Order Date", datetime.date(2023, 1, 1))
            item_delivery_date = st.date_input("Estimated Delivery Date", datetime.date(2023, 1, 1))

        if st.button('Predict Price'):
            encoded_country = encode_country.transform([country_code])[0]
            encoded_status = encode_status.transform([status])[0]
            encoded_item_type = encode_item.transform([item_type])[0]

            order_date = datetime.datetime.strptime(str(item_order_date), "%Y-%m-%d")
            delivery_date = datetime.datetime.strptime(str(item_delivery_date), "%Y-%m-%d")
            days_diff = delivery_date - order_date

            input_data = [quantity, thickness, width, encoded_country, encoded_status, encoded_item_type, application, product, days_diff.days]
            predicted_price = predict_price(input_data)
            st.write(f'Predicted Selling Price: :green[₹] :green[{predicted_price:.2f}]')

    elif task == 'Status Prediction':
        st.write('##### ***<span style="color:yellow">Fill all the fields and Press the below button to view the status :red[WON / LOST] of copper in the desired time range</span>***', unsafe_allow_html=True)

        cc1, cc2, cc3 = st.columns([2, 2, 2])
        with cc1:
            quantity_cls = st.text_input('Enter Quantity (Min:611728 & Max:1722207579) in tons')
            thickness_cls = st.text_input('Enter Thickness (Min:0.18 & Max:400)')
            width_cls = st.text_input('Enter Width (Min:1, Max:2990)')

        with cc2:
            selling_price_cls = st.text_input('Enter Selling Price (Min:1, Max:100001015)')
            item_cls = st.selectbox('Item Type', encode_item.classes_)
            country_cls = st.selectbox('Country Code', encode_country.classes_)

        with cc3:
            application_cls = st.number_input('Application Type', min_value=2, max_value=99)
            product_cls = st.number_input('Product Reference', min_value=611728, max_value=1722207579)
            item_order_date_cls = st.date_input("Order Date", datetime.date(2023, 1, 1))
            item_delivery_date_cls = st.date_input("Estimated Delivery Date", datetime.date(2023, 1, 1))

        if st.button('Predict Status'):
            encoded_country_cls = encode_country.transform([country_cls])[0]
            encoded_item_cls = encode_item.transform([item_cls])[0]

            order_date_cls = datetime.datetime.strptime(str(item_order_date_cls), "%Y-%m-%d")
            delivery_date_cls = datetime.datetime.strptime(str(item_delivery_date_cls), "%Y-%m-%d")
            days_diff_cls = delivery_date_cls - order_date_cls

            input_data_cls = [quantity_cls, thickness_cls, width_cls, selling_price_cls, encoded_country_cls, encoded_item_cls, application_cls, product_cls, days_diff_cls.days]
            predicted_status = predict_status(input_data_cls)
            st.write(f'Predicted Status: :green[{predicted_status}]')

    st.info("The Predicted selling price/status may differ from various reasons like Supply and Demand Imbalances, Infrastructure and Transportation, etc.", icon='ℹ️')

if __name__ == "__main__":
    app()