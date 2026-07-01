# life of an employee at an organization.
# Author: Vishwa, Raghu - Drpinnacle

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import xlrd
import io
import joblib
from category_encoders import one_hot
from io import StringIO

# import libraries from streamlit
import streamlit as st

# machinlearning libraries
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# set page configuration and this can be initiated only once
st.set_page_config(layout="wide", page_title="DrLoE App", page_icon='random')

# Title, Header, sub header
st.title("DrLoE (Life of an Employee) App")

st.subheader("Predict when your employee will leave the company")

# initiate side bar for navigation
st.sidebar.header("DrLoE App")
with st.sidebar.expander("About the DrLoE App", expanded=True):
    st.write(
        """
        This interactive people management App was built to predict employee attrition.
     """
    )

# Rating for the app
with st.sidebar.form(
    key="columns_in_form", clear_on_submit=True
):  # set clear_on_submit=True so that the form will be reset/cleared once it's submitted
    st.write("Please help us improve!")
    st.write(
        "<style>div.row-widget.stRadio > div{flex-direction:row;} </style>",
        unsafe_allow_html=True,
    )  # Make horizontal radio buttons
    rating = st.radio(
        "Please rate the app", ("1", "2", "3", "4", "5"), index=4
    )  # Use radio buttons for ratings
    text = st.text_input(
        label="Please leave your feedback here"
    )  # Collect user feedback
    submitted = st.form_submit_button("Submit")
    if submitted:
        st.write("Thanks for your feedback!")
        st.markdown(f"**Your Rating:** {rating}")
        st.markdown(f"**Your Feedback:** {text}")

with st.sidebar.expander("Contact Us", expanded=True):
    st.write("info@drpinnacle.com")

# User Choice
user_choices = ["Online Predictions", "Batch Predictions"]
selected_choice = st.selectbox("Please select your choice:", user_choices)


@st.cache_resource
def load_models():
    """Load the machine learning model and one-hot encoder once and cache them."""
    ce_ohe = joblib.load("clean_ohe.pkl")
    model = joblib.load("emp_rf.pkl")
    return ce_ohe, model


def cleanData(df, cat_columns):
    """
    Clean the dataframe, arrange the dataframe into
    model required format
    Input:   df (dataframe)
            cat_columns: categorical columns (['salary','dept'])
    Output: prediction result
    """
    ce_ohe, model = load_models()
    loaded_ce_dummies = ce_ohe.transform(df[cat_columns])
    data_other_cols = df.drop(columns=cat_columns)
    # Concatenate the two dataframes
    pred_df = pd.concat([loaded_ce_dummies, data_other_cols], axis=1)
    pred_res = model.predict(pred_df.values)
    return pred_res[0]


def cleanDataBatch(df, cat_columns):
    """
    Clean the dataframe, arrange the dataframe into
    model required format
    Input:   df (dataframe)
            cat_columns: categorical columns (['salary','dept'])
    Output: prediction result
    """
    ce_ohe, model = load_models()
    loaded_ce_dummies = ce_ohe.transform(df[cat_columns])
    data_other_cols = df.drop(columns=cat_columns)
    # Concatenate the two dataframes
    pred_df = pd.concat([loaded_ce_dummies, data_other_cols], axis=1)
    pred_res = model.predict(pred_df.values)
    return pred_res


# for online predictions
if selected_choice == "Online Predictions":
    st.write(
        "Please fill in the form below to predict attrition for an individual employee."
    )
    
    def user_input_online():
        Employee_name = st.text_input("Please enter employee name", value="John Doe")
        Employee_ID = st.text_input("Please enter employee ID", value="EMP001")
        
        col1, col2 = st.columns(2)
        with col1:
            Satisfaction_level = st.number_input(
                "Satisfaction level (0.0 - 1.0)", min_value=0.00, max_value=1.00, value=0.5
            )
            Last_evaluation = st.number_input(
                "Last Evaluation (0.0 - 1.0)", min_value=0.00, max_value=1.00, value=0.5
            )
            number_project = st.number_input(
                "Number of projects", min_value=0, max_value=10, value=5
            )
            average_montly_hours = st.number_input(
                "Average monthly hours",
                min_value=0.00,
                max_value=1000.00,
                value=300.00,
            )
        
        with col2:
            time_spend_company = st.number_input(
                "Years in company", min_value=0, max_value=20, value=5
            )
            Work_accident = st.selectbox("Work accident", (0, 1))
            promotion_last_5years = st.selectbox("Promotion last 5 years", (0, 1))
            dept = st.selectbox(
                "Department",
                (
                    "sales",
                    "technical",
                    "support",
                    "IT",
                    "hr",
                    "accounting",
                    "marketing",
                    "product_mng",
                    "randD",
                    "mangement",
                ),
            )
            salary = st.selectbox("Salary Level", ("low", "medium", "high"))

        input_user = {
            "Satisfaction_level": Satisfaction_level,
            "Last_evaluation": Last_evaluation,
            "number_project": number_project,
            "average_montly_hours": average_montly_hours,
            "time_spend_company": time_spend_company,
            "Work_accident": Work_accident,
            "promotion_last_5years": promotion_last_5years,
            "dept": dept,
            "salary": salary,
        }

        # Converting to DataFrame
        input_user_df = pd.DataFrame(input_user, index=[0])
        return input_user_df, Employee_ID, Employee_name

    input_user_df, Employee_ID, Employee_name = user_input_online()

    if st.button("Predict"):
        pred_result = cleanData(input_user_df, ["salary", "dept"])
        if pred_result == 0:
            st.success(f"**{Employee_name}** ({Employee_ID}) is predicted to **stay** in the organization.")
        else:
            st.warning(f"**{Employee_name}** ({Employee_ID}) is predicted to **leave** the organization.")

# For Batch Predictions
elif selected_choice == "Batch Predictions":
    st.markdown("Please make sure you have collected the **right data points** to run the batch predictions.")
    st.write("Required columns: Satisfaction level, Last Evaluation, Number of project, The average montly hours, Time spend in company, Work accident, Promotion last 5 years, Department, Salary Level")
    
    image = Image.open('sample.png')
    st.image(image, caption='Example data format and columns')
    
    st.write("Please upload your dataset in CSV format")
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        dataframe = pd.read_csv(uploaded_file, index_col=0)
        pred_result = cleanDataBatch(dataframe, ["salary", "dept"])
        result_df = dataframe.copy()
        result_df["Attrition_Prediction"] = pred_result

        @st.cache_data
        def convert_df(df):
            return df.to_csv().encode("utf-8")

        csv = convert_df(result_df)
        file_name = uploaded_file.name.split(".")[0] + "_result" + ".csv"
        st.download_button(
            label="Download predictions as CSV",
            data=csv,
            file_name=file_name,
            mime="text/csv",
        )

