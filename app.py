# life of an employee at an organization.
# Author: Vishwa, Raghu - Drpinnacle (Refined by Antigravity)

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import joblib
from category_encoders import one_hot
from io import StringIO
import streamlit as st

# set page configuration
st.set_page_config(layout="wide", page_title="DrLoE - Employee Attrition Analytics", page_icon="📈")

# Premium Custom CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    /* Font styling */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Header Gradient styling */
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #FF4B4B 0%, #FF8F8F 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }
    .sub-title {
        font-size: 1.15rem;
        color: #808495;
        margin-bottom: 25px;
    }
    
    /* Premium Metric Card Styling */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        backdrop-filter: blur(10px);
    }
    .metric-label {
        font-size: 0.9rem;
        font-weight: 500;
        color: #808495;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 5px;
    }
    
    /* Custom divider */
    .divider {
        height: 2px;
        background: linear-gradient(90deg, rgba(255, 75, 75, 0.2) 0%, rgba(255, 255, 255, 0) 100%);
        margin: 20px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar styling and content
st.sidebar.markdown(
    """
    <div style="text-align: center; margin-bottom: 20px;">
        <h2 style="font-weight: 700; margin-bottom: 5px;">DrLoE</h2>
        <p style="color: #808495; font-size: 0.9rem;">Life of an Employee Analytics</p>
    </div>
    """,
    unsafe_allow_html=True
)

with st.sidebar.expander("ℹ️ About the App", expanded=True):
    st.write(
        """
        DrLoE is an interactive HR decision-support tool powered by Machine Learning. 
        It predicts the likelihood of employee attrition based on core performance indicators, work characteristics, and satisfaction levels.
        """
    )

# Sidebar Feedback Form
st.sidebar.markdown("### 💬 App Feedback")
with st.sidebar.form(key="feedback_form", clear_on_submit=True):
    st.write("<style>div.row-widget.stRadio > div{flex-direction:row;} </style>", unsafe_allow_html=True)
    rating = st.radio("Rate your experience:", ("1", "2", "3", "4", "5"), index=4)
    feedback_text = st.text_input(label="Help us improve:")
    submitted = st.form_submit_button("Submit Feedback")
    if submitted:
        st.sidebar.success("Thank you for your feedback!")

with st.sidebar.expander("📞 Contact & Support", expanded=False):
    st.write("Email: info@drpinnacle.com")

# Main Header
st.markdown('<h1 class="main-title">DrLoE Employee Attrition Suite</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Leverage predictive analytics to anticipate talent churn and improve workforce retention.</p>', unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Tabs Navigation
tab_online, tab_batch = st.tabs(["🎯 Individual Prediction", "📂 Batch File Analysis"])


@st.cache_resource
def load_models():
    """Load the machine learning model and one-hot encoder once and cache them."""
    ce_ohe = joblib.load("clean_ohe.pkl")
    model = joblib.load("emp_rf.pkl")
    return ce_ohe, model


def cleanData(df, cat_columns):
    """
    Clean the input dataframe, encode categorical variables, and run predictions.
    Returns the predicted class (0/1) and the attrition probability.
    """
    ce_ohe, model = load_models()
    loaded_ce_dummies = ce_ohe.transform(df[cat_columns])
    data_other_cols = df.drop(columns=cat_columns)
    pred_df = pd.concat([loaded_ce_dummies, data_other_cols], axis=1)
    pred_class = model.predict(pred_df.values)[0]
    pred_prob = model.predict_proba(pred_df.values)[0][1]
    return pred_class, pred_prob


def cleanDataBatch(df, cat_columns):
    """
    Clean batch dataframe, encode categorical variables, and run predictions.
    Returns the array of predicted classes and their corresponding attrition probabilities.
    """
    ce_ohe, model = load_models()
    loaded_ce_dummies = ce_ohe.transform(df[cat_columns])
    data_other_cols = df.drop(columns=cat_columns)
    pred_df = pd.concat([loaded_ce_dummies, data_other_cols], axis=1)
    pred_classes = model.predict(pred_df.values)
    pred_probs = model.predict_proba(pred_df.values)[:, 1]
    return pred_classes, pred_probs


# ---------------- INDIVIDUAL PREDICTIONS ----------------
with tab_online:
    st.markdown("### 🔍 Employee Risk Profile")
    st.write("Enter the employee details below to analyze their retention risk.")
    
    # Input layouts
    col_meta1, col_meta2 = st.columns(2)
    with col_meta1:
        emp_name = st.text_input("Employee Name", value="Jane Doe")
    with col_meta2:
        emp_id = st.text_input("Employee ID / Reference", value="EMP-8492")
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        satisfaction = st.slider(
            "Satisfaction Level", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.65, 
            step=0.01,
            help="Employee's self-reported job satisfaction score"
        )
        evaluation = st.slider(
            "Last Evaluation Score", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.70, 
            step=0.01,
            help="Score received during the last performance review"
        )
        projects = st.number_input(
            "Number of Active Projects", 
            min_value=1, 
            max_value=10, 
            value=4, 
            step=1
        )
        monthly_hours = st.slider(
            "Average Monthly Working Hours", 
            min_value=80, 
            max_value=320, 
            value=200, 
            step=5,
            help="Average hours worked per month"
        )
        
    with col2:
        tenure = st.number_input(
            "Years at Company (Tenure)", 
            min_value=1, 
            max_value=20, 
            value=3, 
            step=1
        )
        accident = st.selectbox(
            "Experienced a Work Accident?", 
            options=[0, 1], 
            format_func=lambda x: "No" if x == 0 else "Yes"
        )
        promotion = st.selectbox(
            "Promoted in Last 5 Years?", 
            options=[0, 1], 
            format_func=lambda x: "No" if x == 0 else "Yes"
        )
        department = st.selectbox(
            "Department",
            ("sales", "technical", "support", "IT", "hr", "accounting", "marketing", "product_mng", "randD", "mangement")
        )
        salary_level = st.selectbox(
            "Salary Level", 
            options=["low", "medium", "high"],
            format_func=lambda x: x.capitalize()
        )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    
    if st.button("📊 Run Attrition Analysis", type="primary"):
        input_data = {
            "Satisfaction_level": satisfaction,
            "Last_evaluation": evaluation,
            "number_project": projects,
            "average_montly_hours": monthly_hours,
            "time_spend_company": tenure,
            "Work_accident": accident,
            "promotion_last_5years": promotion,
            "dept": department,
            "salary": salary_level,
        }
        
        input_df = pd.DataFrame(input_data, index=[0])
        
        with st.spinner("Analyzing risk parameters..."):
            pred_class, pred_prob = cleanData(input_df, ["salary", "dept"])
            
        # Display Prediction Result Card
        st.markdown("### 📊 Analysis Results")
        
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Churn Risk Score</div>
                    <div class="metric-value" style="color: {'#FF4B4B' if pred_prob > 0.6 else '#FFAA00' if pred_prob > 0.3 else '#00CC66'};">
                        {pred_prob * 100:.1f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            if pred_class == 1:
                st.error(f"⚠️ **Warning**: {emp_name} ({emp_id}) shows high attrition risk and is predicted to **leave** the company.")
            else:
                st.success(f"✅ **Retention Confirmed**: {emp_name} ({emp_id}) is predicted to **stay** with the company.")
                
        with res_col2:
            st.markdown("#### Risk Analysis Gauge")
            st.progress(pred_prob)
            
            # Actionable insights
            st.markdown("#### 💡 Retention Insights")
            insights = []
            if satisfaction < 0.4:
                insights.append("- **Low Satisfaction**: The employee's satisfaction is critically low. Consider a feedback session to understand their pain points.")
            if monthly_hours > 250:
                insights.append("- **High Workload / Burnout Risk**: Average monthly hours are exceptionally high. Restructuring projects or distributing workload is recommended.")
            if tenure >= 3 and promotion == 0:
                insights.append("- **Stagnation Factor**: The employee has been with the company for 3+ years without a promotion. Review growth and career path opportunities.")
            if salary_level == "low" and satisfaction < 0.6:
                insights.append("- **Compensation Gap**: Lower salary level paired with lower satisfaction might indicate the employee is feeling under-compensated.")
                
            if insights:
                for insight in insights:
                    st.write(insight)
            else:
                st.write("- **Healthy Metrics**: Core parameters are within healthy thresholds. Continue standard engagement practices.")


# ---------------- BATCH PREDICTIONS ----------------
with tab_batch:
    st.markdown("### 📂 Bulk Processing & Analytics Dashboard")
    st.write("Upload a CSV file containing employee metrics to run batch attrition analysis and view workforce dashboards.")
    
    # Guidance details
    with st.expander("📋 Required CSV Schema & Column Reference", expanded=False):
        st.write("Ensure your CSV file contains the following columns exactly (ignoring order):")
        st.code("Satisfaction_level, Last_evaluation, number_project, average_montly_hours, time_spend_company, Work_accident, promotion_last_5years, dept, salary")
        try:
            sample_img = Image.open('sample.png')
            st.image(sample_img, caption="CSV Schema Sample Format", width=500)
        except Exception:
            pass

    st.markdown("<br>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Employee Data (CSV Format)", type=["csv"])
    
    if uploaded_file is not None:
        try:
            dataframe = pd.read_csv(uploaded_file, index_col=0)
            
            with st.spinner("Processing batch dataset..."):
                pred_classes, pred_probs = cleanDataBatch(dataframe, ["salary", "dept"])
                
            result_df = dataframe.copy()
            result_df["Attrition_Prediction"] = pred_classes
            result_df["Attrition_Risk_Score"] = pred_probs
            
            # Map predictions to human readable strings
            result_df["Status"] = result_df["Attrition_Prediction"].map({0: "Stay", 1: "Leave"})
            
            total_employees = len(result_df)
            attrition_count = int(np.sum(pred_classes))
            attrition_rate = (attrition_count / total_employees) * 100
            avg_risk = np.mean(pred_probs) * 100
            
            st.markdown("### 📊 Batch Performance Dashboard")
            
            # Metric Columns
            met1, met2, met3, met4 = st.columns(4)
            with met1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Total Evaluated</div>
                        <div class="metric-value">{total_employees}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with met2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Predicted Attrition</div>
                        <div class="metric-value" style="color: #FF4B4B;">{attrition_count}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with met3:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Attrition Rate</div>
                        <div class="metric-value" style="color: {'#FF4B4B' if attrition_rate > 30 else '#FFAA00' if attrition_rate > 15 else '#00CC66'};">
                            {attrition_rate:.1f}%
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with met4:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Avg Risk Score</div>
                        <div class="metric-value">{avg_risk:.1f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Visualizations
            fig_col1, fig_col2 = st.columns(2)
            
            with fig_col1:
                st.markdown("#### Attrition Forecast Distribution")
                fig, ax = plt.subplots(figsize=(6, 4))
                colors = ["#2ECC71", "#E74C3C"]
                result_df["Status"].value_counts().plot(kind="pie", autopct="%1.1f%%", colors=colors, startangle=90, ax=ax)
                ax.set_ylabel("")
                fig.patch.set_facecolor("none")
                ax.set_facecolor("none")
                st.pyplot(fig)
                
            with fig_col2:
                st.markdown("#### Attrition Rate by Department")
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                dept_attrition = result_df.groupby("dept")["Attrition_Prediction"].mean().reset_index()
                dept_attrition["Attrition_Prediction"] *= 100
                dept_attrition = dept_attrition.sort_values(by="Attrition_Prediction", ascending=False)
                
                sns.barplot(
                    data=dept_attrition, 
                    x="Attrition_Prediction", 
                    y="dept", 
                    palette="Reds_r", 
                    ax=ax2
                )
                ax2.set_xlabel("Attrition Rate (%)")
                ax2.set_ylabel("Department")
                fig2.patch.set_facecolor("none")
                ax2.set_facecolor("none")
                st.pyplot(fig2)
                
            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
            
            # Download Section
            csv_data = result_df.to_csv().encode("utf-8")
            file_name = uploaded_file.name.split(".")[0] + "_predictions.csv"
            
            st.markdown("#### 📂 Export Results")
            st.download_button(
                label="📥 Download Annotated Predictions (CSV)",
                data=csv_data,
                file_name=file_name,
                mime="text/csv",
                type="primary"
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 📄 Data Preview")
            st.dataframe(
                result_df[["Status", "Attrition_Risk_Score", "Satisfaction_level", "Last_evaluation", "number_project", "average_montly_hours", "time_spend_company", "dept", "salary"]].head(20),
                use_container_width=True
            )
            
        except Exception as e:
            st.error(f"Error parsing dataset. Please verify formatting matches requirements. Error details: {e}")
