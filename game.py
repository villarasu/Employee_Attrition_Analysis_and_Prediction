import pandas as pd
import streamlit as st 
import seaborn as sns
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.metrics import accuracy_score, classification_report

##  Page Configuration ##
st.set_page_config(
    page_title="🌟 Employee Attrition Analyzer",
    layout="wide",
    page_icon="📊"
)

## Stylish Theme & Background ##
st.markdown("""
    <style>
    body {
        background: linear-gradient(to right, #f0f2f5, #e0eafc);
    }
    .main {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #1a1a1a;
    }
    </style>
""", unsafe_allow_html=True)

## Sidebar Navigation ##
st.sidebar.title(" 📡 Explore")
page = st.sidebar.radio("Go to", ["🏠 Home", "🔍 Predict Attrition", "📊 EDA", "📚 About Me"])

## Sidebar File Upload ##
uploaded_file = st.sidebar.file_uploader("📂 Upload your CSV", type=["csv"])

## Load Data Function ##
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    drop_cols = ['Over18', 'StandardHours', 'EmployeeCount', 'EmployeeNumber']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)
    return df

MODEL_FILE = "saved_model.pkl"
ENCODER_FILE = "label_encoders.pkl"

## About Me Page ##
if page == "📚 About Me":
    st.title("📚 About Me")
    st.image("img.png", caption="Villarasu - Data Analyst", width=150)
    st.markdown("""
        ### 👨‍💻 Developer: Villarasu_siva
        - 🔬 **Project**: Employee Attrition Analysis & Prediction  
        - 🧠 **Skills**: Python, Streamlit, Machine Learning, EDA, SMOTE, Random Forest  
        - 📊 **Goal**: Help companies understand why employees leave and predict future attrition  
        - 🛠️ **Tools**: Streamlit, pandas, scikit-learn, seaborn, matplotlib  
        - 🌐 **Contact**: [LinkedIn](https://www.linkedin.com/in/villarasu-siva-9780a8288/) | [GitHub](https://github.com/villarasu?tab=repositories)
        ---
        **Thank you for visiting the Employee Attrition Analyzer App!** 😊  
    """)

elif uploaded_file:
    df = load_data(uploaded_file)

    ## Load pre-trained model ##
    if os.path.exists(MODEL_FILE) and os.path.exists(ENCODER_FILE):
        model = joblib.load(MODEL_FILE)
        label_encoders = joblib.load(ENCODER_FILE)
        st.success("✅ Loaded pre-trained model and encoders.")
    else:
        st.error("❌ Pre-trained model or encoders not found. Please make sure 'saved_model.pkl' and 'label_encoders.pkl' exist.")
        st.stop()

    ## Apply label encoding using loaded encoders ##
    label_mappings = {}
    for col in df.select_dtypes(include='object'):
        if col in label_encoders:
            le = label_encoders[col]
            df[col] = le.transform(df[col])
            label_mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))

    ## Features & target ##
    if 'Attrition' in df.columns:
        X = df.drop('Attrition', axis=1)
        y = df['Attrition']

        ## Predictions on test data ##
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred, target_names=["Stay", "Leave"])
    else:
        X = df
        y = None
        acc = None
        report = None

    ## Home ##
    if page == "🏠 Home":
        st.title("🌟 Welcome to the Employee Attrition Analyzer")
        st.markdown("""
        This dashboard allows you to:
        - 🔍 Predict employee attrition
        - 📊 Explore attrition insights with EDA  
        - 📚 About Me
        **📂 Upload your CSV to begin!**
        """)

    ## Prediction Page ##
    elif page == "🔍 Predict Attrition":
        st.title("🔍 Employee Attrition Prediction")
        if acc:
            st.subheader("📈 Model Performance")
            col1, col2 = st.columns(2)
            col1.metric("✅ Accuracy", f"{acc * 100:.2f}%")
            col2.code(report)

        st.divider()
        st.subheader("🧠 Predict for a New Employee")
        input_data = {}
        for col in X.columns:
            if col in label_mappings:
                input_data[col] = st.selectbox(f"{col}", list(label_mappings[col].keys()))
                input_data[col] = label_mappings[col][input_data[col]]
            else:
                input_data[col] = st.slider(f"{col}", int(X[col].min()), int(X[col].max()), int(X[col].median()))

        if st.button("🚀 Predict Now"):
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]
            age = input_df['Age'].values[0] if 'Age' in input_df.columns else 'Unknown'

            if prediction == 1:
                st.image("exit.png", width=200, caption=f"⚠️ Likely to Leave (Age: {age})")
            else:
                st.image("stay.png", width=200, caption=f"✅ Likely to Stay (Age: {age})")

    ## EDA ##
    elif page == "📊 EDA":
        st.title("📊 Exploratory Data Analysis")
        if 'Attrition' in df.columns:
            st.subheader("📌 Quick Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("👥 Total Employees", len(df))
            col2.metric("🚪 Left", df['Attrition'].sum())
            col3.metric("📈 Attrition Rate", f"{df['Attrition'].mean() * 100:.2f}%")

        with st.expander("🔍 View Raw Dataset"):
            st.dataframe(df.head(20))

        st.subheader("📊 Detailed Visualizations")
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Attrition Count", "Age vs Attrition", "Department-wise", "OverTime Effect", "Correlation", "Monthly income and Job role"
        ])

        if 'Attrition' in df.columns:
            with tab1:
                fig1, ax1 = plt.subplots(figsize=(3, 3))
                sns.countplot(x='Attrition', data=df, palette='Set2', ax=ax1)
                st.pyplot(fig1)

            with tab2:
                fig2, ax2 = plt.subplots(figsize=(6, 3))
                sns.boxplot(x='Attrition', y='Age', data=df, palette='Pastel1', ax=ax2)
                st.pyplot(fig2)

            with tab3:
                if 'Department' in df.columns:
                    fig3, ax3 = plt.subplots(figsize=(6, 2))
                    sns.countplot(x='Department', hue='Attrition', data=df, palette='coolwarm', ax=ax3)
                    st.pyplot(fig3)

            with tab4:
                if 'OverTime' in df.columns:
                    fig4, ax4 = plt.subplots(figsize=(6, 2))
                    sns.countplot(x='OverTime', hue='Attrition', data=df, palette='flare', ax=ax4)
                    st.pyplot(fig4)

            with tab5:
                corr = df.corr(numeric_only=True)
                top_corr = corr['Attrition'].abs().sort_values(ascending=False).head(11).index
                fig5, ax5 = plt.subplots(figsize=(10, 6))
                sns.heatmap(df[top_corr].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax5)
                st.pyplot(fig5)

            with tab6:
                if 'MonthlyIncome' in df.columns:
                    fig6, ax6 = plt.subplots(figsize=(6, 3))
                    sns.kdeplot(data=df, x='MonthlyIncome', hue='Attrition', fill=True, common_norm=False, palette='Set1', ax=ax6)
                    st.pyplot(fig6)

else:
    if page != "📚 About Me":
        st.warning("📁 Please upload a CSV file to begin.")
