import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# --- Page Configuration ---
st.set_page_config(
    page_title="🌟 Employee Attrition Analyzer",
    layout="wide",
    page_icon="📊"
)

# --- Custom Styling ---
st.markdown("""
    <style>
    .main {background-color: #f4f4f4;}
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    .sidebar .sidebar-content {background-color: #f0f2f6;}
    h1, h2, h3, h4, h5, h6 {color: #333333;}
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Navigation ---
st.sidebar.title("📁 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🔍 Predict Attrition", "📊 EDA"])

# --- Sidebar File Upload ---
uploaded_file = st.sidebar.file_uploader("📂 Upload your CSV", type=["csv"])

# --- Load Data Function ---
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    drop_cols = ['Over18', 'StandardHours', 'EmployeeCount', 'EmployeeNumber']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)
    return df

# --- Prediction and EDA Logic ---
if uploaded_file:
    df = load_data(uploaded_file)

    # Encode Categorical Variables
    label_encoders = {}
    label_mappings = {}
    for col in df.select_dtypes(include='object'):
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        label_mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))

    # Features and Target
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']

    # SMOTE to balance data
    sm = SMOTE(random_state=42)
    X_resampled, y_resampled = sm.fit_resample(X, y)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Train Model
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    # Model Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Stay", "Leave"])

    # --- Page Routing ---
    if page == "🏠 Home":
        st.title("🌟 Welcome to the Employee Attrition Analyzer")
        st.markdown("""
        This dashboard allows you to:
        - 🔍 Predict employee attrition based on various factors
        - 📊 Explore insightful visualizations about attrition trends
        
        **Upload your dataset to get started!**
        """)

    elif page == "🔍 Predict Attrition":
        st.title("🔍 Employee Attrition Prediction")
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

    elif page == "📊 EDA":
        st.title("📊 Exploratory Data Analysis")

        st.subheader("📋 Dataset Preview")
        st.dataframe(df.head(10))

        st.subheader("🔢 Attrition Count")
        fig1, ax1 = plt.subplots(figsize=(3,3))

        sns.countplot(x='Attrition', data=df, ax=ax1)
        st.pyplot(fig1)

        st.subheader("📉 Age Distribution by Attrition")
        fig2, ax2 = plt.subplots()
        sns.boxplot(x='Attrition', y='Age', data=df, ax=ax2)
        st.pyplot(fig2)

        if 'Department' in df.columns:
            st.subheader("🏢 Department-wise Attrition")
            fig3, ax3 = plt.subplots()
            sns.countplot(x='Department', hue='Attrition', data=df, ax=ax3)
            st.pyplot(fig3)

        if 'OverTime' in df.columns:
            st.subheader("🕒 OverTime vs Attrition")
            fig4, ax4 = plt.subplots()
            sns.countplot(x='OverTime', hue='Attrition', data=df, ax=ax4)
            st.pyplot(fig4)

else:
    st.warning("📁 Please upload a CSV file to begin.")
