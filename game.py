import pandas as pd
import streamlit as st 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

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

##  Sidebar Navigation ##

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

## build project developer ##

if page == "📚 About Me":
    st.title("📚 About Me")
    st.image( "img.png", caption=" Villarasu - Data Analyst", width=150)
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

    ##  Label Encoding ##

    label_encoders = {}
    label_mappings = {}
    for col in df.select_dtypes(include='object'):
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        label_mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))

    ##  feature & target ##

    X = df.drop('Attrition', axis=1)
    y = df['Attrition']

    ## SMOTE fixes the class imbalance ##

    sm = SMOTE(random_state=42)
    X_resampled, y_resampled = sm.fit_resample(X, y)

    ## Train-Test Split ## train 80% test 20%

    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42)

    ## build  Model ##

    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)

    ##  Evaluation ##

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Stay", "Leave"])

    ## streamlit home page ##
    
    if page == "🏠 Home":
        st.title("🌟 Welcome to the Employee Attrition Analyzer")
        st.markdown("""
        This dashboard allows you to:
        - 🔍 Predict employee attrition
        - 📊 Explore attrition insights with EDA  
        - 📚 About Me
                    
        **📂 Upload your CSV to begin!**
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

        ##  Metrics ##

        st.subheader("📌 Quick Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("👥 Total Employees", len(df))
        col2.metric("🚪 Left", df['Attrition'].sum())
        col3.metric("📈 Attrition Rate", f"{df['Attrition'].mean() * 100:.2f}%")

        ##  Preview ##

        with st.expander("🔍 View Raw Dataset"):
            st.dataframe(df.head(20))

        ## Visual ##
    
        st.subheader("📊 Detailed Visualizations")
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Attrition Count", "Age vs Attrition", "Department-wise", "OverTime Effect", "Correlation","Monthly income and Job role"
        ])
        

        with tab1:
            st.markdown("#### 🔢 Attrition Count")
            fig1, ax1 = plt.subplots(figsize=(3, 3))
            sns.countplot(x='Attrition', data=df, palette='Set2', ax=ax1)
            st.pyplot(fig1)

        with tab2:
            st.markdown("#### 📉 Age Distribution by Attrition")
            fig2, ax2 = plt.subplots(figsize=(6, 3))
            sns.boxplot(x='Attrition', y='Age', data=df, palette='Pastel1', ax=ax2)
            st.pyplot(fig2)

        with tab3:
            if 'Department' in df.columns:
                st.markdown("#### 🏢 Department-wise Attrition")
                fig3, ax3 = plt.subplots(figsize=(6, 2))
                sns.countplot(x='Department', hue='Attrition', data=df, palette='coolwarm', ax=ax3)
                st.pyplot(fig3)
            else:
                st.info("⚠️ 'Department' column not found in dataset.")

        with tab4:
            if 'OverTime' in df.columns:
                st.markdown("#### 🕒 OverTime vs Attrition")
                fig4, ax4 = plt.subplots(figsize=(6, 2))
                sns.countplot(x='OverTime', hue='Attrition', data=df, palette='flare', ax=ax4)
                st.pyplot(fig4)
            else:
                st.info("⚠️ 'OverTime' column not found in dataset.")

        with tab5:
            st.markdown("#### 🔥 Top Feature Correlations with Attrition")
            corr = df.corr(numeric_only=True)
            top_corr = corr['Attrition'].abs().sort_values(ascending=False).head(11).index
            fig5, ax5 = plt.subplots(figsize=(10, 6))
            sns.heatmap(df[top_corr].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax5)
            st.pyplot(fig5)
        with tab6:
            st.markdown("#### 💰 Monthly Income Distribution by Attrition")
            if 'MonthlyIncome' in df.columns:
                fig6, ax6 = plt.subplots(figsize=(6, 3))
                sns.kdeplot(data=df, x='MonthlyIncome', hue='Attrition', fill=True, common_norm=False, palette='Set1', ax=ax6)
                st.pyplot(fig6)
            else:
                st.info("⚠️ 'MonthlyIncome' column not found.")

            st.markdown("#### 👔 Job Role-wise Attrition")
            if 'JobRole' in df.columns:
                fig7, ax7 = plt.subplots(figsize=(8, 3))
                sns.countplot(x='JobRole', hue='Attrition', data=df, palette='husl', ax=ax7)
                plt.xticks(rotation=45)
                st.pyplot(fig7)
            else:
                st.info("⚠️ 'JobRole' column not found.")

            st.markdown("#### 🎓 Education Field-wise Attrition")
            if 'EducationField' in df.columns:
                fig8, ax8 = plt.subplots(figsize=(8, 3))
                sns.countplot(x='EducationField', hue='Attrition', data=df, palette='cubehelix', ax=ax8)
                plt.xticks(rotation=30)
                st.pyplot(fig8)
            else:
                st.info("⚠️ 'EducationField' column not found.")

            st.markdown("#### 📈 Years at Company vs Attrition")
            if 'YearsAtCompany' in df.columns:
                fig9, ax9 = plt.subplots(figsize=(6, 3))
                sns.boxplot(x='Attrition', y='YearsAtCompany', data=df, palette='Accent', ax=ax9)
                st.pyplot(fig9)
            else:
                st.info("⚠️ 'YearsAtCompany' column not found.")
        
else:
    if page != "📚 About Me":
        st.warning("📁 Please upload a CSV file to begin.")
