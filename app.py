import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Data Analysis Learning Platform",
    layout="wide"
)

# Custom CSS for professional styling with better contrast
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 2rem;
        margin-bottom: 1rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .subsection-header {
        font-size: 1.4rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #f8f9fa;
        border: 2px solid #dee2e6;
        border-radius: 8px;
        padding: 20px;
        margin: 15px 0;
        font-size: 1rem;
        color: #212529;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .success-message {
        background-color: #d4edda;
        border: 2px solid #c3e6cb;
        border-radius: 8px;
        padding: 15px;
        margin: 15px 0;
        color: #155724;
        font-weight: 500;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: white;
        border: 2px solid #e9ecef;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .dataframe {
        border: 2px solid #dee2e6;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stTextInput, .stSelectbox, .stMultiselect, .stSlider {
        border: 2px solid #ced4da;
        border-radius: 6px;
        padding: 8px;
        background-color: white;
    }
    .stButton button {
        border: 2px solid #007bff;
        border-radius: 6px;
        background-color: #007bff;
        color: white;
        font-weight: 500;
        padding: 8px 16px;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: #0056b3;
        border-color: #0056b3;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,123,255,0.3);
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
        border-right: 2px solid #dee2e6;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        border: 2px solid #dee2e6;
        background-color: #f8f9fa;
        color: #495057;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: white;
        border-bottom: 2px solid #007bff;
        color: #007bff;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the TB burden dataset"""
    try:
        df = pd.read_csv('TB_Burden_Country.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def create_database(df):
    """Create SQLite database from DataFrame"""
    conn = sqlite3.connect(':memory:')
    df.to_sql('tb_data', conn, index=False, if_exists='replace')
    return conn

def main():
    st.markdown('<div class="main-header">📊 Data Analysis Learning Platform</div>', unsafe_allow_html=True)
    st.markdown("### Master Data Science Fundamentals with Tuberculosis Burden Data")

    # Load data
    df = load_data()
    if df is None:
        return

    # Create database connection
    conn = create_database(df)

    # Sidebar navigation
    st.sidebar.title("Navigation")
    section = st.sidebar.radio(
        "Choose a learning section:",
        ["Overview", "Statistics", "Data Cleaning", "SQL Queries",
         "Visualization", "Machine Learning"]
    )

    if section == "Overview":
        show_overview(df)
    elif section == "Statistics":
        show_statistics(df)
    elif section == "Data Cleaning":
        show_data_cleaning(df)
    elif section == "SQL Queries":
        show_sql_queries(conn)
    elif section == "Visualization":
        show_visualization(df)
    elif section == "Machine Learning":
        show_machine_learning(df)

def show_overview(df):
    st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)

    st.markdown('<div class="info-box">This interactive platform uses the World Health Organization\'s Tuberculosis (TB) burden dataset to teach fundamental data science concepts. TB is a major global health concern, and analyzing this data helps us understand patterns and trends in disease prevalence.</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Rows", f"{len(df):,}")
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        st.metric("Countries", df['Country or territory name'].nunique())

    st.markdown("### Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.markdown("### Column Information")
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes,
        'Non-Null Count': df.notnull().sum(),
        'Null Count': df.isnull().sum()
    })
    st.dataframe(col_info, use_container_width=True)

def show_statistics(df):
    st.markdown('<div class="section-header">Descriptive Statistics</div>', unsafe_allow_html=True)

    st.markdown('<div class="info-box">Descriptive statistics help us understand the basic properties of our data. These measures summarize and describe the main features of a dataset, providing insights into its central tendency, variability, and distribution.</div>', unsafe_allow_html=True)

    # Select numeric columns for statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        st.error("No numeric columns found for statistical analysis.")
        return

    selected_col = st.selectbox("Select a numeric column for analysis:", numeric_cols)

    st.markdown(f'<div class="subsection-header">Statistics for {selected_col}</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Mean", f"{df[selected_col].mean():.2f}")
    with col2:
        st.metric("Median", f"{df[selected_col].median():.2f}")
    with col3:
        st.metric("Std Dev", f"{df[selected_col].std():.2f}")
    with col4:
        st.metric("Count", df[selected_col].count())

    # Distribution plot
    st.markdown("### Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df[selected_col].dropna(), kde=True, ax=ax)
    ax.set_title(f'Distribution of {selected_col}')
    st.pyplot(fig)

    # Box plot
    st.markdown("### Box Plot")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(y=df[selected_col].dropna(), ax=ax)
    ax.set_title(f'Box Plot of {selected_col}')
    st.pyplot(fig)

def show_data_cleaning(df):
    st.markdown('<div class="section-header">Data Cleaning & Preprocessing</div>', unsafe_allow_html=True)

    st.markdown('<div class="info-box">Data cleaning is often the most time-consuming part of data analysis. Real-world datasets frequently contain missing values, inconsistencies, and errors that must be addressed before meaningful analysis can be performed.</div>', unsafe_allow_html=True)

    st.markdown("### Missing Values Analysis")
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100

    missing_df = pd.DataFrame({
        'Column': missing_data.index,
        'Missing Count': missing_data.values,
        'Missing Percentage': missing_percent.values
    }).sort_values('Missing Count', ascending=False)

    st.dataframe(missing_df[missing_df['Missing Count'] > 0], use_container_width=True)

    # Data cleaning options
    st.markdown('<div class="subsection-header">Cleaning Options</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Remove rows with missing values"):
            cleaned_df = df.dropna()
            st.success(f"Removed {len(df) - len(cleaned_df)} rows with missing values")
            st.dataframe(cleaned_df.head(), use_container_width=True)

    with col2:
        if st.button("Fill missing values with mean (numeric)"):
            cleaned_df = df.copy()
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].mean())
            st.success("Filled missing numeric values with column means")
            st.dataframe(cleaned_df.head(), use_container_width=True)

def show_sql_queries(conn):
    st.markdown('<div class="section-header">SQL Query Interface</div>', unsafe_allow_html=True)

    st.markdown('<div class="info-box">SQL (Structured Query Language) is the standard language for managing and manipulating relational databases. Learning SQL is essential for data analysts as it allows efficient querying, filtering, and aggregation of large datasets.</div>', unsafe_allow_html=True)

    st.markdown("### Available Tables")
    tables_query = "SELECT name FROM sqlite_master WHERE type='table';"
    tables = pd.read_sql_query(tables_query, conn)
    st.write("Tables:", tables['name'].tolist())

    # Get table schema
    schema_query = "PRAGMA table_info(tb_data);"
    schema = pd.read_sql_query(schema_query, conn)
    st.markdown("### Table Schema")
    st.dataframe(schema[['name', 'type']], use_container_width=True)

    # SQL Query input
    st.markdown('<div class="subsection-header">Execute SQL Query</div>', unsafe_allow_html=True)

    default_query = "SELECT * FROM tb_data LIMIT 10;"
    query = st.text_area("Enter your SQL query:", value=default_query, height=100)

    if st.button("Execute Query"):
        try:
            result = pd.read_sql_query(query, conn)
            st.success(f"Query executed successfully. Returned {len(result)} rows.")
            st.dataframe(result, use_container_width=True)
        except Exception as e:
            st.error(f"Error executing query: {e}")

def show_visualization(df):
    st.markdown('<div class="section-header">Data Visualization</div>', unsafe_allow_html=True)

    st.markdown('<div class="info-box">Data visualization transforms complex datasets into visual representations that are easier to understand and interpret. Effective visualizations help identify patterns, trends, and relationships that might not be apparent from raw data alone.</div>', unsafe_allow_html=True)

    viz_type = st.selectbox("Choose visualization type:",
                           ["Scatter Plot", "Bar Chart", "Line Chart", "Heatmap", "Box Plot"])

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if viz_type == "Scatter Plot":
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("X-axis:", numeric_cols, key="scatter_x")
        with col2:
            y_col = st.selectbox("Y-axis:", numeric_cols, key="scatter_y")

        if x_col and y_col:
            fig = px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
            st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Bar Chart":
        cat_col = st.selectbox("Categorical column:", df.select_dtypes(include=['object']).columns.tolist())
        num_col = st.selectbox("Numeric column:", numeric_cols)

        if cat_col and num_col:
            # Group by categorical column and aggregate
            grouped = df.groupby(cat_col)[num_col].mean().reset_index()
            fig = px.bar(grouped, x=cat_col, y=num_col, title=f"Average {num_col} by {cat_col}")
            st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Line Chart":
        x_col = st.selectbox("X-axis (time/ordinal):", df.columns.tolist())
        y_col = st.selectbox("Y-axis:", numeric_cols)

        if x_col and y_col:
            fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} over {x_col}")
            st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Heatmap":
        selected_cols = st.multiselect("Select numeric columns for correlation:", numeric_cols,
                                      default=numeric_cols[:5] if len(numeric_cols) >= 5 else numeric_cols)

        if len(selected_cols) > 1:
            corr_matrix = df[selected_cols].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)

    elif viz_type == "Box Plot":
        col = st.selectbox("Select column for box plot:", numeric_cols)
        if col:
            fig = px.box(df, y=col, title=f"Box Plot of {col}")
            st.plotly_chart(fig, use_container_width=True)

def show_machine_learning(df):
    st.markdown('<div class="section-header">Machine Learning Models</div>', unsafe_allow_html=True)

    st.markdown('<div class="info-box">Machine Learning algorithms learn patterns from data to make predictions or decisions. Scikit-learn provides efficient implementations of common ML algorithms, making it accessible for data analysis tasks.</div>', unsafe_allow_html=True)

    # Prepare data for ML
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        st.error("Need at least 2 numeric columns for ML analysis.")
        return

    # Model selection
    model_type = st.selectbox("Choose ML model:",
                             ["Linear Regression", "Logistic Regression", "Decision Tree"])

    # Feature and target selection
    col1, col2 = st.columns(2)

    with col1:
        target_col = st.selectbox("Select target variable:", numeric_cols)

    with col2:
        available_features = [col for col in numeric_cols if col != target_col]
        default_features = available_features[:2] if len(available_features) >= 2 else available_features
        feature_cols = st.multiselect("Select feature variables:",
                                     available_features,
                                     default=default_features)

    if not feature_cols:
        st.warning("Please select at least one feature variable.")
        return

    # Prepare data
    ml_df = df[feature_cols + [target_col]].dropna()

    if len(ml_df) == 0:
        st.error("No valid data after removing missing values.")
        return

    X = ml_df[feature_cols]
    y = ml_df[target_col]

    # For logistic regression, convert target to binary
    if model_type == "Logistic Regression":
        median_val = y.median()
        y_binary = (y > median_val).astype(int)
        y = y_binary

    # Train-test split
    test_size = st.slider("Test set size (%):", 10, 50, 20) / 100
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Train model
    if st.button("Train Model"):
        try:
            if model_type == "Linear Regression":
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)

                st.success("Linear Regression model trained successfully!")
                st.metric("RMSE", f"{rmse:.4f}")
                st.metric("R² Score", f"{model.score(X_test, y_test):.4f}")

            elif model_type == "Logistic Regression":
                model = LogisticRegression(random_state=42, max_iter=1000)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')

                st.success("Logistic Regression model trained successfully!")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{accuracy:.4f}")
                with col2:
                    st.metric("Precision", f"{precision:.4f}")
                with col3:
                    st.metric("Recall", f"{recall:.4f}")
                with col4:
                    st.metric("F1-Score", f"{f1:.4f}")

            elif model_type == "Decision Tree":
                model = DecisionTreeClassifier(random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')

                st.success("Decision Tree model trained successfully!")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{accuracy:.4f}")
                with col2:
                    st.metric("Precision", f"{precision:.4f}")
                with col3:
                    st.metric("Recall", f"{recall:.4f}")
                with col4:
                    st.metric("F1-Score", f"{f1:.4f}")

        except Exception as e:
            st.error(f"Error training model: {e}")

if __name__ == "__main__":
    main()