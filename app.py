import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

# Configure Pandas settings
pd.set_option("styler.render.max_elements", 4000000)
pd.set_option('display.max_columns', None)

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main {padding: 2rem;}
    .stButton > button {width: 100%;}
    .reportview-container {background: #fafafa;}
    .sidebar .sidebar-content {background: #ffffff;}
    .Widget>label {font-size: 16px;}
    .st-emotion-cache-1y4p8pa {padding: 2rem;}
</style>
""", unsafe_allow_html=True)

# Title and Introduction
st.title("🔍 Credit Card Fraud Detection")
st.markdown("""
This application demonstrates an end-to-end machine learning pipeline for detecting credit card fraud using anomaly detection techniques.
The project uses advanced algorithms like Isolation Forest and One-Class SVM to identify potentially fraudulent transactions.
""")

# Sidebar
with st.sidebar:
    st.header("📋 Pipeline Steps")
    pipeline_step = st.radio(
        "Select Step",
        ["1. Data Upload & Preprocessing", 
         "2. Exploratory Data Analysis",
         "3. Model Training",
         "4. Model Evaluation",
         "5. Real-time Prediction"]
    )
    
    st.markdown("---")
    
    # Model Settings
    st.header("⚙️ Model Settings")
    model_choice = st.selectbox(
        "Select Model",
        ["Isolation Forest", "One-Class SVM"]
    )
    
    if model_choice == "Isolation Forest":
        contamination = st.slider("Contamination Factor", 0.01, 0.1, 0.01)
        n_estimators = st.slider("Number of Estimators", 50, 200, 100)
    else:
        nu = st.slider("Nu Parameter", 0.01, 0.1, 0.01)
        kernel = st.selectbox("Kernel", ["rbf", "linear", "poly"])

# Main Content
if pipeline_step == "1. Data Upload & Preprocessing":
    st.header("📥 Data Upload & Preprocessing")
    
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        st.subheader("Preprocessing Options")
        col1, col2 = st.columns(2)
        
        with col1:
            scaling_method = st.selectbox(
                "Scaling Method",
                ["RobustScaler", "StandardScaler", "None"]
            )
            
        with col2:
            handle_missing = st.selectbox(
                "Handle Missing Values",
                ["Median", "Mean", "Drop", "None"]
            )
        
        if st.button("Apply Preprocessing"):
            with st.spinner("Preprocessing data..."):
                # Handle missing values
                if handle_missing != "None":
                    if handle_missing == "Drop":
                        df = df.dropna()
                    elif handle_missing == "Median":
                        df = df.fillna(df.median())
                    else:
                        df = df.fillna(df.mean())
                
                # Apply scaling
                if scaling_method != "None":
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if scaling_method == "RobustScaler":
                        scaler = RobustScaler()
                    else:
                        scaler = StandardScaler()
                    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
                
                st.session_state['preprocessed_data'] = df
                st.success("✅ Preprocessing complete!")
                
                st.subheader("Preprocessed Data Preview")
                st.dataframe(df.head())
                
                st.download_button(
                    "Download Preprocessed Data",
                    df.to_csv(index=False).encode('utf-8'),
                    "preprocessed_data.csv",
                    "text/csv"
                )

elif pipeline_step == "2. Exploratory Data Analysis":
    st.header("📊 Exploratory Data Analysis")
    
    if 'preprocessed_data' not in st.session_state:
        st.warning("⚠️ Please upload and preprocess data first!")
    else:
        df = st.session_state['preprocessed_data']
        
        # Data Overview with enhanced metrics
        st.subheader("📈 Data Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", f"{len(df):,}")
        with col2:
            st.metric("Features", len(df.columns))
        with col3:
            st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
        with col4:
            if 'Class' in df.columns:
                fraud_percentage = (df['Class'].sum() / len(df)) * 100
                st.metric("Fraud Percentage", f"{fraud_percentage:.2f}%")
        
        # Data Summary Statistics
        st.subheader("📊 Statistical Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Descriptive Statistics")
            st.dataframe(df.describe())
        with col2:
            if 'Class' in df.columns:
                st.write("Class Distribution")
                class_dist = df['Class'].value_counts()
                fig = px.pie(values=class_dist.values, 
                           names=['Normal', 'Fraud'],
                           title="Transaction Distribution",
                           color_discrete_sequence=['#00CC96', '#EF553B'])
                st.plotly_chart(fig)
        
        # Feature Analysis
        st.subheader("🔍 Feature Analysis")
        
        # Feature Selection
        col1, col2 = st.columns(2)
        with col1:
            feature = st.selectbox("Select Feature for Analysis", df.columns)
        with col2:
            if 'Class' in df.columns:
                show_by_class = st.checkbox("Show Distribution by Class", value=True)
        
        # Distribution Analysis
        st.write("#### Distribution Analysis")
        col1, col2 = st.columns(2)
        with col1:
            if show_by_class and 'Class' in df.columns:
                fig = px.histogram(df, x=feature, color='Class',
                                 title=f"Distribution of {feature} by Class",
                                 color_discrete_map={0: '#00CC96', 1: '#EF553B'},
                                 marginal="box")
            else:
                fig = px.histogram(df, x=feature,
                                 title=f"Distribution of {feature}",
                                 marginal="box")
            fig.update_layout(bargap=0.1)
            st.plotly_chart(fig)
        
        with col2:
            fig = px.box(df, y=feature,
                        title=f"Box Plot of {feature}")
            if show_by_class and 'Class' in df.columns:
                fig = px.box(df, y=feature, color='Class',
                           title=f"Box Plot of {feature} by Class",
                           color_discrete_map={0: '#00CC96', 1: '#EF553B'})
            st.plotly_chart(fig)
        
        # Feature Statistics
        st.write("#### Feature Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean", f"{df[feature].mean():.2f}")
            st.metric("Skewness", f"{df[feature].skew():.2f}")
        with col2:
            st.metric("Median", f"{df[feature].median():.2f}")
            st.metric("Kurtosis", f"{df[feature].kurtosis():.2f}")
        with col3:
            st.metric("Std Dev", f"{df[feature].std():.2f}")
            missing_pct = (df[feature].isnull().sum() / len(df)) * 100
            st.metric("Missing Values", f"{missing_pct:.2f}%")
        
        # Correlation Analysis
        st.subheader("🔗 Correlation Analysis")
        
        # Correlation Matrix with enhanced visualization
        corr = df.corr()
        
        # Option to filter correlation strength
        col1, col2 = st.columns(2)
        with col1:
            corr_threshold = st.slider("Correlation Strength Threshold", 0.0, 1.0, 0.5)
        with col2:
            sort_by_feature = st.checkbox("Sort by Selected Feature", value=True)
        
        if sort_by_feature:
            corr_with_feature = corr[feature].sort_values(ascending=False)
            strong_corr = corr_with_feature[abs(corr_with_feature) > corr_threshold]
            if len(strong_corr) > 0:
                st.write(f"#### Strong Correlations with {feature}")
                fig = px.bar(
                    x=strong_corr.index,
                    y=strong_corr.values,
                    title=f"Features Correlated with {feature} (|correlation| > {corr_threshold})",
                    labels={'x': 'Feature', 'y': 'Correlation Coefficient'}
                )
                fig.update_traces(marker_color=['#EF553B' if x < 0 else '#00CC96' for x in strong_corr.values])
                st.plotly_chart(fig)
        
        # Full Correlation Matrix
        st.write("#### Correlation Matrix Heatmap")
        fig = px.imshow(
            corr,
            title="Feature Correlation Matrix",
            color_continuous_scale="RdBu",
            aspect="auto"
        )
        fig.update_layout(height=800)
        st.plotly_chart(fig)
        
        # Time Series Analysis if timestamp is available
        if any(df.columns.str.contains('time', case=False)) or any(df.columns.str.contains('date', case=False)):
            st.subheader("📅 Time Series Analysis")
            time_cols = df.columns[df.columns.str.contains('time', case=False) | df.columns.str.contains('date', case=False)]
            time_col = st.selectbox("Select Time Column", time_cols)
            
            try:
                df[time_col] = pd.to_datetime(df[time_col])
                df_time = df.set_index(time_col)
                
                if 'Class' in df.columns:
                    daily_fraud = df_time[df_time['Class'] == 1].resample('D').size()
                    daily_normal = df_time[df_time['Class'] == 0].resample('D').size()
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=daily_normal.index, y=daily_normal.values,
                                           name='Normal', line=dict(color='#00CC96')))
                    fig.add_trace(go.Scatter(x=daily_fraud.index, y=daily_fraud.values,
                                           name='Fraud', line=dict(color='#EF553B')))
                    fig.update_layout(title="Daily Transaction Patterns",
                                    xaxis_title="Date",
                                    yaxis_title="Number of Transactions")
                    st.plotly_chart(fig)
            except:
                st.warning("⚠️ Could not parse time column for analysis")

elif pipeline_step == "3. Model Training":
    st.header("🤖 Model Training")
    
    if 'preprocessed_data' not in st.session_state:
        st.warning("⚠️ Please upload and preprocess data first!")
    else:
        df = st.session_state['preprocessed_data']
        
        # Training settings
        st.subheader("Training Settings")
        test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2)
        
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                # Prepare data
                X = df.drop('Class', axis=1) if 'Class' in df.columns else df
                
                # Split data
                X_train, X_test = train_test_split(X, test_size=test_size, random_state=42)
                
                # Train model
                if model_choice == "Isolation Forest":
                    model = IsolationForest(
                        contamination=contamination,
                        n_estimators=n_estimators,
                        random_state=42
                    )
                else:
                    model = OneClassSVM(
                        kernel=kernel,
                        nu=nu
                    )
                
                model.fit(X_train)
                
                # Save model and data
                st.session_state['model'] = model
                st.session_state['X_test'] = X_test
                
                st.success("✅ Model training complete!")

elif pipeline_step == "4. Model Evaluation":
    st.header("📈 Model Evaluation")
    
    if 'model' not in st.session_state or 'X_test' not in st.session_state:
        st.warning("⚠️ Please train a model first!")
    else:
        model = st.session_state['model']
        X_test = st.session_state['X_test']
        
        # Make predictions and convert to binary format (0 for normal, 1 for anomaly)
        predictions = model.predict(X_test)
        predictions = np.where(predictions == -1, 1, 0)  # Convert -1 to 1 (anomaly) and 1 to 0 (normal)
        
        # Calculate metrics if we have actual labels
        if 'preprocessed_data' in st.session_state and 'Class' in st.session_state['preprocessed_data'].columns:
            y_test = st.session_state['preprocessed_data'].loc[X_test.index, 'Class'].astype(int)
            
            # Display confusion matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, predictions)
            
            # Create confusion matrix using go.Figure
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Normal', 'Fraud'],
                y=['Normal', 'Fraud'],
                colorscale='RdBu_r',
                showscale=False
            ))
            
            # Add text annotations
            for i in range(len(cm)):
                for j in range(len(cm[i])):
                    fig.add_annotation(
                        x=j,
                        y=i,
                        text=str(cm[i, j]),
                        showarrow=False,
                        font=dict(color="white" if cm[i, j] > cm.max()/2 else "black")
                    )
            
            fig.update_layout(
                title='Confusion Matrix',
                xaxis_title="Predicted Label",
                yaxis_title="True Label",
                xaxis=dict(side="bottom"),
                width=500,
                height=500
            )
            
            st.plotly_chart(fig)
            
            # Classification report
            st.subheader("Classification Report")
            report = classification_report(y_test, predictions, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)
            
            # Additional metrics
            st.subheader("Additional Metrics")
            col1, col2, col3 = st.columns(3)
            with col1:
                accuracy = (cm[0,0] + cm[1,1]) / np.sum(cm)
                st.metric("Accuracy", f"{accuracy:.2%}")
            with col2:
                precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
                st.metric("Precision", f"{precision:.2%}")
            with col3:
                recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
                st.metric("Recall", f"{recall:.2%}")
        
        # Anomaly scores distribution
        st.subheader("Anomaly Score Distribution")
        scores = model.score_samples(X_test)
        
        # Create a DataFrame for better plotting
        score_df = pd.DataFrame({
            'Anomaly Score': scores,
            'Prediction': ['Fraud' if p == 1 else 'Normal' for p in predictions]
        })
        
        fig = px.histogram(
            score_df, 
            x='Anomaly Score',
            color='Prediction',
            nbins=50,
            title="Distribution of Anomaly Scores",
            color_discrete_map={'Normal': '#4CAF50', 'Fraud': '#FF5722'}
        )
        
        fig.update_layout(
            xaxis_title="Anomaly Score",
            yaxis_title="Count",
            showlegend=True,
            bargap=0.1
        )
        
        st.plotly_chart(fig)
        
        # Add threshold analysis
        st.subheader("Threshold Analysis")
        threshold = st.slider(
            "Anomaly Score Threshold",
            float(scores.min()),
            float(scores.max()),
            float(scores.mean()),
            step=0.01
        )
        
        # Calculate predictions based on threshold
        threshold_predictions = np.where(scores < threshold, 1, 0)
        if 'Class' in st.session_state['preprocessed_data'].columns:
            threshold_cm = confusion_matrix(y_test, threshold_predictions)
            threshold_accuracy = (threshold_cm[0,0] + threshold_cm[1,1]) / np.sum(threshold_cm)
            threshold_precision = threshold_cm[1,1] / (threshold_cm[1,1] + threshold_cm[0,1]) if (threshold_cm[1,1] + threshold_cm[0,1]) > 0 else 0
            threshold_recall = threshold_cm[1,1] / (threshold_cm[1,1] + threshold_cm[1,0]) if (threshold_cm[1,1] + threshold_cm[1,0]) > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Threshold Accuracy", f"{threshold_accuracy:.2%}")
            with col2:
                st.metric("Threshold Precision", f"{threshold_precision:.2%}")
            with col3:
                st.metric("Threshold Recall", f"{threshold_recall:.2%}")

else:  # Real-time Prediction
    st.header("🎯 Real-time Prediction")
    
    if 'model' not in st.session_state:
        st.warning("⚠️ Please train a model first!")
    else:
        st.subheader("Enter Transaction Details")
        
        # Create input form based on features
        if 'preprocessed_data' in st.session_state:
            features = st.session_state['preprocessed_data'].drop('Class', axis=1).columns if 'Class' in st.session_state['preprocessed_data'].columns else st.session_state['preprocessed_data'].columns
            
            input_data = {}
            for feature in features:
                input_data[feature] = st.number_input(f"Enter {feature}", value=0.0)
            
            if st.button("Predict"):
                # Prepare input data
                X = pd.DataFrame([input_data])
                
                # Make prediction
                prediction = st.session_state['model'].predict(X)[0]
                score = st.session_state['model'].score_samples(X)[0]
                
                # Display result
                if prediction == -1:
                    st.error("⚠️ Potential Fraud Detected!")
                else:
                    st.success("✅ Transaction appears normal")
                
                st.metric("Anomaly Score", f"{score:.4f}")