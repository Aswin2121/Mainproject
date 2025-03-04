import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os

# Set page configuration
st.set_page_config(page_title="Flood Prediction System", layout="wide")

# Function to load data
@st.cache_data
def load_data():
    # This is a sample dataset structure. In a real scenario, you would replace this with your actual dataset
    # This simulates flood data with features like rainfall, river level, soil moisture, etc.
    if os.path.exists('flood_data.csv'):
        return pd.read_csv('flood_data.csv')
    else:
        # Generate sample data for demonstration
        np.random.seed(42)
        n_samples = 1000
        
        rainfall = np.random.gamma(2, 10, n_samples)  # mm/day
        river_level = np.random.normal(5, 2, n_samples)  # meters
        soil_moisture = np.random.beta(2, 2, n_samples) * 100  # percentage
        temperature = np.random.normal(25, 5, n_samples)  # celsius
        elevation = np.random.gamma(10, 10, n_samples)  # meters above sea level
        
        # Define flood based on conditions (a simple rule for demonstration)
        flood = ((rainfall > 25) & (river_level > 7)) | ((soil_moisture > 80) & (rainfall > 15))
        
        # Create dataframe
        df = pd.DataFrame({
            'Rainfall': rainfall,
            'RiverLevel': river_level,
            'SoilMoisture': soil_moisture,
            'Temperature': temperature,
            'Elevation': elevation,
            'Flood': flood.astype(int)
        })
        
        # Save the generated data
        df.to_csv('flood_data.csv', index=False)
        return df

# Function to preprocess data
def preprocess_data(df):
    # Split features and target
    X = df.drop('Flood', axis=1)
    y = df['Flood']
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# Function to train models
def train_models(X_train, y_train):
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        
    return trained_models

# Function to evaluate models
def evaluate_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results[name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        }
    
    return results

# Function to make predictions
def predict_flood(model, input_data, scaler):
    # Scale input data
    input_scaled = scaler.transform(input_data)
    
    # Get prediction and probability
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)
    
    return prediction[0], probability[0][1]  # Return prediction and probability of flood

# Function to save models
def save_models(models, scaler):
    for name, model in models.items():
        joblib.dump(model, f"{name.replace(' ', '_').lower()}_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

# Function to load models
def load_models():
    model_names = ['Logistic Regression', 'Random Forest', 'SVM', 'K-Nearest Neighbors', 'Gradient Boosting']
    loaded_models = {}
    
    for name in model_names:
        filename = f"{name.replace(' ', '_').lower()}_model.pkl"
        if os.path.exists(filename):
            loaded_models[name] = joblib.load(filename)
    
    scaler = None
    if os.path.exists("scaler.pkl"):
        scaler = joblib.load("scaler.pkl")
    
    return loaded_models, scaler

# Function to plot feature importance
def plot_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title('Feature Importance')
        plt.bar(range(len(indices)), importances[indices], align='center')
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        return plt
    return None

# Function to plot confusion matrix
def plot_confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    return plt

# Function to plot probability distribution
def plot_probability_distribution(models, X_test):
    plt.figure(figsize=(12, 6))
    
    for name, model in models.items():
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_test)[:, 1]
            sns.kdeplot(probabilities, label=name)
    
    plt.title('Probability Distribution of Flood Prediction')
    plt.xlabel('Probability of Flood')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    return plt

# Main function for Streamlit app
def main():
    st.title("Flood Prediction System")
    
    # Sidebar navigation
    page = st.sidebar.selectbox("Navigate", ["Data Overview", "Model Training", "Model Evaluation", "Prediction", "Visualizations"])
    
    # Load data
    df = load_data()
    
    if page == "Data Overview":
        st.header("Data Overview")
        st.write("This dataset contains features relevant to flood prediction.")
        
        # Show data
        st.subheader("Sample Data")
        st.dataframe(df.head())
        
        # Data statistics
        st.subheader("Data Statistics")
        st.dataframe(df.describe())
        
        # Data distribution
        st.subheader("Feature Distributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            feature = st.selectbox("Select Feature for Histogram", df.columns)
            fig, ax = plt.subplots()
            sns.histplot(data=df, x=feature, hue='Flood', kde=True, ax=ax)
            st.pyplot(fig)
        
        with col2:
            x_feature = st.selectbox("Select X-axis Feature", df.drop('Flood', axis=1).columns)
            y_feature = st.selectbox("Select Y-axis Feature", [col for col in df.drop('Flood', axis=1).columns if col != x_feature])
            
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=x_feature, y=y_feature, hue='Flood', ax=ax)
            st.pyplot(fig)
        
        # Correlation matrix
        st.subheader("Correlation Matrix")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    
    elif page == "Model Training":
        st.header("Model Training")
        
        # Check if models already exist
        model_files = [f for f in os.listdir() if f.endswith('_model.pkl')]
        
        if model_files:
            st.info("Models have already been trained. You can retrain them if needed.")
            if st.button("Retrain Models"):
                X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_data(df)
                models = train_models(X_train_scaled, y_train)
                save_models(models, scaler)
                st.success("Models retrained successfully!")
        else:
            st.info("Models need to be trained.")
            if st.button("Train Models"):
                X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_data(df)
                models = train_models(X_train_scaled, y_train)
                save_models(models, scaler)
                st.success("Models trained successfully!")
        
        # Explain the models
        st.subheader("Model Descriptions")
        
        st.markdown("""
        1. **Logistic Regression**: A linear model that predicts the probability of flood occurrence.
        2. **Random Forest**: An ensemble learning method that builds multiple decision trees and merges their predictions.
        3. **Support Vector Machine (SVM)**: A powerful classifier that finds the optimal hyperplane to separate flood and non-flood conditions.
        4. **K-Nearest Neighbors**: Classifies based on the majority class of the K nearest data points.
        5. **Gradient Boosting**: An ensemble technique that builds models sequentially, with each new model correcting errors from previous ones.
        """)
    
    elif page == "Model Evaluation":
        st.header("Model Evaluation")
        
        # Check if models exist
        models, scaler = load_models()
        
        if not models:
            st.warning("No trained models found. Please go to the Model Training page and train the models first.")
        else:
            # Evaluate models
            X_train_scaled, X_test_scaled, y_train, y_test, _ = preprocess_data(df)
            results = evaluate_models(models, X_test_scaled, y_test)
            
            # Display results
            st.subheader("Model Performance Metrics")
            
            # Prepare data for display
            results_df = pd.DataFrame.from_dict(results, orient='index')
            
            # Display metrics
            st.dataframe(results_df.style.highlight_max(axis=0))
            
            # Plot metrics comparison
            st.subheader("Model Comparison")
            
            metric = st.selectbox("Select Metric for Comparison", ["Accuracy", "Precision", "Recall", "F1 Score"])
            
            fig, ax = plt.subplots(figsize=(10, 6))
            results_df[metric].sort_values().plot(kind='barh', ax=ax)
            ax.set_title(f"Model Comparison by {metric}")
            st.pyplot(fig)
            
            # Confusion Matrix for selected model
            st.subheader("Confusion Matrix")
            selected_model = st.selectbox("Select Model for Confusion Matrix", list(models.keys()))
            
            cm_plot = plot_confusion_matrix(models[selected_model], X_test_scaled, y_test)
            st.pyplot(cm_plot)
            
            # Feature importance for applicable models
            st.subheader("Feature Importance")
            selected_model_imp = st.selectbox("Select Model for Feature Importance", 
                                         [name for name, model in models.items() 
                                          if hasattr(model, 'feature_importances_')])
            
            if selected_model_imp:
                imp_plot = plot_feature_importance(models[selected_model_imp], df.drop('Flood', axis=1).columns)
                if imp_plot:
                    st.pyplot(imp_plot)
                else:
                    st.info("Feature importance not available for this model.")
    
    elif page == "Prediction":
        st.header("Flood Prediction")
        
        # Check if models exist
        models, scaler = load_models()
        
        if not models or scaler is None:
            st.warning("No trained models found. Please go to the Model Training page and train the models first.")
        else:
            st.subheader("Enter Input Values")
            
            # Create columns for input fields
            col1, col2 = st.columns(2)
            
            with col1:
                rainfall = st.slider("Rainfall (mm/day)", 0.0, 100.0, 20.0)
                river_level = st.slider("River Level (meters)", 0.0, 15.0, 5.0)
                soil_moisture = st.slider("Soil Moisture (%)", 0.0, 100.0, 50.0)
            
            with col2:
                temperature = st.slider("Temperature (°C)", 0.0, 40.0, 25.0)
                elevation = st.slider("Elevation (meters)", 0.0, 200.0, 50.0)
            
            # Select model
            selected_model = st.selectbox("Select Model for Prediction", list(models.keys()))
            
            # Create input dataframe
            input_data = pd.DataFrame({
                'Rainfall': [rainfall],
                'RiverLevel': [river_level],
                'SoilMoisture': [soil_moisture],
                'Temperature': [temperature],
                'Elevation': [elevation]
            })
            
            # Prediction button
            if st.button("Predict"):
                prediction, probability = predict_flood(models[selected_model], input_data, scaler)
                
                st.subheader("Prediction Result")
                
                # Create columns for prediction display
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction == 1:
                        st.error("⚠️ Flood Risk Detected ⚠️")
                    else:
                        st.success("✅ No Flood Risk Detected")
                
                with col2:
                    st.metric("Probability of Flood", f"{probability*100:.2f}%")
                
                # Compare all models
                st.subheader("Comparison Across All Models")
                
                # Collect predictions from all models
                model_predictions = {}
                for name, model in models.items():
                    pred, prob = predict_flood(model, input_data, scaler)
                    model_predictions[name] = {
                        'Prediction': 'Flood Risk' if pred == 1 else 'No Risk',
                        'Probability': f"{prob*100:.2f}%"
                    }
                
                st.dataframe(pd.DataFrame.from_dict(model_predictions, orient='index'))
                
                # Plot probability bar chart
                st.subheader("Flood Probability by Model")
                
                # Extract probabilities for plotting
                model_names = list(model_predictions.keys())
                probabilities = [float(model_predictions[name]['Probability'].replace('%', '')) / 100 for name in model_names]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.barh(model_names, probabilities)
                
                # Color bars based on prediction (red for flood, green for no flood)
                for i, bar in enumerate(bars):
                    bar.set_color('red' if probabilities[i] > 0.5 else 'green')
                
                ax.set_xlim(0, 1)
                ax.set_xlabel('Probability of Flood')
                ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.5)
                
                # Add percentage labels
                for i, v in enumerate(probabilities):
                    ax.text(v + 0.02, i, f"{v*100:.1f}%", va='center')
                
                st.pyplot(fig)
                
                # Risk factors analysis
                st.subheader("Risk Factor Analysis")
                
                # Define thresholds for risk factors (these are hypothetical and would be adjusted based on domain knowledge)
                risk_factors = {
                    'Rainfall': {'threshold': 25, 'value': rainfall, 'description': "Heavy rainfall can lead to flooding, especially when exceeding 25mm per day."},
                    'RiverLevel': {'threshold': 7, 'value': river_level, 'description': "River levels above 7 meters may cause overflows and flooding."},
                    'SoilMoisture': {'threshold': 80, 'value': soil_moisture, 'description': "Saturated soil (>80%) cannot absorb additional water, increasing flood risk."},
                }
                
                # Display risk factors
                for factor, info in risk_factors.items():
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        if info['value'] > info['threshold']:
                            st.warning(f"⚠️ High {factor}")
                        else:
                            st.info(f"✅ Normal {factor}")
                    
                    with col2:
                        # Create a progress bar
                        progress = info['value'] / (info['threshold'] * 1.5)  # Scale for visualization
                        st.progress(min(progress, 1.0))
                        st.caption(f"{info['value']} ({info['description']})")
    
    elif page == "Visualizations":
        st.header("Visualizations")
        
        # Check if models exist
        models, scaler = load_models()
        
        if not models:
            st.warning("No trained models found. Please go to the Model Training page and train the models first.")
        else:
            # Preprocess data for visualizations
            X_train_scaled, X_test_scaled, y_train, y_test, _ = preprocess_data(df)
            
            # Model selection for visualizations
            viz_type = st.selectbox("Select Visualization Type", 
                                     ["Probability Distribution", "Decision Boundaries", "ROC Curves"])
            
            if viz_type == "Probability Distribution":
                st.subheader("Probability Distribution of Flood Prediction")
                prob_plot = plot_probability_distribution(models, X_test_scaled)
                st.pyplot(prob_plot)
                
                st.markdown("""
                This plot shows how each model distributes its prediction probabilities. 
                - Peaks near 0 indicate confident predictions of no flood
                - Peaks near 1 indicate confident predictions of flood
                - A good model typically shows a bimodal distribution with clear separation
                """)
            
            elif viz_type == "Decision Boundaries":
                st.subheader("Decision Boundaries")
                st.warning("This visualization is only available for 2D feature spaces.")
                
                # Select features for visualization
                feature1 = st.selectbox("Select First Feature", df.drop('Flood', axis=1).columns, index=0)
                feature2 = st.selectbox("Select Second Feature", 
                                         [col for col in df.drop('Flood', axis=1).columns if col != feature1], 
                                         index=0)
                
                # Select model
                model_name = st.selectbox("Select Model", list(models.keys()))
                
                # Create mesh grid for decision boundary
                X = df[[feature1, feature2]].values
                y = df['Flood'].values
                
                # Split and scale
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                scaler_2d = StandardScaler()
                X_train_scaled_2d = scaler_2d.fit_transform(X_train)
                X_test_scaled_2d = scaler_2d.transform(X_test)
                
                # Train a new model on just these 2 features
                model_2d = type(models[model_name])()
                if hasattr(model_2d, 'random_state'):
                    model_2d.random_state = 42
                if isinstance(model_2d, SVC):
                    model_2d.probability = True
                
                model_2d.fit(X_train_scaled_2d, y_train)
                
                # Create a mesh grid
                h = 0.02  # Step size
                x_min, x_max = X_train_scaled_2d[:, 0].min() - 1, X_train_scaled_2d[:, 0].max() + 1
                y_min, y_max = X_train_scaled_2d[:, 1].min() - 1, X_train_scaled_2d[:, 1].max() + 1
                xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
                
                # Predict on mesh grid
                Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                
                # Plot decision boundary
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu)
                
                # Plot test points
                scatter = ax.scatter(X_test_scaled_2d[:, 0], X_test_scaled_2d[:, 1], 
                                     c=y_test, edgecolors='k', cmap=plt.cm.RdBu)
                
                ax.set_xlabel(feature1)
                ax.set_ylabel(feature2)
                ax.set_title(f'Decision Boundary - {model_name}')
                plt.colorbar(scatter)
                
                st.pyplot(fig)
                
                st.markdown("""
                This visualization shows:
                - The decision boundary of the selected model for two features
                - Blue regions represent areas predicted as no flood
                - Red regions represent areas predicted as flood
                - Points are actual test data colored by their true class
                """)
            
            elif viz_type == "ROC Curves":
                st.subheader("ROC Curves")
                
                # Plot ROC curves for all models
                fig, ax = plt.subplots(figsize=(10, 8))
                
                from sklearn.metrics import roc_curve, auc
                
                for name, model in models.items():
                    if hasattr(model, 'predict_proba'):
                        y_score = model.predict_proba(X_test_scaled)[:, 1]
                        fpr, tpr, _ = roc_curve(y_test, y_score)
                        roc_auc = auc(fpr, tpr)
                        ax.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
                
                ax.plot([0, 1], [0, 1], 'k--')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('Receiver Operating Characteristic (ROC) Curves')
                ax.legend(loc="lower right")
                
                st.pyplot(fig)
                
                st.markdown("""
                The ROC curve shows the trade-off between:
                - True Positive Rate (sensitivity): The proportion of actual flood events correctly predicted
                - False Positive Rate (1-specificity): The proportion of actual non-flood events incorrectly predicted as floods
                
                A good model has a curve that approaches the top-left corner, with a high Area Under the Curve (AUC).
                """)

if __name__ == "__main__":
    main()