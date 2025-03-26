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
import hashlib
import datetime
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from geopy.geocoders import Nominatim
import pydeck as pdk

# Set page configuration
st.set_page_config(page_title="Flood Prediction System", layout="wide")

# Email configuration (replace with your SMTP details)
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
SENDER_EMAIL = "achums1212@gmail.com"  # Replace with your email
SENDER_PASSWORD = "vvmirqrkzgzwloln"  # Replace with your app password

# Session state initialization
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'alerts' not in st.session_state:
    st.session_state.alerts = []

# User authentication functions
def init_users():
    if not os.path.exists('users.json'):
        default_user = {
            "admin": {
                "password": hashlib.sha256("admin123".encode()).hexdigest(),
                "email": "admin@example.com",
                "alert_threshold": 0.7,
                "alert_channels": ["email", "dashboard"],
                "locality": "Default Locality"
            }
        }
        with open('users.json', 'w') as f:
            json.dump(default_user, f)

def get_users():
    try:
        with open('users.json', 'r') as f:
            return json.load(f)
    except:
        return {}

def save_users(users):
    with open('users.json', 'w') as f:
        json.dump(users, f, indent=4)

def verify_user(username, password):
    users = get_users()
    if username in users:
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        if users[username]["password"] == hashed_password:
            return True
    return False

def register_user(username, password, email, locality):
    users = get_users()
    if username in users:
        return False
    
    users[username] = {
        "password": hashlib.sha256(password.encode()).hexdigest(),
        "email": email,
        "alert_threshold": 0.7,
        "alert_channels": ["dashboard", "email"],  # Enable email by default
        "locality": locality
    }
    
    save_users(users)
    return True

# Alert system functions
def init_alerts():
    if not os.path.exists('alerts.json'):
        with open('alerts.json', 'w') as f:
            json.dump([], f)

def get_alerts():
    try:
        with open('alerts.json', 'r') as f:
            return json.load(f)
    except:
        return []

def save_alert(username, alert_message, risk_level, details=None):
    alerts = get_alerts()
    
    alert = {
        "username": username,
        "message": alert_message,
        "risk_level": risk_level,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "details": details or {}
    }
    
    alerts.append(alert)
    
    with open('alerts.json', 'w') as f:
        json.dump(alerts, f, indent=4)
    
    st.session_state.alerts.append(alert)

def send_email_alert(receiver_email, alert_message, risk_level, details=None):
    try:
        # Create a unique ID for this alert to prevent duplicates
        alert_id = hashlib.md5(f"{receiver_email}{alert_message}{risk_level}".encode()).hexdigest()
        
        # Check if this alert was already sent
        if os.path.exists('sent_alerts.json'):
            with open('sent_alerts.json', 'r') as f:
                sent_alerts = json.load(f)
        else:
            sent_alerts = []
            
        if alert_id in sent_alerts:
            return True  # Already sent
            
        # Create detailed email content
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = receiver_email
        msg['Subject'] = f"🚨 Flood Alert - Risk Level: {risk_level:.0%}"
        
        # Enhanced email body with more details
        body = f"""
        ==============================
        FLOOD RISK ALERT NOTIFICATION
        ==============================
        
        🚨 ALERT: {alert_message}
        
        🔍 Risk Assessment:
        - Risk Level: {risk_level:.0%} ({'HIGH' if risk_level >= 0.8 else 'MEDIUM' if risk_level >= 0.5 else 'LOW'})
        - Time Detected: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        
        📍 Location Details:
        - Address: {details.get('Location', 'N/A')}
        - Coordinates: {details.get('Coordinates', 'N/A')}
        
        ⚙️ Prediction Model:
        - Model Used: {details.get('Model', 'N/A')}
        - Confidence: {risk_level:.0%}
        
        🛡️ Recommended Actions:
        - Monitor local flood warnings
        - Prepare emergency supplies
        - Follow evacuation routes if necessary
        
        This is an automated alert from the Flood Prediction System.
        Please verify with official sources before taking action.
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Connect and send
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
            
        # Record this alert as sent
        sent_alerts.append(alert_id)
        with open('sent_alerts.json', 'w') as f:
            json.dump(sent_alerts, f)
            
        return True
        
    except Exception as e:
        st.error(f"Failed to send email: {str(e)}")
        return False

def send_alerts_to_all_users(alert_message, risk_level, details=None):
    users = get_users()
    for username, user_data in users.items():
        if "email" in user_data.get("alert_channels", []) and risk_level >= user_data.get("alert_threshold", 0.7):
            send_email_alert(
                user_data["email"],
                alert_message,
                risk_level,
                details
            )

# Function to load data
@st.cache_data
def load_data():
    # Load the dataset with latitude and longitude
    if os.path.exists('flood_risk_dataset_india.csv'):
        df = pd.read_csv('flood_risk_dataset_india.csv')
        # Drop 'Land Cover' and 'Soil Type' columns
        df = df.drop(columns=['Land Cover', 'Soil Type'])
        return df
    else:
        st.error("Dataset file 'flood_risk_dataset_india.csv' not found.")
        return None

# Function to preprocess data
def preprocess_data(df):
    # Exclude 'Latitude' and 'Longitude' from features
    X = df.drop(columns=['Latitude', 'Longitude', 'Flood Occurred'])  # Features
    y = df['Flood Occurred']  # Target
    
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

def display_location_on_map(latitude, longitude):
    st.subheader("Detailed Location Map")
    
    # Create a pydeck map with a more visible red marker
    layers = [
        pdk.Layer(
            "ScatterplotLayer",
            data=pd.DataFrame({'lat': [latitude], 'lon': [longitude]}),
            get_position='[lon, lat]',
            get_color='[255, 0, 0, 200]',  # Bright red color
            get_radius=200,  # Larger radius
            pickable=True
        ),
        pdk.Layer(
            "TextLayer",
            data=pd.DataFrame({'lat': [latitude], 'lon': [longitude], 'text': ['Flood Risk Area']}),
            get_position='[lon, lat]',
            get_text='text',
            get_color='[255, 255, 255, 255]',  # White text
            get_size=16,
            get_alignment_baseline="'bottom'"
        )
    ]
    
    view_state = pdk.ViewState(
        latitude=latitude,
        longitude=longitude,
        zoom=12,
        pitch=50
    )
    
    r = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style='mapbox://styles/mapbox/satellite-streets-v11',
        tooltip={
            "html": "<b>Flood Risk Location</b><br/>Latitude: {lat}<br/>Longitude: {lon}",
            "style": {
                "backgroundColor": "red",
                "color": "white"
            }
        }
    )
    
    st.pydeck_chart(r)

# Function to get place name from latitude and longitude
def get_place_name(latitude, longitude):
    geolocator = Nominatim(user_agent="flood_prediction_system")
    location = geolocator.reverse((latitude, longitude), exactly_one=True)
    if location:
        return location.address
    return "Location not found"

# Function to send alerts to users in the same locality
def send_locality_alerts(locality, alert_message, risk_level, details=None):
    users = get_users()
    for username, user_data in users.items():
        if user_data.get("locality") == locality:
            save_alert(username, alert_message, risk_level, details)

# Login page
def login_page():
    st.title("Flood Prediction System - Login")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login"):
            if verify_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Login successful!")
            else:
                st.error("Invalid username or password")
    
    with tab2:
        new_username = st.text_input("Choose Username", key="reg_username")
        new_password = st.text_input("Choose Password", type="password", key="reg_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
        email = st.text_input("Email Address", key="reg_email")
        locality = st.text_input("Locality", key="reg_locality")
        
        if st.button("Register"):
            if new_password != confirm_password:
                st.error("Passwords do not match")
            elif len(new_password) < 6:
                st.error("Password must be at least 6 characters long")
            elif not new_username or not email or not locality:
                st.error("Username, email, and locality are required")
            else:
                if register_user(new_username, new_password, email, locality):
                    st.success("Registration successful! You can now login.")
                else:
                    st.error("Username already exists")

# Alert Dashboard
def alert_dashboard():
    st.header("Alert Dashboard")
    
    # Display user-specific alerts
    user_alerts = [alert for alert in st.session_state.alerts if alert["username"] == st.session_state.username]
    
    if not user_alerts:
        st.info("No alerts to display")
    else:
        # Sort alerts by timestamp (most recent first)
        user_alerts.sort(key=lambda x: x["timestamp"], reverse=True)
        
        # Display alerts
        for alert in user_alerts:
            if alert["risk_level"] >= 0.8:
                severity = "🔴 High"
                color = "red"
            elif alert["risk_level"] >= 0.5:
                severity = "🟠 Medium"
                color = "orange"
            else:
                severity = "🟡 Low"
                color = "yellow"
                
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"### {severity} Risk Alert")
                    st.markdown(f"**{alert['message']}**")
                    st.markdown(f"*{alert['timestamp']}*")
                
                with col2:
                    st.markdown(f"<div style='background-color:{color};padding:10px;border-radius:5px;text-align:center;color:white;'>Risk: {alert['risk_level']:.2f}</div>", unsafe_allow_html=True)
                
                # Check if details exist and are a dictionary
                if alert.get("details") and isinstance(alert["details"], dict):
                    with st.expander("View Details"):
                        for key, value in alert["details"].items():
                            st.write(f"**{key}:** {value}")
                
                st.markdown("---")

# User settings
def user_settings():
    st.header("Alert Settings")
    
    users = get_users()
    user_data = users[st.session_state.username]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Profile Information")
        email = st.text_input("Email Address", value=user_data["email"])
        locality = st.text_input("Locality", value=user_data.get("locality", ""))
        
    with col2:
        st.subheader("Alert Preferences")
        alert_threshold = st.slider("Alert Threshold", 0.0, 1.0, float(user_data.get("alert_threshold", 0.7)), 0.05)
        
        alert_channels = st.multiselect(
            "Alert Notification Channels",
            options=["email", "dashboard"],
            default=user_data.get("alert_channels", ["dashboard"])
        )
    
    if st.button("Save Settings"):
        users[st.session_state.username]["email"] = email
        users[st.session_state.username]["locality"] = locality
        users[st.session_state.username]["alert_threshold"] = alert_threshold
        users[st.session_state.username]["alert_channels"] = alert_channels
        save_users(users)
        st.success("Settings saved successfully!")

# Main function for Streamlit app
def main():
    # Initialize users and alerts
    init_users()
    init_alerts()
    
    # Check if user is logged in
    if not st.session_state.logged_in:
        login_page()
    else:
        # Load alerts for the logged-in user (for notification display)
        if os.path.exists('alerts.json'):
            with open('alerts.json', 'r') as f:
                all_alerts = json.load(f)
                st.session_state.alerts = all_alerts
        
        # Display welcome message and logout button
        col1, col2 = st.columns([3, 1])
        with col1:
            st.title("Flood Prediction System")
            st.write(f"Welcome, {st.session_state.username}!")
        with col2:
            if st.button("Logout"):
                st.session_state.logged_in = False
                st.session_state.username = None
                st.session_state.alerts = []
                st.success("Logged out successfully!")
        
        # Sidebar navigation
        page = st.sidebar.selectbox(
            "Navigate", 
            ["Data Overview", "Model Training", "Model Evaluation", "Prediction", "Visualizations", "Alert Dashboard", "Settings"]
        )
        
        # Load data
        df = load_data()
        
        if page == "Data Overview":
            st.header("Data Overview")
            st.write("This dataset contains features relevant to flood prediction.")
            
            # Show data
            st.subheader("Sample Data")
            st.dataframe(df.head())
            
            # Data statistics (excluding lat/long)
            st.subheader("Data Statistics")
            st.dataframe(df.drop(columns=['Latitude', 'Longitude']).describe())
            
            # Data distribution
            st.subheader("Feature Distributions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                feature = st.selectbox("Select Feature for Histogram", df.drop(columns=['Latitude', 'Longitude']).columns)
                fig, ax = plt.subplots()
                sns.histplot(data=df, x=feature, hue='Flood Occurred', kde=True, ax=ax)
                st.pyplot(fig)
            
            with col2:
                x_feature = st.selectbox("Select X-axis Feature", df.drop(columns=['Latitude', 'Longitude', 'Flood Occurred']).columns)
                y_feature = st.selectbox("Select Y-axis Feature", [col for col in df.drop(columns=['Latitude', 'Longitude', 'Flood Occurred']).columns if col != x_feature])
                
                fig, ax = plt.subplots()
                sns.scatterplot(data=df, x=x_feature, y=y_feature, hue='Flood Occurred', ax=ax)
                st.pyplot(fig)
            
            # Correlation matrix (excluding lat/long)
            st.subheader("Correlation Matrix")
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(df.drop(columns=['Latitude', 'Longitude']).corr(), annot=True, cmap="coolwarm", ax=ax)
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
                    imp_plot = plot_feature_importance(
                        models[selected_model_imp], 
                        df.drop(columns=['Latitude', 'Longitude', 'Flood Occurred']).columns
                    )
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
                st.subheader("Select Location for Prediction")
                
                # Load dataset
                df = load_data()
                if df is None:
                    return
                
                # Allow user to select latitude and longitude
                selected_lat = st.selectbox("Select Latitude", df['Latitude'].unique())
                selected_lon = st.selectbox("Select Longitude", df[df['Latitude'] == selected_lat]['Longitude'].unique())
                
                # Filter the dataset for the selected latitude and longitude
                selected_row = df[(df['Latitude'] == selected_lat) & (df['Longitude'] == selected_lon)]
                
                if selected_row.empty:
                    st.error("No data found for the selected location.")
                    return
                
                # Display the selected row's data (excluding 'Latitude' and 'Longitude')
                st.write("Selected Row Data:")
                st.dataframe(selected_row.drop(columns=['Latitude', 'Longitude']))
                
                # Get place name from latitude and longitude
                place_name = get_place_name(selected_lat, selected_lon)
                st.subheader("Location Details")
                st.text_area("Place Name", value=place_name, height=68, disabled=True)
                
                # Display enhanced location map
                display_location_on_map(selected_lat, selected_lon)
                
                # Extract features for prediction (excluding 'Latitude', 'Longitude', and 'Flood Occurred')
                input_data = selected_row.drop(columns=['Latitude', 'Longitude', 'Flood Occurred'])
                
                # Select model
                selected_model = st.selectbox("Select Model for Prediction", list(models.keys()))
                
                # Prediction button
                if st.button("Predict"):
                    prediction, probability = predict_flood(models[selected_model], input_data, scaler)
                    
                    # Get user's alert threshold
                    users = get_users()
                    user_alert_threshold = users[st.session_state.username].get("alert_threshold", 0.7)
                    
                    # Check if we should create an alert
                    if probability >= user_alert_threshold:
                        risk_level = probability
                        
                        # Determine alert severity text
                        if probability >= 0.8:
                            severity = "HIGH"
                        elif probability >= 0.5:
                            severity = "MEDIUM"
                        else:
                            severity = "LOW"
                        
                        alert_message = f"{severity} flood risk detected at {place_name}"
                        
                        # Prepare details for the alert
                        details = {
                            "Location": place_name,
                            "Coordinates": f"Lat: {selected_lat}, Lon: {selected_lon}",
                            "Model": selected_model,
                            **selected_row.drop(columns=['Latitude', 'Longitude', 'Flood Occurred']).iloc[0].to_dict()
                        }
                        
                        # Save the alert for the current user
                        save_alert(st.session_state.username, alert_message, risk_level, details)
                        
                        # Send alerts to ALL users who have email notifications enabled
                        send_alerts_to_all_users(alert_message, risk_level, details)
                        
                        # Show alert notification
                        st.warning(f"⚠️ Alert created: {alert_message}")
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
                    
                    # Select features for visualization (excluding lat/long)
                    available_features = [col for col in df.columns if col not in ['Latitude', 'Longitude', 'Flood Occurred']]
                    feature1 = st.selectbox("Select First Feature", available_features, index=0)
                    feature2 = st.selectbox("Select Second Feature", 
                                             [col for col in available_features if col != feature1], 
                                             index=0)
                    
                    # Select model
                    model_name = st.selectbox("Select Model", list(models.keys()))
                    
                    # Create mesh grid for decision boundary
                    X = df[[feature1, feature2]].values
                    y = df['Flood Occurred'].values
                    
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
        
        elif page == "Alert Dashboard":
            alert_dashboard()
        
        elif page == "Settings":
            user_settings()

if __name__ == "__main__":
    main()