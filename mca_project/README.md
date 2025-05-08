# 🌊 Flood Prediction System

**A machine learning-based web application for predicting flood risks with real-time alerts**

![Flood Prediction System Screenshot](https://via.placeholder.com/800x400?text=Flood+Prediction+System+Screenshot)

## 📌 Overview

This system predicts flood risks using various machine learning models and provides real-time alerts to users via email and dashboard notifications. Designed specifically for Indian geographical data, it helps authorities and citizens prepare for potential flood events.

## ✨ Key Features

- **Multiple ML Models**: 
  - Logistic Regression
  - Random Forest
  - Support Vector Machines (SVM)
  - K-Nearest Neighbors
  - Gradient Boosting

- **Interactive Dashboard**:
  - Data visualization
  - Model performance metrics
  - Probability distribution charts

- **Alert System**:
  - Email notifications
  - Dashboard alerts
  - Customizable risk thresholds

- **Geospatial Analysis**:
  - Interactive map visualization
  - Location-based predictions
  - Address lookup via coordinates

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **Machine Learning**: Scikit-learn
- **Data Visualization**: Matplotlib, Seaborn, PyDeck
- **Geocoding**: Geopy (Nominatim)
- **Email**: SMTP

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Aswin2121/Mainproject.git

   Install dependencies:

bash
pip install -r requirements.txt

🏃‍♂️ Running the Application
bash
streamlit run app.py
📂 Project Structure
flood-prediction-system/
├── app.py                # Main application file
├── requirements.txt      # Python dependencies
├── .env.example          # Environment variables template
├── data/
│   └── flood_risk_dataset_india.csv  # Sample dataset
├── models/               # Saved ML models
└── README.md             # This file
🔐 Authentication
Default Admin Credentials:

Username: admin

Password: admin123

User Registration:

Requires valid email verification

Password hashed with SHA-256

📈 Model Performance
The system evaluates models using:

Accuracy

Precision

Recall

F1 Score

ROC Curves

Confusion Matrices

📧 Alert Configuration
Users can customize:

Alert threshold (0-1 probability)

Notification channels (email/dashboard)

Location preferences

🌍 Map Features
Interactive selection of locations

Visual representation of flood risks

Address lookup from coordinates

📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request.

⚠️ Note: This is a prototype system. Always verify predictions with official flood warnings from government agencies.
