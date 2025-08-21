# 🌫️ Delhi PM2.5 Prediction using Machine Learning

## 📌 Project Overview
Air pollution in Delhi is a serious health concern, with **PM2.5** being one of the most dangerous pollutants.  
This project builds a **machine learning model** to predict PM2.5 levels and provides a **Streamlit web app** for real-time predictions.  

---

## ⚙️ Features
- 📊 **Data Processing & EDA** → Cleaning, visualization, and correlation analysis  
- 🧩 **Feature Engineering** → Time-based features, lag values, and rolling averages  
- 🤖 **Model** → Trained with **XGBoost** for high accuracy  
- 💻 **Web App** → Built with **Streamlit** for interactive prediction  
- 🔍 **Feature Importance** → Displays which pollutants/variables impact PM2.5 the most  

---

## 🗂️ Project Structure
Delhi-PM2.5-Prediction/
│── Research Paper Phase 1.docx # Project documentation

│── DelhiAQI.ipynb # Jupyter Notebook (EDA + Model Training)

│── app2.py # Streamlit App Code

│── delhi_aqi.csv # Dataset (if allowed to share, else mention source)

│── aqi_model.joblib # Saved ML model

│── README.md # Project Readme

---

## 🚀 How to Run
### 1. Clone the repo
git clone https://github.com/your-username/Delhi-PM2.5-Prediction.git
cd Delhi-PM2.5-Prediction
2. Install dependencies
pip install -r requirements.txt
3. Run Streamlit App
streamlit run app2.py
📊 Example Output
Prediction Example: PM2.5 ≈ 220.55 µg/m³

Feature Importance plot (PM10 & lag features strongly influence predictions)

📌 Future Scope
🔮 Integrate Traffic Congestion Prediction

🏥 Link with Health Data (hospital admissions, respiratory cases)

📈 Build an Urban Health Risk Index (UHRI) for smarter public health forecasting

👩‍💻 Author
Developed by Vinayak Jain 
📧 Contact: vinayakjainn11@gmail.com

