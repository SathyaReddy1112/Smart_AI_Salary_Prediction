
# 🚀 Smart AI Powered Employee Salary Prediction

A web application built with Streamlit that predicts employee salaries based on job parameters and provides intelligent, AI-generated career insights using Google's Gemini API.

---

## 📌 Features

- **🔮 Salary Prediction**: Predict salaries based on input features like rating, location, skills, and more.
- **🎛️ Interactive UI**: Clean, user-friendly Streamlit interface.
- **🤖 AI-Powered Insights (Gemini)**: Understand *why* the predicted salary is what it is, with natural language insights.
- **📈 Robust ML Pipeline**: Includes EDA, data preprocessing, model comparison (Linear, Decision Tree, RF, XGBoost), and performance reports.
- **🎨 Modern Styling**: Polished interface with custom CSS effects like gradients and animated card UI.

---

## 🗂️ Project Structure

```
employee-salary-prediction/
│
├── .ipynb_checkpoints/             # Auto-saved Jupyter notebook states
├── app.py                          # Main Streamlit app with prediction + Gemini
├── salary_data_cleaned.csv         # Preprocessed dataset
├── salary_predictor_model.pkl      # Trained ML model with preprocessing pipeline
├── salary_model_training.ipynb     # Jupyter notebook for EDA + training
├── model_evaluation_report.csv     # Model comparison metrics
├── requirements.txt                # All project dependencies
├── README.md                       # You are here
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository or Create Manually

```bash
git clone <your-repo-url>
cd employee-salary-prediction
```

> Or manually create the folder and add all files inside it.

### 2. Create and Activate Virtual Environment

**Windows**
```bash
python -m venv venv
.env\Scriptsctivate
```

**macOS/Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> Or install manually:
```bash
pip install streamlit pandas scikit-learn joblib xgboost google-generativeai
```

---

## 🔑 Gemini API Key Setup

1. Create a `.streamlit/` directory:
```bash
mkdir .streamlit
```

2. Inside it, create `secrets.toml`:
```toml
# .streamlit/secrets.toml
gemini_api_key = "YOUR_GEMINI_API_KEY_HERE"
```

3. In `app.py`, use:
```python
GOOGLE_API_KEY = st.secrets["gemini_api_key"]
```

---

## 🚀 Usage

1. Ensure `salary_predictor_model.pkl` and `salary_data_cleaned.csv` are present.
2. Run the app:
```bash
streamlit run app.py
```

3. Open browser at: [http://localhost:8501](http://localhost:8501)

---

## 💡 Gemini API Integration

The Gemini 2.0 model is used to:
- Explain *why* a predicted salary is high or low.
- Recommend *career actions* or *skill enhancements*.
- Help users *understand job market trends* using natural language.

---

## 🤝 Contributing

- Fork this repo.
- Create a new branch: `git checkout -b feature/my-feature`
- Commit your changes: `git commit -m "Add feature"`
- Push and open a Pull Request.

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` file for details.
