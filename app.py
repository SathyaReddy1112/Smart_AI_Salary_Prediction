import streamlit as st
import pandas as pd
import joblib
import os
import time # For simulating loading animation
import google.generativeai as genai # Import the Google Generative AI library

# --- Configuration ---
MODEL_PATH = 'salary_predictor_model.pkl'
DATA_PATH = 'salary_data_cleaned.csv' # Path to your original dataset for dropdown options

# --- Gemini API Configuration ---
# WARNING: For production apps, store your API key securely using Streamlit Secrets!
# Example: GOOGLE_API_KEY = st.secrets["gemini_api_key"]
GOOGLE_API_KEY = "AIzaSyCEvNMP8lofcyuy06fWEFV3Wh8Jg-y3ziQ" # Your provided API key
genai.configure(api_key=GOOGLE_API_KEY)

# --- Streamlit UI Configuration (MUST BE FIRST STREAMLIT COMMAND) ---
st.set_page_config(page_title="Smart AI Powered Employee Salary Prediction", layout="centered")

# --- Custom CSS for enhanced UI/UX ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
        color: #333;
    }

    /* Main background gradient */
    .stApp {
        background: linear-gradient(135deg, #f0f2f5 0%, #e0e5ec 100%);
    }

    /* Title styling */
    h1 {
        color: #2c3e50;
        text-align: center;
        font-weight: 700;
        margin-bottom: 20px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }

    h2, h3 {
        color: #34495e;
        font-weight: 600;
        margin-top: 30px;
        border-bottom: 2px solid #a8dadc;
        padding-bottom: 5px;
    }

    /* Card-like containers for sections */
    .stContainer {
        background-color: #ffffff;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        border: 1px solid #e0e0e0;
    }

    /* Input widgets styling */
    .stSlider > div > div > div {
        background-color: #4CAF50; /* Green slider track */
    }
    .stSlider > div > div > div > div {
        background-color: #2E8B57; /* Darker green slider handle */
    }

    div[data-baseweb="input"],
    div[data-baseweb="select"],
    div[data-baseweb="textarea"] {
        border-radius: 8px;
        border: 1px solid #ced4da;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.05);
        transition: all 0.3s ease-in-out;
    }
    div[data-baseweb="input"]:focus-within,
    div[data-baseweb="select"]:focus-within,
    div[data-baseweb="textarea"]:focus-within {
        border-color: #4CAF50;
        box-shadow: 0 0 0 0.2rem rgba(76, 175, 80, 0.25);
    }

    /* Checkbox styling */
    .stCheckbox > label {
        color: #555;
        font-weight: 400;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #4CAF50 0%, #2E8B57 100%);
        color: white;
        border-radius: 12px;
        border: none;
        padding: 12px 25px;
        font-size: 18px;
        font-weight: 600;
        box-shadow: 0 6px 12px rgba(0, 128, 0, 0.3);
        transition: all 0.3s ease;
        cursor: pointer;
        width: 100%;
        margin-top: 20px;
    }
    .stButton > button:hover {
        background: linear-gradient(45deg, #2E8B57 0%, #4CAF50 100%);
        box-shadow: 0 8px 16px rgba(0, 128, 0, 0.4);
        transform: translateY(-2px);
    }
    .stButton > button:active {
        transform: translateY(0);
        box-shadow: 0 4px 8px rgba(0, 128, 0, 0.2);
    }

    /* Success/Error messages */
    .stSuccess {
        background-color: #e6ffe6;
        color: #28a745;
        border-left: 5px solid #28a745;
        border-radius: 8px;
        padding: 15px;
        margin-top: 20px;
        font-size: 20px;
        font-weight: 600;
    }
    .stError {
        background-color: #ffe6e6;
        color: #dc3545;
        border-left: 5px solid #dc3545;
        border-radius: 8px;
        padding: 15px;
        margin-top: 20px;
    }
    .stWarning {
        background-color: #fff3cd;
        color: #ffc107;
        border-left: 5px solid #ffc107;
        border-radius: 8px;
        padding: 15px;
        margin-top: 20px;
    }

    /* Footer styling */
    .st-emotion-cache-1c7y2kl { /* Target the footer container */
        visibility: hidden;
    }
    footer {
        visibility: visible;
        text-align: center;
        padding: 10px;
        color: #777;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Load Model and Data ---
@st.cache_resource # Cache the model loading for efficiency
def load_model():
    """Loads the pre-trained machine learning pipeline."""
    with st.spinner("Loading AI model..."):
        time.sleep(1) # Simulate loading time
        if not os.path.exists(MODEL_PATH):
            st.error(f"Error: Model file '{MODEL_PATH}' not found. Please ensure it's in the same directory.")
            st.stop()
        try:
            pipeline = joblib.load(MODEL_PATH)
            return pipeline
        except Exception as e:
            st.error(f"Error loading the model: {e}")
            st.stop()

@st.cache_data # Cache the data loading for dropdowns
def load_data_for_options():
    """Loads the original dataset to populate dropdown options."""
    with st.spinner("Preparing data for options..."):
        time.sleep(0.5) # Simulate loading time
        if not os.path.exists(DATA_PATH):
            st.warning(f"Warning: Original data file '{DATA_PATH}' not found. Some dropdowns might be empty.")
            return pd.DataFrame()
        try:
            df_options = pd.read_csv(DATA_PATH)
            # Apply the same rare category handling as in the training pipeline
            if 'Job Title' in df_options.columns:
                job_counts = df_options['Job Title'].value_counts()
                rare_jobs = job_counts[job_counts < 10].index
                df_options['Job Title'] = df_options['Job Title'].replace(rare_jobs, 'Other')
            return df_options
        except Exception as e:
            st.error(f"Error loading data for options: {e}")
            return pd.DataFrame()

pipeline = load_model()
df_options = load_data_for_options()

# --- Gemini Insight Generation Function ---
def generate_salary_insight(input_data_dict, predicted_salary):
    """
    Generates a natural language explanation and insights for the predicted salary
    using the Gemini API.
    """
    model = genai.GenerativeModel('gemini-2.0-flash')

    # Format input data for the prompt
    job_details_str = "\n".join([f"- {key.replace('_', ' ').title()}: {value}"
                                 for key, value in input_data_dict.items() if key not in ['min_salary', 'max_salary']])
    skills_str = f"Python={input_data_dict['python_yn']}, R={input_data_dict['R_yn']}, Spark={input_data_dict['spark']}, AWS={input_data_dict['aws']}, Excel={input_data_dict['excel']}"

    prompt = f"""
    Given the following job details and predicted average salary, provide a concise explanation of factors that likely contributed to this salary, and offer one or two actionable insights or suggestions for career growth related to these details.

    Job Details:
    {job_details_str}
    - Min Expected Salary: ${input_data_dict['min_salary']}K
    - Max Expected Salary: ${input_data_dict['max_salary']}K
    - Skills: {skills_str}

    Predicted Average Salary: ${predicted_salary:,.2f}K

    Explanation and Insights:
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Could not generate insights: {e}"

# --- Streamlit UI ---
st.title("💰 Smart AI Powered Employee Salary Prediction")
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 1.1em; color: #555;'>Enter the job details below to get an estimated average salary.</p>", unsafe_allow_html=True)

# Input Fields Container
with st.container(border=True):
    st.header("💼 Job Details")

    col1, col2 = st.columns(2)

    with col1:
        rating = st.slider("Company Rating (Glassdoor)", 1.0, 5.0, 3.5, 0.1)
        age = st.number_input("Employee Age (Years)", min_value=18, max_value=70, value=30)
        min_salary = st.number_input("Minimum Expected Salary (K USD)", min_value=0, value=50)
        max_salary = st.number_input("Maximum Expected Salary (K USD)", min_value=0, value=90)

        st.markdown("---")
        st.subheader("Employment Type")
        hourly = st.checkbox("Hourly Wage Job?", False)
        employer_provided = st.checkbox("Employer Provided Salary?", False)
        same_state = st.checkbox("Job in Same State as Headquarters?", False)

    with col2:
        # Categorical features - populate from unique values in original data
        job_title_options = ['Other'] + sorted(df_options['Job Title'].unique().tolist()) if 'Job Title' in df_options.columns else ["Data Scientist", "Software Engineer", "Other"]
        job_title = st.selectbox("Job Title", job_title_options, index=job_title_options.index("Data Scientist") if "Data Scientist" in job_title_options else 0)

        location_options = sorted(df_options['Location'].unique().tolist()) if 'Location' in df_options.columns else ["New York, NY", "Chicago, IL", "Los Angeles, CA"]
        location = st.selectbox("Location", location_options, index=location_options.index("New York, NY") if "New York, NY" in location_options else 0)

        ownership_options = sorted(df_options['Type of ownership'].unique().tolist()) if 'Type of ownership' in df_options.columns else ["Company - Private", "Company - Public"]
        type_of_ownership = st.selectbox("Type of Ownership", ownership_options, index=ownership_options.index("Company - Private") if "Company - Private" in ownership_options else 0)

        industry_options = sorted(df_options['Industry'].unique().tolist()) if 'Industry' in df_options.columns else ["IT Services", "Biotech & Pharmaceuticals"]
        industry = st.selectbox("Industry", industry_options, index=industry_options.index("IT Services") if "IT Services" in industry_options else 0)

        sector_options = sorted(df_options['Sector'].unique().tolist()) if 'Sector' in df_options.columns else ["Information Technology", "Biotechnology"]
        sector = st.selectbox("Sector", sector_options, index=sector_options.index("Information Technology") if "Information Technology" in sector_options else 0)

        job_state_options = sorted(df_options['job_state'].unique().tolist()) if 'job_state' in df_options.columns else ["CA", "NY", "TX"]
        job_state = st.selectbox("Job State", job_state_options, index=job_state_options.index("CA") if "CA" in job_state_options else 0)

# Skills Section
with st.container(border=True):
    st.subheader("💡 Required Skills")
    col3, col4, col5, col6, col7 = st.columns(5)
    with col3:
        python_yn = st.checkbox("Python", True)
    with col4:
        r_yn = st.checkbox("R", False)
    with col5:
        spark = st.checkbox("Spark", False)
    with col6:
        aws = st.checkbox("AWS", False)
    with col7:
        excel = st.checkbox("Excel", True)

# --- Prediction Button ---
if st.button("Predict Salary"):
    # Create a dictionary from user inputs for both ML model and Gemini prompt
    input_data_dict = {
        'Rating': rating,
        'age': age,
        'min_salary': min_salary,
        'max_salary': max_salary,
        'hourly': int(hourly),
        'employer_provided': int(employer_provided),
        'same_state': int(same_state),
        'python_yn': int(python_yn),
        'R_yn': int(r_yn),
        'spark': int(spark),
        'aws': int(aws),
        'excel': int(excel),
        'Job Title': job_title,
        'Location': location,
        'Type of ownership': type_of_ownership,
        'Industry': industry,
        'Sector': sector,
        'job_state': job_state
    }

    # Convert dictionary to DataFrame for ML model prediction
    input_df_for_ml = pd.DataFrame([input_data_dict])

    try:
        with st.spinner("Calculating estimated salary..."):
            time.sleep(2) # Simulate prediction time
            predicted_salary = pipeline.predict(input_df_for_ml)[0]
        st.success(f"🎉 Estimated Average Salary: **${predicted_salary:,.2f}K**")
        st.balloons()

        # --- Generate and display insights using Gemini ---
        with st.spinner("Generating insights with AI..."):
            time.sleep(1) # Simulate Gemini call time
            gemini_insight = generate_salary_insight(input_data_dict, predicted_salary)
            st.subheader("AI-Powered Insights")
            st.info(gemini_insight) # Display insights in an info box

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.info("Please check your inputs and ensure the model is loaded correctly.")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #777;'>Developed by Smart AI Powered Employee Salary Prediction Team</p>", unsafe_allow_html=True)
