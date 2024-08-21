import os
import re
import streamlit as st
import pandas as pd
import json
import time
from groq import Groq
import subprocess
import tempfile
import plotly.express as px

# Set page config at the very beginning
st.set_page_config(layout="wide")

@st.cache_resource
def initialize_groq_client(api_key):
    """Initialize Groq client with API key."""
    return Groq(api_key=api_key)

client = initialize_groq_client(api_key="your_api_key_here")  # Replace with your actual API key

# Initialize conversation history for chat
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = [
        {
            "role": "system",
            "content": "You are a data analyst, data engineer, and business analyst."
        }
    ]

@st.cache_resource
def convert_dataframe_types(df):
    """Ensure all DataFrame columns have consistent and appropriate types."""
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast='integer')
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype(str)
        # Add more type conversions as needed
    return df

@st.cache_resource
def load_data(file_path):
    """Load the dataset efficiently with caching."""
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        else:
            return None
        
        df = convert_dataframe_types(df)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def get_response(user_query):
    """Get a response from Groq's model with retry logic and improved error handling."""
    st.session_state.conversation_history.append({
        "role": "user",
        "content": user_query
    })

    conversation_history = st.session_state.conversation_history[-10:]

    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                messages=conversation_history,
                model="Llama-3.1-70b-Versatile",
                temperature=0.5,
                max_tokens=1024,
                top_p=1,
                stop=None,
                stream=False,
            )
            assistant_response = response.choices[0].message.content
            st.session_state.conversation_history.append({
                "role": "assistant",
                "content": assistant_response
            })
            return assistant_response
        except Exception as e:
            if "rate limit" in str(e).lower():
                wait_time = 2 ** attempt
                st.warning(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            elif "invalid api key" in str(e).lower():
                st.error("Invalid API key. Please check your API key and try again.")
                return None
            else:
                st.error(f"Error: {e}")
                return None

def clean_code(code):
    """Extract only Python code from the response."""
    code_blocks = re.findall(r'```python\n(.*?)\n```', code, re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()
    else:
        st.error("No Python code found in the response.")
        return ""

def execute_code(code, data_file_path):
    """Execute the dynamically generated code."""
    try:
        cleaned_code = clean_code(code)
        if not cleaned_code:
            st.error("No valid Python code to execute.")
            return None

        cleaned_code = cleaned_code.replace('your_file_path_here', data_file_path)

        # Save the cleaned code to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as temp_script:
            temp_script.write(cleaned_code.encode())
            temp_script_path = temp_script.name

        # Print the cleaned code for debugging
        st.write("Generated Code:")
        st.code(cleaned_code, language='python')

        # Run the code
        result = subprocess.run(
            ["python", temp_script_path],
            capture_output=True,
            text=True
        )

        # Print the result and errors
        st.write("Code Execution Output:")
        st.write(result.stdout)

        if result.returncode != 0:
            error_message = f"Error executing the code: {result.stderr}"
            st.error(error_message)
            return error_message
        else:
            return temp_script_path
    except Exception as e:
        st.error(f"Execution error: {e}")
        time.sleep(2)

def generate_cleaning_code(data_description, file_path):
    """Generate Python code for data cleaning and preprocessing."""
    prompt = f"""
    Based on the following data description, generate optimized Python code for data cleaning, Exploratory Data Analysis (EDA), and preprocessing. The code should be dynamic and scalable to handle the entire dataset. Prioritize key preprocessing steps and essential EDA techniques that will effectively support most data visualizations. Minimize unnecessary operations to ensure efficiency. Convert the dataset to a DataFrame, and adjust the code to correctly read the file based on its extension at the following path: {file_path}.
    Use st.cache_data for Streamlit and also show initial data shape and cleaned data shape too.
    Data Description:
    {data_description}
    """
    code = get_response(prompt)
    return code

def generate_visualization_code(data_description, file_path):
    """Generate Python code for data visualization."""
    prompt = f"""
    Based on the dataset, generate a Streamlit Python code to create a 'dataset name Dashboard' with 7 essential graphs and plots that fully summarize the dataset. The code should include various graph types like pie charts, bar graphs, histograms, and other relevant plots to provide a comprehensive overview. The layout should be inspired by a Power BI analytical dashboard, ensure a wide and aesthetically pleasing horizontal display of the graphs in Streamlit columns.
    make sure st.set_page_config(layout='wide') is at the begining of the streamlit generated code . dont include this "st.set_page_config(layout='wide')" anywhere else in the generate code except the first line 
    Ensure that the generated dashboard is easy to interpret, even for someone with no knowledge of EDA or data analysis, effectively conveying the key insights from the dataset. Use only Plotly Express for the visualizations, and make sure the code is error-free. Analyze the dataset thoroughly before plotting.
    Use 'data' as the name of the DataFrame variable.
    """
    code = get_response(prompt)
    cleaned_code = clean_code(code)
    
    if cleaned_code:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as temp_script:
            temp_script.write(f"import pandas as pd\nimport plotly.express as px\nimport streamlit as st\n\n".encode())
            temp_script.write(f"data = pd.read_csv(r'{file_path}')\n\n".encode())
            temp_script.write(cleaned_code.encode())
            visualization_code_path = temp_script.name
        st.success("Visualization code generated and saved.")
        return visualization_code_path
    else:
        st.error("Failed to generate visualization code.")
        return None

def run_visualizations(script_path):
    """Run the Streamlit visualization script."""
    try:
        st.write("Visualization script generated. Use the link below to view it.")
        st.write(f"Run the script using: `streamlit run {script_path}`")
    except Exception as e:
        st.error(f"Unexpected error: {e}")

def generate_business_recommendations(data_description):
    """Generate business recommendations based on the dataset."""
    prompt = f"""
    Based on the following data description, provide 10 business recommendations. These recommendations should be actionable and based on the insights derived from the dataset. Focus on key areas where improvements can be made, trends that can be leveraged, and strategies to optimize business operations.
    
    Data Description:
    {data_description}
    """
    response = get_response(prompt)
    recommendations = response.split('\n')
    return recommendations

# Streamlit UI
st.title("Stat IQ Dashboard")
st.write("Upload your dataset and let our model handle the analysis and visualization.")

uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "json"])

if uploaded_file is not None:
    # Save the uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
        file_path = temp_file.name
        temp_file.write(uploaded_file.getbuffer())

    # Load the data
    data = load_data(file_path)

    if data is not None:
        # Create tabs for different sections of the dashboard
        tab1, tab2, tab3, tab4 = st.tabs(["Data Overview", "Data Analysis and Visualization", "Chatbot", "Business Recommendations"])

        with tab1:
            st.header("Data Overview")
            st.write(f"Number of rows: {data.shape[0]}")
            st.write(f"Number of columns: {data.shape[1]}")
            st.write("Data Sample:")
            st.write(data.head())

        with tab2:
            st.header("Data Analysis and Visualization")
            if st.button("Generate Cleaning and EDA Code"):
                data_description = data.describe(include='all').to_json()
                cleaning_code = generate_cleaning_code(data_description, file_path)
                if cleaning_code:
                    st.write("Generated Data Cleaning and EDA Code:")
                    st.code(cleaning_code, language='python')

                    # Execute the cleaning code
                    cleaned_file_path = execute_code(cleaning_code, file_path)

                    if cleaned_file_path:
                        # Load the cleaned dataset
                        cleaned_data = pd.read_csv(cleaned_file_path)
                        
                        # Generate visualization code
                        visualization_code_path = generate_visualization_code(data_description, cleaned_file_path)
                        
                        if visualization_code_path:
                            # Display the path to run the visualizations
                            run_visualizations(visualization_code_path)

        with tab3:
            st.header("Stat-IQ GPT")
            st.write("Chat with your data and get personalized plots and graphs.")
            question = st.text_input("Ask a question or request a specific plot:")
            if st.button("Submit"):
                if question:
                    with st.spinner('Generating response...'):
                        response = get_response(question)
                    st.write("Response:", response)

                # Check if response contains code
                if "```python" in response:
                    cleaned_code = clean_code(response)
                    if cleaned_code:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as temp_file:
                            temp_file.write(f"import pandas as pd\nimport plotly.express as px\nimport streamlit as st\n\n".encode())
                            temp_file.write(f"data = pd.read_csv(r'{file_path}')\n\n".encode())
                            temp_file.write(cleaned_code.encode())
                            custom_script_path = temp_file.name
                        # Display the path to run the custom visualizations
                        run_visualizations(custom_script_path)

        with tab4:
            st.header("Business Recommendations")
            if st.button("Generate Recommendations"):
                data_description = data.describe(include='all').to_json()
                recommendations = generate_business_recommendations(data_description)
                st.write("Business Recommendations:")
                for i, recommendation in enumerate(recommendations):
                    st.write(f"{i + 1}. {recommendation}")

    else:
        st.error("Failed to load data from the uploaded file.")

if st.button("Download Cleaned Dataset"):
    if 'data' in locals() and data is not None:
        csv = data.to_csv(index=False)
        st.download_button(
            label="Download cleaned data as CSV",
            data=csv,
            file_name="cleaned_data.csv",
            mime="text/csv",
        )
    else:
        st.error("No cleaned dataset available for download.")
