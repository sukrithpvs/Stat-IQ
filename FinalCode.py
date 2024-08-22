import os
import re
import streamlit as st
import pandas as pd
import json
import time
import plotly.express as px
from groq import Groq
from io import StringIO

# Set page config at the very beginning
st.set_page_config(layout="wide")


client = Groq(api_key="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") 

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
def load_data(file):
    """Load the dataset from a file-like object."""
    if file.type == 'text/csv':
        df = pd.read_csv(file)
    elif file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
        df = pd.read_excel(file)
    elif file.type == 'application/json':
        df = pd.read_json(file)
    else:
        return None

    df = convert_dataframe_types(df)
    return df

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
    code_blocks = re.findall(r'python\n(.*?)\n', code, re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()
    else:
        st.error("No Python code found in the response.")
        return ""

def execute_code(code, df):
    """Execute the dynamically generated code."""
    try:
        cleaned_code = clean_code(code)
        if not cleaned_code:
            st.error("No valid Python code to execute.")
            return None

        cleaned_code = cleaned_code.replace('your_file_path_here', 'data')

        # Create an in-memory string buffer for the DataFrame
        buffer = StringIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)

        # Execute the code in the current environment
        namespace = {'data': buffer}
        exec(cleaned_code, globals(), namespace)

        # Return the DataFrame if it was created
        if 'cleaned_data' in namespace:
            cleaned_df = pd.read_csv(namespace['cleaned_data'])
            return cleaned_df
        else:
            return None
    except Exception as e:
        st.error(f"Execution error: {e}")
        time.sleep(2)

def generate_cleaning_code(data_description):
    """Generate Python code for data cleaning and preprocessing."""
    prompt = f"""
    Based on the following data description, generate optimized Python code for data cleaning, Exploratory Data Analysis (EDA), and preprocessing. The code should be dynamic and scalable to handle the entire dataset. Prioritize key preprocessing steps and essential EDA techniques that will effectively support most data visualizations. Minimize unnecessary operations to ensure efficiency. Adjust the code to use an in-memory DataFrame.
    Use st.cache_data for Streamlit and also show initial data shape and cleaned data shape too.
    Data Description:
    {data_description}
    """
    code = get_response(prompt)
    return code

def generate_visualization_code(data_description):
    """Generate Python code for data visualization."""
    prompt = f"""
    Based on the cleaned dataset, generate a Streamlit Python code to create a 'dataset name Dashboard' with 7 essential graphs and plots that fully summarize the dataset. The code should include various graph types like pie charts, bar graphs, histograms, and other relevant plots to provide a comprehensive overview. The layout should be inspired by a Power BI analytical dashboard, ensure a wide and aesthetically pleasing horizontal display of the graphs in Streamlit columns.
    make sure st.set_page_config(layout='wide') is at the beginning of the streamlit generated code. don’t include this "st.set_page_config(layout='wide')" anywhere else in the generate code except the first line.
    Ensure that the generated dashboard is easy to interpret, even for someone with no knowledge of EDA or data analysis, effectively conveying the key insights from the dataset. Use only Plotly Express for the visualizations, and make sure the code is error-free. Analyze the clean dataset thoroughly before plotting.
    """
    code = get_response(prompt)
    cleaned_code = clean_code(code)
    return cleaned_code

def run_visualizations(visualization_code):
    """Run the visualization code."""
    try:
        namespace = {}
        exec(visualization_code, globals(), namespace)

        # Display the visualizations if any
        if 'figures' in namespace:
            for fig in namespace['figures']:
                st.plotly_chart(fig)
        else:
            st.error("No visualizations found in the generated code.")
    except Exception as e:
        st.error(f"Execution error: {e}")

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
    # Load the data
    data = load_data(uploaded_file)

    if data is not None:
        # Create tabs for different sections of the dashboard
        tab1, tab2, tab3, tab4 = st.tabs(["Data Overview", "Data Analysis and Visualization", "Chatbot", "Business Recommendations"])

        with tab1:
            st.header("Data Overview")
            st.write(f"Number of rows: {data.shape[0]}")
            st.write(f"Number of columns: {data.shape[1]}")
            st.write("Data Sample:")
            st.write(data.head())
            st.subheader("Data Types")
            st.write(data.dtypes)
            st.subheader("Summary Statistics")
            st.write(data.describe())

        with tab2:
            st.header("Data Analysis and Visualization")
            if st.button("Generate Cleaning and EDA Code"):
                data_description = data.describe(include='all').to_json()
                cleaning_code = generate_cleaning_code(data_description)
                if cleaning_code:
                    st.write("Generated Data Cleaning and EDA Code:")
                    st.code(cleaning_code, language='python')

                    # Execute the cleaning code
                    cleaned_data = execute_code(cleaning_code, data)

                    if cleaned_data is not None:
                        # Generate visualization code
                        visualization_code = generate_visualization_code(data_description)

                        if visualization_code:
                            # Run the visualization code
                            run_visualizations(visualization_code)

        with tab3:
            st.header("Stat-IQ GPT")
            st.write("Chat with your data and get personalized plots and graphs.")
            question = st.text_input("Ask a question or request a specific plot:")
            if st.button("Submit"):
                if question:
                    with st.spinner():
                        response = get_response(question)
                    st.write("Response",response)
                                        # Check if response contains code
                    if "python" in response:
                        cleaned_code = clean_code(response)
                        if cleaned_code:
                            # Directly use the generated code for visualization
                            run_visualizations(cleaned_code)
            else:
                st.error("Please enter a question.")

    with tab4:
        st.header("Business Recommendations")
        if st.button("Generate Recommendations"):
            data_description = data.describe(include='all').to_json()
            recommendations = generate_business_recommendations(data_description)
            st.write("Business Recommendations:")
            for i, recommendation in enumerate(recommendations):
                st.write(f"{recommendation}")

else:
    st.error("Failed to load data from the uploaded file.")

