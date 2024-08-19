import os
import re
import streamlit as st
import pandas as pd
import json
import time
from groq import Groq
from io import StringIO
import matplotlib.pyplot as plt

# Initialize Groq client with your API key
client = Groq(api_key="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")  # Replace with your actual API key

# Initialize conversation history for chat
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = [
        {
            "role": "system",
            "content": "You are a data analyst, data engineer, and business analyst."
        }
    ]

def get_response(user_query):
    """Get a response from Groq's model with retry logic."""
    st.session_state.conversation_history.append({
        "role": "user",
        "content": user_query
    })

    # Limit conversation history to the last 10 messages
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

def execute_code(code, data_file_path):
    """Execute the dynamically generated code."""
    try:
        cleaned_code = clean_code(code)

        if not cleaned_code:
            st.error("No valid Python code to execute.")
            return None

        # Adjust file path in code
        cleaned_code = cleaned_code.replace('your_file_path_here', data_file_path)

        # Print the cleaned code for debugging
        st.write("Generated Code:")
        st.code(cleaned_code, language='python')

        # Run the code using exec (in-memory execution)
        local_vars = {}
        exec(cleaned_code, {"__name__": "__main__"}, local_vars)

        cleaned_data = local_vars.get('cleaned_data')

        if cleaned_data is not None:
            return cleaned_data
        else:
            st.error("No cleaned data generated.")
            return None

    except Exception as e:
        st.error(f"Execution error: {e}")
        time.sleep(2)  # Wait before retrying

def generate_code(data_description, file_path):
    """Generate Python code for data cleaning and preprocessing."""
    prompt = f"""
    Based on the following data description, generate Python code for cleaning, EDA, and preprocessing the data.
    NOTE: THIS DATA IS JUST A SAMPLE OF A LARGER DATASET. MAKE SURE THE CODE IS DYNAMIC TO HANDLE THE ENTIRE DATASET.
    Convert the entire file as dataframe, in the python code. the path of the dataset is "{file_path}", change code accordingly for reading dataset based on file extension.

    Data Description:
    {data_description}
    """
    code = get_response(prompt)
    return code

def generate_insight_for_graph(graph_type, column_name, data):
    """Generate insights for each type of graph based on the data."""
    if graph_type == "Histogram":
        mean = data[column_name].mean()
        median = data[column_name].median()
        std_dev = data[column_name].std()
        return f"The histogram of '{column_name}' shows the distribution of the data. The mean is {mean:.2f}, the median is {median:.2f}, and the standard deviation is {std_dev:.2f}. This indicates the spread and central tendency of the data."

    elif graph_type == "Bar Chart":
        value_counts = data[column_name].value_counts()
        most_common = value_counts.idxmax()
        most_common_count = value_counts.max()
        return f"The bar chart of '{column_name}' shows the frequency of different categories. The most common category is '{most_common}' with {most_common_count} occurrences."

    elif graph_type == "Line Chart":
        trend = data[column_name].rolling(window=5).mean()
        return f"The line chart of '{column_name}' shows the trend over time. The rolling average indicates the overall trend and smoothing of fluctuations."

    elif graph_type == "Pie Chart":
        value_counts = data[column_name].value_counts()
        return f"The pie chart of '{column_name}' shows the proportion of different categories. The chart provides a visual representation of category proportions."

    else:
        return "No specific insights available for this graph type."

def generate_business_recommendations(data_description):
    """Generate business recommendations based on the data description."""
    prompt = f"""
    Based on the following data description, provide 10 business recommendations and analytics.

    Data Description:
    {data_description}
    """
    recommendations = get_response(prompt)
    return recommendations

# Streamlit UI
st.title("Stat IQ")
st.write("Upload your dataset and let the system handle the analysis and cleaning.")

uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "json"])

if uploaded_file is not None:
    # Use Streamlit's temporary directory
    temp_folder = st.experimental_get_tmp_dir()
    temp_file_path = os.path.join(temp_folder, uploaded_file.name)

    # Save the uploaded file
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(uploaded_file.getvalue())

    # Load the data into a DataFrame
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(temp_file_path)
    elif uploaded_file.name.endswith('.xlsx'):
        data = pd.read_excel(temp_file_path)
    elif uploaded_file.name.endswith('.json'):
        data = pd.read_json(temp_file_path)

    # Display top 10 rows of the pre-cleaned dataset
    st.write("Top 10 rows of the pre-cleaned dataset:")
    st.write(data.head(10))

    # Prepare data description
    data_description = json.dumps(data.describe(include='all').to_dict(), indent=2)

    # Generate code for data cleaning and preprocessing
    code = generate_code(data_description, temp_file_path)

    if code:
        # Display the generated code
        st.write("Generated Code for Data Cleaning and Preprocessing:")
        st.code(code, language='python')

        # Execute the generated code
        cleaned_data = execute_code(code, temp_file_path)
        if cleaned_data is not None:
            # Display top 10 rows of the cleaned dataset
            st.write("Top 10 rows of the cleaned dataset:")
            st.write(cleaned_data.head(10))

            # Provide a download button for the cleaned data
            csv_buffer = StringIO()
            cleaned_data.to_csv(csv_buffer, index=False)
            st.download_button(
                label="Download cleaned data as CSV",
                data=csv_buffer.getvalue(),
                file_name="cleaned_data.csv",
                mime="text/csv",
            )

            # Plot and analyze graphs
            st.write("Generating graphs and insights...")

            # Generate graphs and insights
            num_plots = 0
            for column in cleaned_data.columns:
                if cleaned_data[column].dtype in ['int64', 'float64']:
                    st.write(f"### Histogram of {column}")
                    st.write(generate_insight_for_graph("Histogram", column, cleaned_data))
                    fig, ax = plt.subplots()
                    cleaned_data[column].hist(ax=ax)
                    st.pyplot(fig)
                    num_plots += 1

                if cleaned_data[column].dtype == 'object':
                    st.write(f"### Bar Chart of {column}")
                    st.write(generate_insight_for_graph("Bar Chart", column, cleaned_data))
                    fig, ax = plt.subplots()
                    cleaned_data[column].value_counts().plot.bar(ax=ax)
                    st.pyplot(fig)
                    num_plots += 1

                if cleaned_data[column].dtype in ['int64', 'float64'] and len(cleaned_data) > 1:
                    st.write(f"### Line Chart of {column}")
                    st.write(generate_insight_for_graph("Line Chart", column, cleaned_data))
                    fig, ax = plt.subplots()
                    cleaned_data[column].plot(ax=ax)
                    ax.plot(cleaned_data[column].rolling(window=5).mean(), color='red')
                    st.pyplot(fig)
                    num_plots += 1

                if cleaned_data[column].dtype == 'object':
                    st.write(f"### Pie Chart of {column}")

                    # Create and display pie chart using matplotlib
                    fig, ax = plt.subplots()
                    cleaned_data[column].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
                    st.pyplot(fig)

                    st.write(generate_insight_for_graph("Pie Chart", column, cleaned_data))
                    num_plots += 1

            if num_plots == 0:
                st.write("No suitable columns found for generating graphs.")

            # Generate business recommendations based on the dataset description
            st.write("Generating business recommendations...")
            recommendations = generate_business_recommendations(data_description)
            st.write("### Business Recommendations and Analytics:")
            st.write(recommendations)

        else:
            st.error("Error in processing the cleaned data.")
