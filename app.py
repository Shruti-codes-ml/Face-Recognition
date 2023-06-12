import streamlit as st
import subprocess
import re

# Run the notebook as a subprocess and capture the output
def run_notebook():
    subprocess.run(["jupyter", "nbconvert", "--to", "script", "Face Recognition.ipynb"])
    process = subprocess.Popen(["python", "face_recognition_python.py"], stdout=subprocess.PIPE)
    output, _ = process.communicate()
    return output.decode("utf-8")

# Format the output
def format_output(output):
    formatted_output = re.sub(r"\r\n", "<br>", output)
    return formatted_output

# Run the Streamlit app
def main():
    st.title("Face Recognition")
    # Add your Streamlit app code here

    # Call the function to run the notebook and capture the output
    output = run_notebook()

    # Format the output
    formatted_output = format_output(output)

    # Display the output on the Streamlit app
    st.markdown(f"Recognized person's name: {formatted_output}", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
