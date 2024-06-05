import streamlit as st
import pandas as pd
from predict import preprocess_file, fraud_prediction

# Set page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
)

def main():
    # Display the banner
    st.image("banner.jpg", use_column_width=True)

    st.title("Credit Card Fraud Detection")

    # Sidebar for about and instructions
    st.sidebar.title("About")
    st.sidebar.info("""
        This application allows you to detect potential credit card fraud by uploading a CSV file with transaction data. 
        The app preprocesses the data, applies a machine learning model to predict fraud, and provides the results for download.
    """)

    st.sidebar.title("Instructions")
    st.sidebar.info("""
        1. Upload a CSV file containing the transaction data using the file uploader below.
        2. The data will be displayed on the main page.
        3. Click the "Predict" button to process the data and make predictions on whether each transaction is fraudulent.
        4. You can download the resulting data with predictions and probabilities as a CSV file.
    """)

    st.sidebar.title("Interpreting the Results")
    st.sidebar.info("""
        - A **Probability** close to **1** means the model is confident the transaction is **fraudulent**.
        - A **Probability** close to **0** means the model is confident the transaction is **not fraudulent**.
        - A **Prediction** of **fraud** indicates a potentially fraudulent transaction.
        - A **Prediction** of **not a fraud** indicates a non-fraudulent transaction.
    """)

    # Main page content
    st.subheader("Upload CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        
        # Dropdown to select number of rows to display
        num_rows = st.selectbox("Select number of rows to display", [10, 20, 30, 40, 50])
        st.write(f"Data Sample (First {num_rows} Rows):")
        st.dataframe(data.head(num_rows))

        if st.button("Predict"):
            # Perform predictions
            with st.spinner("Processing..."):
                processed_data, predictions, probabilities = fraud_prediction(data)

                # Add predictions and probabilities to the original data
            data["Prediction"] = predictions
            data["Prediction"] = data["Prediction"].apply(lambda x: "fraud" if x == 1 else "not a fraud")
            data["Probability"] = probabilities

            st.write(f"Data Sample with Predictions and Probabilities (First {num_rows} Rows):")
            st.dataframe(data.head(num_rows))

        if "Prediction" in data.columns and "Probability" in data.columns:
                # Create CSV for download
            csv = data.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name='fraud_predictions.csv',
                mime='text/csv'
                )

if __name__ == "__main__":
    main()
