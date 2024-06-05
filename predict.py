import pandas as pd
import joblib
from datetime import datetime
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier

# Load the encoder and model
enc = joblib.load(r'encoder.joblib')
enc.handle_unknown = 'ignore'  # Add this line to set handle_unknown to 'ignore'
model = joblib.load(r'model.joblib')

def preprocess_file(df):
    df = df.rename(columns={"trans_date_trans_time":"transaction_time",
                            "cc_num":"credit_card_number",
                            "amt":"amount(usd)",
                            "trans_num":"transaction_id"}
                  )

    df['time'] = df['unix_time'].apply(datetime.utcfromtimestamp)
    df['hour_of_day'] = df.time.dt.hour
    features = ['transaction_id', 'hour_of_day', 'category', 'amount(usd)', 'merchant', 'job']
    df = df[features].set_index("transaction_id")
    df.loc[:, ['category','merchant','job']] = enc.transform(df[['category','merchant','job']])
    return df

def fraud_prediction(df):
    processed_df = preprocess_file(df.copy())
    predictions = model.predict(processed_df)
    probabilities = model.predict_proba(processed_df)[:, 1]
    return processed_df, predictions, probabilities
