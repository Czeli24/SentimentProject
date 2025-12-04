import pandas as pd
import streamlit as st

@st.cache_data
def load_local_data(file_path="data/Tweets.csv"):
    try:
        df = pd.read_csv(file_path, encoding="utf-8-sig")

        # detect sentiment column
        if "airline_sentiment" not in df.columns:
            for c in df.columns:
                if "sentiment" in c.lower():
                    df = df.rename(columns={c: "airline_sentiment"})
                    break

        if "airline_sentiment" not in df.columns:
            st.error("Dataset does NOT contain the column 'airline_sentiment'.")
            st.stop()

        return df[['text', 'airline_sentiment']]

    except FileNotFoundError:
        st.error(f"The file '{file_path}' was NOT found.")
        st.stop()
    except Exception as e:
        st.error(f"Error reading the local file: {e}")
        st.stop()
