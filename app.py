import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from modules.data_loader import load_local_data
from modules.preprocessing import clean_tweet
from modules.vectorizer import get_tfidf_vectors
from modules.model import train_logistic_model
from modules.evaluation import calculate_metrics, get_classification_report_df, get_confusion_matrix_df

st.set_page_config(page_title="Twitter Sentiment NLP Project", layout="wide")

# Load & preprocess
df = load_local_data()
df["clean_text"] = df["text"].apply(clean_tweet)

# TF-IDF
X, vectorizer = get_tfidf_vectors(df["clean_text"])
y = df["airline_sentiment"]

# Train/test split & model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = train_logistic_model(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
metrics = calculate_metrics(y_test, y_pred)
report_df = get_classification_report_df(y_test, y_pred)
cm_df = get_confusion_matrix_df(y_test, y_pred)

# Streamlit UI
st.title("✈️ Twitter US Airlines Sentiment Analysis - Czeli Zoltán-Dragoș")

tab1, tab2, tab3 = st.tabs(["📊 Analysis & Metrics", "🤖 Test Model", "ℹ️ Requirements"])

# TAB 1: Visualizations & Metrics
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sentiment distribution in dataset")
        fig, ax = plt.subplots()
        sns.countplot(x='airline_sentiment', data=df, palette="viridis", ax=ax, hue=None)
        ax.set_xlabel("Sentiment")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    with col2:
        st.subheader("Model Performance")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
        m2.metric("F1-Score Weighted", f"{metrics['f1']*100:.2f}%")
        m3.metric("Recall Weighted", f"{metrics['recall']*100:.2f}%")
        m4.metric("Precision Weighted", f"{metrics['precision']*100:.2f}%")

        # Classification report HTML
        report_df_html = report_df.round(2).reset_index()
        report_df_html.rename(columns={"index": "Class"}, inplace=True)
        html_report = report_df_html.to_html(index=False, classes="table-class")
        st.markdown(
            f"""
            <h5>Classification Report</h5>
            <div style="overflow-x:auto;">
                {html_report}
            </div>
            <style>
                .table-class {{
                    width: 100%;
                    border-collapse: collapse;
                    background-color: white;
                    color: black;
                }}
                .table-class th, .table-class td {{
                    border: 1px solid black;
                    padding: 6px;
                    text-align: center;
                }}
                .table-class td:first-child {{
                    text-align: left;
                }}
            </style>
            """, unsafe_allow_html=True
        )

        # Confusion matrix
        cm_df_html = cm_df.reset_index()
        cm_df_html.rename(columns={"index": "Actual/Pred"}, inplace=True)
        html_cm = cm_df_html.to_html(index=False, classes="table-cm")
        st.markdown(
            f"""
            <h5>Confusion Matrix</h5>
            <div style="overflow-x:auto;">
                {html_cm}
            </div>
            <style>
                .table-cm {{
                    width: 100%;
                    border-collapse: collapse;
                    background-color: white;
                    color: black;
                }}
                .table-cm th, .table-cm td {{
                    border: 1px solid black;
                    padding: 6px;
                    text-align: center;
                }}
                .table-cm td:first-child {{
                    text-align: left;
                }}
            </style>
            """, unsafe_allow_html=True
        )

# TAB 2: Test new text
with tab2:
    st.subheader("Test the model with a new tweet")
    text_input = st.text_area("Write a tweet:", "My flight was delayed but the staff was helpful.")

    if st.button("Analyze Sentiment"):
        clean = clean_tweet(text_input)
        vect = vectorizer.transform([clean])
        pred = model.predict(vect)[0]

        st.divider()
        if pred == "negative":
            st.error("Sentiment: NEGATIVE 😡")
        elif pred == "positive":
            st.success("Sentiment: POSITIVE 😊")
        else:
            st.warning("Sentiment: NEUTRAL 😐")

        st.info(f"Internally cleaned text: '{clean}'")

# TAB 3: Project Requirements
with tab3:
    st.markdown("""
    ### Project Steps Completed
    1. Data collection and preprocessing (local CSV)
    2. TF-IDF numeric representation
    3. Logistic Regression model training
    4. Evaluation metrics: Accuracy, F1, Recall, Precision
    5. Sentiment distribution visualization
    6. Confusion matrix
    7. Testing with new text
    8. Functional Streamlit UI for user input
    """)
