# ✈️ Twitter US Airlines Sentiment Analysis

**Author:** Czeli Zoltán-Dragoș  
**Project:** Automatic Sentiment Analysis on Social Media  
**Tech Stack:** Python, Streamlit, Scikit-learn, Pandas

---

## 📖 Project Description

This project implements a complete **Artificial Intelligence pipeline** to automatically analyze the sentiment of tweets related to US Airlines. 

The system detects whether a tweet is **Positive**, **Negative**, or **Neutral** by processing natural language text. It utilizes **TF-IDF (Term Frequency-Inverse Document Frequency)** for vectorization and a **Logistic Regression** model for classification.

The project is wrapped in an interactive **Streamlit** web application that allows users to view training metrics and test the model with custom text in real-time.

### 🌟 Key Features
* **Modular Architecture:** Code is organized into separate modules (loader, preprocessing, vectorizer, model, evaluation).
* **Data Cleaning:** Automatic removal of user mentions (`@User`), URLs, special characters, and conversion to lowercase.
* **Machine Learning:** Logistic Regression classifier trained on TF-IDF features (max 5000 features).
* **Visualizations:** Confusion Matrix and Sentiment Distribution charts.
* **Interactive UI:** A user-friendly interface to test new tweets instantly.

---

## 📂 Project Structure

The project follows a modular structure for maintainability:

```text
SentimentProject/
│
├── app.py             # Main Streamlit Application (Entry Point)
├── requirements.txt         # List of Python dependencies
├── README.md                # Project Documentation
│
├── data/
│   └── Tweets.csv           # Local dataset (Source: Kaggle)
│
└── modules/
    ├── data_loader.py       # Handles CSV loading
    ├── preprocessing.py     # Cleans raw text data
    ├── vectorizer.py        # Converts text to TF-IDF numeric vectors
    ├── model.py             # Trains and manages the Logistic Regression model
```

## 🚀 How to Run

Follow these steps to set up the project on your local machine.

### 1. Clone the Repository.
```bash
  git clone <your-repo-url>
  cd SentimentProject
```

### 2. Create a Virtual Environment.

It is recommended to use a virtual environment to manage dependencies.

#### Windows:
```bash
  python -m venv venv
  venv\Scripts\activate
```

#### macOS / Linux:
```bash
  python3 -m venv venv
  source venv/bin/activate
```

### 3. Install Dependencies.

Install the required libraries listed in requirements.txt:

``` bash 
  pip install -r requirements.txt
```

### 4. Run the Application.

Launch the Streamlit interface:

``` bash 
  streamlit run app.py   
```

## 🛠️ Dependencies

streamlit - Web application framework

pandas - Data manipulation

numpy - Numerical operations

scikit-learn - Machine Learning algorithms and tools

matplotlib & seaborn - Data visualization

## 📝 Dataset

Name: Twitter US Airline Sentiment

Source: Kaggle

Content: The dataset contains tweets classified as positive, negative, or neutral regarding six US airlines.

## © 2025 Czeli Zoltán-Dragoș | Anul III , Grupa: 1631A