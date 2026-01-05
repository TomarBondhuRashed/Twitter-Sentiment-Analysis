# 🐦 Twitter Sentiment Analysis (NLP)

### 🚀 Day 14 of 30: Machine Learning Projects Challenge

**Goal:** Build a Natural Language Processing (NLP) model that can analyze a tweet and determine if the emotion is **Positive** 😃 or **Negative** 😞.

**The Dataset:** [Sentiment140 Dataset (Kaggle)](https://www.kaggle.com/datasets/kazanova/sentiment140), containing 1.6 million tweets.

---

## 🧠 Project Overview
Social media data is unstructured and messy. This project uses **Natural Language Processing (NLP)** techniques to clean raw text and train a Machine Learning model to detect sentiment.

* **Input:** Raw Tweet (e.g., *"I hate waiting in the rain!"*)
* **Output:** Sentiment Class (Positive / Negative)
* **Model Accuracy:** ~78% on test data.

---

## 🛠️ Tech Stack
* **Python** (Core Logic)
* **NLTK** (Natural Language Toolkit) - Used for Stopword removal and Stemming.
* **Scikit-Learn** (Logistic Regression, TF-IDF Vectorizer).
* **Pandas & NumPy** (Data Manipulation).
* **Joblib** (Model Persistence).
* **Matplotlib & WordCloud** (EDA & Visualization).

---

## ⚙️ How It Works (The Pipeline)

1.  **Preprocessing:**
    * **Regex Cleaning:** Removed special characters (`@`, `#`, `http://`).
    * **Tokenization:** Split sentences into words.
    * **Stopword Removal:** Removed common filler words ("the", "is", "at").
    * **Stemming:** Reduced words to their root form (e.g., "running" $\rightarrow$ "run") using `PorterStemmer`.
2.  **Vectorization:**
    * Converted text into numerical vectors using **TF-IDF (Term Frequency-Inverse Document Frequency)** to highlight important words.
3.  **Model Training:**
    * Trained a **Logistic Regression** classifier, which is efficient for high-dimensional sparse data like text.
4.  **Deployment:**
    * The trained model is saved as `test_model.joblib` for reuse.

---

## 📊 Visualizations
The project includes Exploratory Data Analysis (EDA) to understand the "DNA" of tweets:
* **Positive Word Cloud:** Dominant words included *"love", "thank", "good", "happy"*.
* **Negative Word Cloud:** Dominant words included *"sad", "miss", "bad", "sorry"*.

---

## 📝 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/TomarBondhuRashed/twitter-sentiment-analysis.git](https://github.com/TomarBondhuRashed/twitter-sentiment-analysis.git)
    ```
2.  **Install dependencies:**
    ```bash
    pip install pandas numpy scikit-learn nltk matplotlib wordcloud joblib
    ```
3.  **Run the Notebook:**
    Open `Twitter_Sentiment_Analysis.ipynb` to see the training process and visualizations.

4.  **Load the Saved Model:**
    ```python
    import joblib
    model = joblib.load('test_model.joblib')
    prediction = model.predict(["I love machine learning!"])
    print(prediction)
    ```

---

## 🤝 Connect
Follow my 30-day coding journey!
* **LinkedIn:** [Your Profile Link]
* **GitHub:** [TomarBondhuRashed](https://github.com/TomarBondhuRashed)
