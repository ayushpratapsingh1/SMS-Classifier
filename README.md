# SMS Spam Classifier

The SMS Spam Classifier is a machine learning-based web application that classifies SMS messages as either spam or ham (non-spam). The model uses a range of machine learning algorithms to predict whether a given message is spam or not.

## Project Description

The SMS Spam Classifier project employs machine learning algorithms to identify spam messages. The algorithms tested include:
- **Bernoulli Naive Bayes**
- **Multinomial Naive Bayes**
- **Gaussian Naive Bayes**
- **Linear Regression**
- **Adaboost**
- **Random Forest**

Among these, **Multinomial Naive Bayes** demonstrated the best performance with an accuracy of **97%**, outperforming the other algorithms.

## Key Features
- **Data Preprocessing**: SMS dataset is cleaned and preprocessed for analysis.
- **Feature Extraction**: TF-IDF Vectorizer and CountVectorizer are used to convert text data into numerical features.
- **Model Training**: The project trains several classifiers (such as Bernoulli Naive Bayes, Multinomial Naive Bayes, etc.) on the preprocessed data.
- **Model Evaluation**: The performance of each algorithm is assessed using metrics like accuracy.
- **Model Selection**: Multinomial Naive Bayes is identified as the best-performing model.

## Tools & Technologies
- **Python**: Programming language used for the project.
- **Streamlit**: Framework for deploying the web application.
- **Scikit-learn**: For implementing machine learning algorithms.
- **NLTK**: Natural Language Toolkit for text preprocessing.
- **Pickle**: Used for saving the trained model and vectorizer for later use.
- **TF-IDF Vectorizer**: Used for transforming text data into numerical vectors based on term frequency-inverse document frequency.
- **CountVectorizer**: Converts text documents into a matrix of token counts.

## How to Run the Project Locally

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/sms-spam-classifier.git
   cd sms-spam-classifier
2. **Install the Required Packages:**
   ```bash
   pip install -r requirements.txt

3. **Run the Application:**
   ```bash
    streamlit run app.py
4. **Open a web browser and navigate to http://localhost:8501 to interact with the SMS Spam Classifier app.**

## File Structure
- app.py: The main script for running the Streamlit application.
- vectorizer.pkl: Serialized TF-IDF vectorizer used for text feature extraction.
- model.pkl: The trained model saved using pickle.
- requirements.txt: List of dependencies for the project.

## Model Performance
- Multinomial Naive Bayes: Achieved an accuracy of 97%.
- Other models, including Bernoulli Naive Bayes and Random Forest, were evaluated, but they showed lower performance than Multinomial Naive Bayes.
