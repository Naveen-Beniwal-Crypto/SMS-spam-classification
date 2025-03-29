# SMS Spam Classification

## Overview
This project implements an SMS spam classification model using machine learning techniques. The dataset used is `SMSSpamCollection`, which contains labeled messages as either `ham` (non-spam) or `spam`. The primary goal is to preprocess the text data, extract meaningful features, and train classification models to identify spam messages.

## Features
- **Data Preprocessing**:
  - Handling imbalanced datasets using oversampling.
  - Feature engineering: word count, presence of currency symbols, presence of numbers.
  - Text cleaning: removal of special characters, stopwords, and lemmatization.
- **Feature Extraction**:
  - TF-IDF vectorization for transforming text data into numerical format.
- **Model Training**:
  - Naïve Bayes Model
  - Decision Tree Classifier
- **Evaluation Metrics**:
  - Classification report
  - Confusion matrix
  - Cross-validation scores
- **Prediction Function**:
  - A function to classify new SMS messages as spam or ham.

## Dataset
- **Source**: `SMSSpamCollection`
- **Format**: Tab-separated values (TSV)
- **Columns**:
  - `label`: (0 - ham, 1 - spam)
  - `message`: The text message content

## Libraries Used
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `nltk`
- `scikit-learn`

## Implementation Details
### 1. Data Preprocessing
- Mapped labels (`ham` → `0`, `spam` → `1`).
- Oversampling applied to balance spam messages.
- Extracted new features:
  - `word_count`: Number of words in a message.
  - `contains_currency_symbols`: Checks for currency symbols (`€, $, ¥, £, ₹`).
  - `contains_numbers`: Checks for numeric characters in the message.
- Visualized dataset distribution with count plots and histograms.

### 2. Text Cleaning
- Converted text to lowercase.
- Removed special characters and numbers.
- Tokenized words and removed stopwords.
- Applied lemmatization using `nltk`.

### 3. Feature Extraction
- Used **TF-IDF Vectorizer** (max features = 500) to represent text numerically.

### 4. Model Training and Evaluation
- **Naïve Bayes Classifier**:
  - Applied `MultinomialNB` from `sklearn.naive_bayes`.
  - Cross-validation (`F1-score`): Mean and Standard Deviation calculated.
  - Generated classification report and confusion matrix.
- **Decision Tree Classifier**:
  - Used `DecisionTreeClassifier` from `sklearn.tree`.
  - Cross-validation (`F1-score`) computed.
  - Evaluated performance using classification report and confusion matrix.

### 5. Spam Prediction Function
- Defined `predict_spam()` function to classify new SMS messages.
- Preprocesses input text and applies trained `DecisionTreeClassifier` for classification.

## Example Prediction
```python
sample_message = "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"
if predict_spam(sample_message):
  print("This is a Spam message.")
else:
  print("This is a Ham(normal) message")
```

## Results
- **Naïve Bayes Model**:
  - Achieved high accuracy and F1-score.
  - Performed well with limited features.
- **Decision Tree Model**:
  - Showed slightly higher variance compared to Naïve Bayes.
  - More sensitive to data distribution.

## Future Improvements
- Implement additional models like SVM and deep learning-based classifiers.
- Fine-tune hyperparameters for better performance.
- Explore feature selection and dimensionality reduction techniques.

## How to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/Naveen-Beniwal-Crypto/SMS-spam-classification.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
4. Open `SMS_Spam_Classification.ipynb` and execute the cells.

## Author
**Naveen Beniwal**

## License
This project is licensed under the MIT License.

