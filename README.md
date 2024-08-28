# Sentiment Analysis of Tweets

This project performs sentiment analysis on tweets using various machine learning models. The objective is to classify tweets into categories such as Positive, Negative, Neutral, and Irrelevant. The dataset consists of labeled tweets, and the project covers data preprocessing, feature extraction, model training, evaluation, and analysis of results.

## Overview

The project involves sentiment analysis on a collection of tweets, aiming to classify them based on sentiment. The main steps include data preprocessing, exploratory data analysis, training machine learning models, and evaluating the performance to draw insights.

## Dataset

The dataset includes:
- `twitter_training.csv`: Training data with columns `ID`, `entity`, `output` (sentiment), and `tweet`.
- `twitter_validation.csv`: Validation data for evaluating model performance.

### Key Statistics:
- The training dataset contains several rows of tweets labeled with sentiments.
- Unique entities and a diverse range of sentiment categories are represented.

## Data Preprocessing

Preprocessing steps include:
- **Lowercasing**: Converting text to lowercase to ensure uniformity.
- **Removing Special Characters**: Cleaning text using regular expressions to remove punctuation and special symbols.
- **Tokenization**: Splitting text into individual tokens (words).
- **Stopword Removal**: Eliminating common English stopwords to focus on meaningful words.

## Exploratory Data Analysis (EDA)

### Entity Distribution
![Entities Distribution](#)

- **Insight**: The bar plot shows the distribution of entities within the dataset, highlighting which entities are most frequently mentioned. The counts suggest a balanced dataset across different entities, with the most frequent entities slightly above the average.

### Sentiment Distribution
![Sentiment Distribution](#)

- **Insight**: The pie chart reveals the proportion of each sentiment in the dataset. Positive and Negative sentiments have significant representation, while Neutral and Irrelevant categories are less frequent. This imbalance could influence the model's ability to accurately predict less frequent categories.

### Cross-Tabulation of Entities and Sentiments
- **Insight**: The cross-tabulation heatmap illustrates the frequency of each sentiment for different entities. It highlights which entities are predominantly associated with positive or negative sentiments, offering valuable context for model predictions.

## Feature Engineering

- **Bag of Words (BoW)**: Applied CountVectorizer to transform text into numerical format, which is essential for training machine learning models.
- **N-grams**: Considered up to four-grams to capture complex text patterns and improve the model’s understanding of context within tweets.

## Model Training

The primary model used:
- **Logistic Regression**: A linear model trained on the Bag of Words features. It was chosen for its simplicity and effectiveness in text classification tasks.

### Training Details:
- **Solver**: liblinear
- **Regularization Parameter (C)**: 0.9
- **Max Iterations**: 1500

## Evaluation

The model's performance was evaluated using:
- **Accuracy Score**: Measures overall correctness of predictions.
- **Confusion Matrix**: Helps visualize the classification results and identify where the model is making errors.
- **Classification Report**: Provides precision, recall, and F1-score for each category, offering a detailed view of model performance.

### Key Results:
- **Training Accuracy**: The model achieved a high training accuracy of **91%**, indicating a strong fit on the training data. This suggests that the model has learned the underlying patterns in the data well, particularly for the Positive and Negative categories, which have high precision and recall scores.
  
- **Validation Accuracy**: The validation accuracy was **98.7%**, showing that the model generalizes exceptionally well to unseen data. The high accuracy across both training and validation sets suggests minimal overfitting, which is an excellent outcome for a sentiment analysis task.

- **Classification Report**: The classification reports indicate that all sentiment categories have high precision, recall, and F1-scores, with values around 0.99 for validation data. This demonstrates the model’s ability to correctly identify and distinguish between different sentiment labels, with **Positive** and **Negative** classes having slightly higher precision and recall compared to **Neutral** and **Irrelevant** classes.

- **Confusion Matrix Analysis**:
  - For the training set, most misclassifications occurred between the Neutral and Irrelevant categories, as well as between Irrelevant and Positive. This is expected given that these categories can have overlapping language.
  - For the validation set, the confusion matrix shows minimal errors across all categories, further validating the model’s robustness.

### Possible Reasons for High Accuracy:
1. **Balanced Representation**: Despite initial imbalances in sentiment categories, effective preprocessing and model training have managed to capture the distinctions between categories effectively, leading to high accuracy.
2. **Feature Representation**: Using Bag of Words with n-grams allowed the model to capture significant patterns and context within tweets, enhancing its predictive power.
3. **Model Selection**: Logistic Regression proved to be a strong choice for this classification task due to its simplicity and effectiveness in handling text data.

### Cross-Tabulation of Entities and Sentiments
- The crosstab analysis revealed how sentiments vary across different entities:
  - **Top Positive Sentiment**: Entities like AssassinsCreed and RedDeadRedemption(RDR) had high counts of positive sentiment, reflecting strong approval from users.
  - **Top Negative Sentiment**: Entities like MaddenNFL and NBA2K were associated with a higher number of negative tweets, potentially indicating dissatisfaction among users.
  - **Irrelevant and Neutral**: Entities such as WorldOfCraft and Cyberpunk2077 had balanced representations across sentiments, showing mixed public reactions.

This analysis provides valuable insights into brand perception and public opinion, which can inform marketing and engagement strategies.

## Visualizations

### Word Clouds for Sentiments

#### Positive Sentiment
![Positive Word Cloud](#)
- **Insight**: Common words in positive tweets include expressions of satisfaction, approval, and joy, such as "love," "great," and "happy." These reflect the overall positive tone expected in this category.

#### Negative Sentiment
![Negative Word Cloud](#)
- **Insight**: Negative sentiment word clouds highlight terms like "bad," "hate," and "worst," indicating dissatisfaction and complaints. This visualization aligns with the expected language in negative tweets.

#### Neutral Sentiment
![Neutral Word Cloud](#)
- **Insight**: Words in the neutral category are often factual or descriptive without a strong opinion, such as "today," "meeting," or "report." The overlap with other categories can make these tweets challenging to classify.

#### Irrelevant Sentiment
![Irrelevant Word Cloud](#)
- **Insight**: Irrelevant tweets contain diverse language, often unrelated to the entities or topics of interest. Common words may not align closely with typical sentiment expressions, which complicates classification.

### Entity and Sentiment Distribution
![Entity and Sentiment Distribution](#)
- **Insight**: This bar plot shows the count of tweets per entity across different sentiments, highlighting entities with predominantly positive or negative feedback. It helps identify which brands or topics receive mixed reactions.

## Conclusion

This sentiment analysis project demonstrates the capability of logistic regression in classifying tweet sentiments. Although the model performs reasonably well, there is room for improvement, particularly in handling imbalanced datasets and overlapping sentiment categories. Future work could involve more advanced models like neural networks or transformer-based approaches (e.g., BERT) for enhanced performance.

## Acknowledgments

Special thanks to the open-source tools and libraries used in this project, including Pandas, Scikit-learn, Matplotlib, and others, for providing robust frameworks for data analysis and visualization.
