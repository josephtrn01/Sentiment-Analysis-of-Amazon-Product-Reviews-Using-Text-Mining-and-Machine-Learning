
# Sentiment Analysis of Amazon Product Reviews Using Text Mining and Machine Learning

![R](https://img.shields.io/badge/R-Programming-276DC3?logo=r&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-NLP-blue)
![TF-IDF](https://img.shields.io/badge/Feature%20Extraction-TF--IDF-orange)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

## Project Overview

This project applies **Natural Language Processing (NLP), Text Mining, and Machine Learning** to analyze Amazon customer reviews and automatically classify customer sentiment.

Using **TF-IDF feature extraction, Naive Bayes, and Logistic Regression**, the project predicts whether customer reviews express positive, neutral, or negative sentiment. Additionally, **Latent Dirichlet Allocation (LDA)** is used to discover recurring topics and customer concerns.

The goal is to demonstrate how businesses can transform large volumes of unstructured customer feedback into meaningful, data-driven insights.

## Technologies Used

- **Programming Language:** R
- **Data Processing:** dplyr, readr, stringr, quanteda
- **Feature Engineering:** TF-IDF
- **Machine Learning:** Naive Bayes, Multinomial Logistic Regression
- **Topic Modeling:** Latent Dirichlet Allocation (LDA)
- **Visualization:** ggplot2
- **Model Evaluation:** Accuracy, Precision, Recall, F1-score, Confusion Matrix

## Project Workflow

1. **Data Preprocessing:** Clean customer reviews, remove duplicates and missing values, and normalize text.
2. **Sentiment Labeling:** Convert star ratings into Negative (1–2), Neutral (3), and Positive (4–5).
3. **Feature Engineering:** Apply tokenization, stopword removal, and TF-IDF vectorization.
4. **Model Training:** Train Naive Bayes and Logistic Regression models using an 80/20 train-test split.
5. **Model Evaluation:** Compare classification accuracy and confusion matrices.
6. **Topic Modeling:** Apply LDA to identify recurring customer discussion themes.

## Model Performance

| Model | Accuracy | Cohen's Kappa |
|---|---:|---:|
| **Naive Bayes** | **87.41%** | **0.72** |
| Logistic Regression | 86.75% | 0.6747 |

Naive Bayes achieved the highest overall accuracy, slightly outperforming Logistic Regression. Both models performed well on positive and negative reviews but struggled to identify neutral sentiment due to class imbalance.

### Model Accuracy Comparison

![Model Accuracy Comparison](Outputs/figures/model_accuracy_comparison.png)

### Confusion Matrices

**Naive Bayes**

![Naive Bayes Confusion Matrix](Outputs/figures/Naive%20Bayes%20Confusion%20matrix.png)

**Logistic Regression**

![Logistic Regression Confusion Matrix](Outputs/figures/logistic_regression_confusion_matrix.png)

## Topic Modeling (LDA)

LDA identified four major discussion themes within customer reviews:

| Topic | Theme |
|---|---|
| 1 | Customer Service and Refund Issues |
| 2 | Delivery and Shipping Experience |
| 3 | Amazon Prime and Shopping Experience |
| 4 | Returns and Seller Disputes |

![LDA Topic Modeling](Outputs/figures/lda_topics.png)

These findings help explain common customer concerns beyond sentiment classification.

## Key Findings

- **Naive Bayes achieved the best performance** with 87.41% accuracy.
- **TF-IDF effectively transformed review text** into features suitable for machine learning.
- **Neutral sentiment remained challenging** due to class imbalance.
- **LDA revealed recurring customer concerns**, including delivery, customer service, and returns.
- Combining sentiment classification with topic modeling provides a broader understanding of customer feedback.

## How to Run

Clone the repository:

```bash
git clone https://github.com/josephtrn01/Sentiment-Analysis-of-Amazon-Product-Reviews-Using-Text-Mining-and-Machine-Learning.git
```

Install the required R packages and run the six R scripts in numerical order, starting with data preprocessing and ending with model comparison.

Ensure the input dataset is located at `Data/Raw/Amazon_Reviews.csv` and the required output directories exist.

## Future Improvements

Future work could explore class-balancing techniques, transformer-based NLP models such as BERT, and interactive dashboards for real-time sentiment analysis.

## Project Team

**City University of Seattle**  
Master of Science in Data Science | DS515 – Data Science Overview

- Tung Tran
- Mansi Sanjay Bhosle
- Hindolo Dominic Magbity

**Repository Maintainer:** [Tung Tran](https://github.com/josephtrn01)

---

*Developed as an academic data science project demonstrating practical applications of NLP, machine learning, and text analytics.*
