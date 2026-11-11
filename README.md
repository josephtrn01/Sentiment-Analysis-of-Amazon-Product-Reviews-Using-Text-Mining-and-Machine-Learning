Amazon Review Sentiment Analysis
Turn customer review text into sentiment predictions and discover recurring themes in customer feedback.

An R-based text mining project that compares Naive Bayes and multinomial logistic regression for three-class sentiment classification, then uses Latent Dirichlet Allocation (LDA) to explore what customers discuss.

The best recorded model achieved 87.41% test accuracy on 4,211 reviews. Neutral sentiment remains a major challenge, making class-level performance essential to interpreting the results.

Getting started · Results · Topic insights · Limitations
What this project does
Customer reviews contain useful feedback about delivery, refunds, customer service, and shopping experiences. This project demonstrates how to analyze that feedback with a complete workflow:
- Clean review text and create sentiment labels from star ratings.
- Convert text into numerical features using TF-IDF.
- Train and compare two sentiment classifiers.
- Evaluate predictions with accuracy, confusion matrices, precision, recall, and F1.
- Explore recurring discussion themes with LDA topic modeling.
- Save trained models, processed datasets, charts, and evaluation tables.
Potential applications include summarizing customer satisfaction, screening feedback for further review, and identifying operational issues worth investigating. This repository is an academic analysis pipeline; it does not currently include an interactive application or a prediction API.
Dataset
The repository includes [`Amazon_Reviews.csv`](Data/Raw/Amazon_Reviews.csv), containing 21,214 raw records. The committed cleaned dataset contains 21,055 reviews.

The input includes review text, ratings, review titles, dates, and reviewer metadata. The classifiers use review text as input and rating-derived sentiment as the target; reviewer metadata and ratings are not used as predictive features.
Star rating	Sentiment	Cleaned reviews	Share
1–2	Negative	14,350	68.15%
3	Neutral	885	4.20%
4–5	Positive	5,820	27.64%
Total		21,055	100%


Percentages are rounded. The data includes feedback about Amazon's services and shopping experience as well as products, so findings should be interpreted within that broader scope.

Source note: The accompanying report cites the Kaggle Amazon Reviews Dataset by Dongrelaxman, but also refers to “Amazon Fine Food Reviews” elsewhere. This README describes the CSV actually committed to this repository; the original dataset attribution should be reconciled before reuse or redistribution.
How it works
1. Data preparation
The preprocessing script standardizes column names, removes duplicate rows and records with missing review text or ratings, extracts numeric ratings, and assigns sentiment labels. It lowercases text, removes characters outside English letters and whitespace, and normalizes spacing.

Before classification, records with invalid labels or empty cleaned text are excluded.
2. Feature engineering
A stratified 80/20 train-test split preserves the sentiment distribution, using seed 123:
Split	Reviews
Training	16,844
Testing	4,211


Using quanteda, the pipeline tokenizes the text, removes English stopwords, builds document-feature matrices, and aligns the test vocabulary to the training vocabulary. TF-IDF weighting produces the features used by both classifiers.
3. Sentiment classification
Model	Implementation
Naive Bayes	quanteda.textmodels::textmodel_nb trained on TF-IDF features
Multinomial logistic regression	glmnet::cv.glmnet with ridge regularization (alpha = 0), cross-validation, and predictions at lambda.min


Both models are evaluated on the same held-out test set. Trained models are saved as .rds files.
4. Topic discovery
A separate exploratory branch samples up to 5,000 cleaned reviews, using seed 42. It tokenizes text, removes stopwords, creates a word-count document-term matrix, and fits an LDA model with four topics. It exports the ten highest-probability terms per topic and a visualization.

LDA provides a corpus-level view of recurring themes. Its outputs are not fed into the sentiment classifiers, and topics are not explanations of individual predictions.
Results
The following values come from the committed evaluation CSVs.
Model	Test accuracy	Cohen's kappa
Naive Bayes	87.41%	Not exported
Logistic regression	86.75%	0.6747
Majority-class baseline: always negative	68.15%	—


The baseline is calculated from the test-set class distribution. Naive Bayes exceeds it by approximately 19.26 percentage points and exceeds logistic regression by 0.66 percentage points. These are results from one train-test split; statistical significance was not assessed.

Performance by sentiment
Model	Sentiment	Precision	Recall	F1
Naive Bayes	Negative	93.05%	92.37%	0.9271
Naive Bayes	Neutral	13.22%	9.04%	0.1074
Naive Bayes	Positive	81.71%	87.11%	0.8432
Logistic regression	Negative	85.92%	98.43%	0.9175
Logistic regression	Neutral	0.00%	0.00%	Undefined in export
Logistic regression	Positive	89.80%	71.13%	0.7939


Interpretation: Naive Bayes provides the better overall accuracy and positive-class recall in this experiment. Logistic regression identifies most negative reviews but correctly identifies no neutral reviews. Naive Bayes correctly identifies only 16 of 177 neutral reviews. Overall accuracy therefore overstates the models' usefulness for balanced three-class classification.

The logistic regression neutral F1 is recorded as NA in the exported metrics; it is left undefined here to match that file.
<details>
<summary>View confusion matrices</summary>
Rows represent predicted sentiment; columns represent actual sentiment.
Naive Bayes

Logistic regression

</details>Topic insights
The following names are human interpretations of the top terms, rather than labels generated or independently validated by the model.
Topic	Suggested theme	Representative terms
1	Customer service and refunds	customer, service, refund, card, call
2	Delivery and package handling	delivery, delivered, package, driver, time
3	Prime and shopping experience	prime, shipping, products, reviews, free
4	Accounts, returns, and sellers	account, return, money, seller, company



The bars show topic-word probabilities, not sentiment scores. These themes can guide further investigation into customer concerns, but they do not establish why a review received a particular sentiment prediction.
Getting started
Requirements
Install R and optionally RStudio. From an R console, install the packages used by the scripts:
install.packages(c(
  "readr", "dplyr", "stringr", "janitor", "ggplot2",
  "caret", "quanteda", "quanteda.textmodels", "reshape2",
  "glmnet", "Matrix", "tidytext", "topicmodels", "tm"
))
Some packages may require platform-specific build tools if binary packages are unavailable. The repository does not currently pin R or package versions.
Clone the repository
git clone https://github.com/josephtrn01/Sentiment-Analysis-of-Amazon-Product-Reviews-Using-Text-Mining-and-Machine-Learning.git
cd Sentiment-Analysis-of-Amazon-Product-Reviews-Using-Text-Mining-and-Machine-Learning
Run the analysis
Open R or RStudio with the repository root as the working directory. All script paths are relative to that directory. Run:
# Ensure output directories exist.
for (path in c("Data/Processed", "Outputs/figures", "Outputs/tables")) {
  dir.create(path, recursive = TRUE, showWarnings = FALSE)
}

source("Scripts/01_data_loading_preprocessing.R")
source("Scripts/02_feature_engineering_tf_idf.R")
source("Scripts/03_naive_bayes_model.R")
source("Scripts/04_logistic_regression_model.R")
source("Scripts/05_lda_topic_modeling.R")
source("Scripts/06_model_comparison_and _summary.R")
The space before _summary.R in the final filename is intentional: the command matches the current repository filename.

Naive Bayes chart export: Script 03 displays its confusion matrix but does not save the image automatically. To regenerate the committed image, run the following immediately after sourcing Script 03, before Script 04 overwrites cm_plot:
ggplot2::ggsave(
  "Outputs/figures/Naive Bayes Confusion matrix.png",
  plot = cm_plot, width = 6, height = 5, dpi = 300
)
Running the pipeline overwrites generated datasets, models, tables, and saved figures. The LDA step may take additional time depending on your machine.
Repository guide
Path	Contents
[`Data/Raw/`](Data/Raw/)	Input review CSV
[`Data/Processed/`](Data/Processed/)	Cleaned reviews, train/test splits, TF-IDF features, target labels, and saved models
[`Scripts/`](Scripts/)	Six numbered R scripts covering the analysis workflow
[`Outputs/figures/`](Outputs/figures/)	Sentiment distribution, confusion matrices, accuracy comparison, and LDA topics
[`Outputs/tables/`](Outputs/tables/)	Overall accuracy, class-level metrics, model comparison, and topic terms
[`Report/`](Report/)	Academic project report
[`Presentation/`](Presentation/)	Project presentation


Limitations and future work
- Class imbalance: Neutral reviews account for only 4.20% of the cleaned data. Explore class weights or training-only resampling, and prioritize macro F1 and per-class recall alongside accuracy.
- TF-IDF consistency: The current script calculates TF-IDF separately on training and test matrices. A future version should learn IDF weights from training data and reuse those weights for test data and new reviews, then rerun evaluation.
- Rating-derived labels: Star ratings serve as proxies for sentiment and may not fully capture mixed opinions or disagreement between the rating and text.
- Text preprocessing: English-only character filtering and stopword removal can discard useful information, including negation. Explore negation-aware preprocessing and word or character n-grams.
- Evaluation scope: Results come from one held-out split. Repeated evaluation and testing on an independent dataset would give stronger evidence of generalization.
- Topic selection: Four topics were chosen manually. Evaluate coherence and alternative topic counts before drawing stronger conclusions.
- Reproducibility and deployment: Add an renv lockfile and a prediction function that preserves preprocessing, vocabulary, and IDF weights. A dashboard or API would be a future extension.
Team
Developed by Tung Tran, Mansi Sanjay Bhosle, and Hindolo Dominic Magbity for DS515: Data Science Overview, MS in Data Science, City University of Seattle.
