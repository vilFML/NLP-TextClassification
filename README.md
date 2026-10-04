# Report

## Problem Presentation
This assignment evaluates binary sentiment classification on book
reviews. Given a review's text as input, the model assigns a positive or
negative sentiment as output. The dataset consists of 12,000 raw Amazon
Kindle book reviews containing the following fields: *ProductId*, star
rating (1--5), full review text, a short review headline, and
helpfulness notes. It is available on
[Kaggle](https://www.kaggle.com/datasets/meetnagadia/amazon-kindle-book-review-for-sentiment-analysis).\
Examples: *\"This book was an absolute page-turner; I couldn't put it
down\"*, *\"It was okay---nothing special, but not bad either\"*,
*\"Poorly written; I regret buying this.\"*

## Text Representation
Because the dataset comprises raw reviews, a preprocessing pipeline is
required to normalize the text:

- Conversion to lowercase.

- Removal of HTML markup and punctuation.

- Elimination of empty review fields.

Once formatted, the text is represented using the **Bag-of-Words** (BoW)
model after removing English stop words.

## Classifier

The classification task is addressed using a Multinomial Naive Bayes
model with Laplace smoothing. The model is trained on word-count vectors
to estimate the conditional probability $$P(\text{word}|\text{class})$$
for each vocabulary term in the training set, applying Bayes' theorem to
predict the class of a given review.

## Limitation

A primary limitation of this binary implementation is the forced
decision between positive and negative classes, even when a review
expresses mixed or ambiguous sentiment. This constraint misrepresents
the underlying data by imposing artificial polarization on the
sentiment.

## Improvement

The original dataset contains raw numerical ratings rather than
categorical sentiment labels. Consequently, classes were derived from
star ratings: 1--2 stars: **Negative**, 3 stars: **Neutral**, and 4--5
stars: **Positive**. The same classifier architecture was trained using
these three categories. Incorporating the neutral label allows the model
to accurately represent reviews with mixed sentiment.

# Results

## Class Distribution

The dataset contains 12,000 instances. The class distribution reveals a
notable imbalance among review sentiments.

  ------------- ------
  (12000, 11)   
  sentiment     
  positive      6000
  negative      4000
  neutral       2000
  ------------- ------

The relatively small proportion of neutral reviews allows the binary
model to yield an apparently satisfactory classification performance
while ignoring this category.

## Forced Binary Model

  --------------- ----------- -------- ---------- ---------
                  Precision   Recall   F1-Score   Support
  Negative        0.65        0.88     0.75       1200
  Neutral         0.00        0.00     0.00       600
  Positive        0.79        0.87     0.82       1800
  Accuracy                             0.73       3600
  Macro Avg.      0.48        0.58     0.52       3600
  Weighted Avg.   0.61        0.73     0.66       3600
  --------------- ----------- -------- ---------- ---------

The neutral class yields precision, recall, and F1-score values of 0.00
because the model was not trained on neutral instances, resulting in an
undefined division $$\frac{0}{0}$$ (zero correct predictions out of zero
positive classifications). Although the overall accuracy is reported at
0.73, this metric is misleading as it completely masks the total failure
to classify neutral instances. Conversely, the Macro Average accounts
for all three classes equally, reflecting the true drop in performance.

## Three-Class Model

  --------------- ----------- -------- ---------- ---------
                  Precision   Recall   F1-Score   Support
  Negative        0.70        0.82     0.76       1200
  Neutral         0.38        0.27     0.31       600
  Positive        0.82        0.81     0.82       1800
  Accuracy                             0.72       3600
  Macro Avg.      0.64        0.63     0.63       3600
  Weighted Avg.   0.71        0.72     0.71       3600
  --------------- ----------- -------- ---------- ---------

While overall accuracy slightly decreased due to minor drops in positive
and negative scores, the macro average improved from 0.52 to 0.63,
representing a favorable trade-off.

Cross-validation results (5-fold CV accuracy: $0.7226 \pm 0.0036$)
confirm that the model's performance is stable and independent of
specific train-test partitions.

## Discriminative Words Feature Analysis

Analysis of the top discriminative features reveals that the most
influential words are proper nouns (e.g., \"ted\", \"cassidy\",
\"meli\") and domain-specific terms rather than explicit sentiment
indicators. This occurs because specific book characters or author names
appear exclusively in highly rated or poorly rated books, causing the
model to associate those terms directly with positive or negative
sentiment. Consequently, a key future improvement would involve
filtering out proper nouns during feature extraction.

![image](./imagenes/wordGraph.png)

# Declarative Use of LLM

For the developing of this assignment, LLMs were used with the purpose
of brainstorming and code debugging. As English is not the author's
native language, the report's text was parsed to maintain an academic
formal tone.
