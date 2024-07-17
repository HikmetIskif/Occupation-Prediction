# Occupation Prediction Using Twitter Data

## Overview
This project aims to predict the occupation of Twitter users based on their Turkish tweets. The study utilizes machine learning and deep learning techniques to analyze two datasets comprising 25,000 and 30,000 tweets, respectively. Various models, including Logistic Regression, Naive Bayes, Random Forest, Support Vector Machine, Multi-Layer Perceptron, Convolutional Neural Network, Recurrent Neural Network, and BERT, are evaluated to determine the best performance.

## Datasets
Two datasets are used in this project:
1. **Dataset 1:** 25,000 tweets from 10 different occupations (2,500 tweets per occupation).
2. **Dataset 2:** 30,000 tweets, an extended version of Dataset 1 with additional 5,000 tweets.

### Occupations Covered:
- Lawyer
- Dietitian
- Doctor
- Economist
- Teacher
- Psychologist
- Sports Commentator
- Historian
- Software Developer
- Agricultural Engineer

### Preprocessing Steps:
- Removal of punctuation
- Conversion to lowercase
- Removal of links
- Root extraction using Zeyrek and Zemberek libraries

## Feature Extraction
- **TF-IDF (Term Frequency-Inverse Document Frequency)**
- **Bag-of-Words Vectors**

## Models Used
### Machine Learning Models:
- Logistic Regression
- Naive Bayes
- Random Forest
- Support Vector Machine

### Deep Learning Models:
- Multi-Layer Perceptron (MLP)
- Convolutional Neural Network (CNN)
- Recurrent Neural Network (RNN)
  - Variants: GRU, LSTM, bi-LSTM

### Pre-trained Model:
- BERT (Bidirectional Encoder Representations from Transformers)

## Results for Dataset with 25,000 Data

| **Method Type**    | **Method**            | **NLP**    | **Single** | **Pairwise** |
|--------------------|-----------------------|------------|------------|--------------|
| **Machine Learning** | Logistic Regression   | Zeyrek     | 0.7316     | 0.8507       |
| **Machine Learning** | Logistic Regression   | Zemberek   | **0.7374** | **0.8509**   |
| **Machine Learning** | Naive Bayes           | Zeyrek     | 0.7231     | 0.8197       |
| **Machine Learning** | Naive Bayes           | Zemberek   | 0.7298     | 0.8378       |
| **Machine Learning** | Random Forest         | Zeyrek     | 0.6531     | 0.7618       |
| **Machine Learning** | Random Forest         | Zemberek   | 0.6789     | 0.7858       |
| **Machine Learning** | Support Vector Machine | Zeyrek     | 0.7199     | 0.8435       |
| **Machine Learning** | Support Vector Machine | Zemberek   | 0.7307     | 0.8421       |
| **Deep Learning**  | MLP                   | Zeyrek     | 0.7231     | 0.8368       |
| **Deep Learning**  | MLP                  | Zemberek   | **0.7310** | **0.8517**   |
| **Deep Learning**  | CNN                   | Zeyrek     | 0.7003     | 0.7951       |
| **Deep Learning**  | CNN                   | Zemberek   | 0.7210     | 0.8277       |
| **Deep Learning**  | LSTM                  | Zeyrek     | 0.6697     | 0.7332       |
| **Deep Learning**  | LSTM                  | Zemberek   | 0.6709     | 0.7573       |
| **Deep Learning**  | Bi-LSTM               | Zeyrek     | 0.6583     | 0.7636       |
| **Deep Learning**  | Bi-LSTM               | Zemberek   | 0.6836     | 0.7840       |
| **Deep Learning**  | GRU                   | Zeyrek     | 0.6552     | 0.7255       |
| **Deep Learning**  | GRU                   | Zemberek   | 0.6609     | 0.7149       |
| **Pre Trained**    | BERT                  | Zeyrek     | 0.7586     | 0.8699       |
| **Pre Trained**    | BERT                  | Zemberek   | **0.7596** | **0.8730**   |

## Results for Dataset with 30,000 Data

| **Method Type**   | **Method**                | **NLP**  | **Single** | **Pairwise** |
|-------------------|---------------------------|----------|------------|--------------|
| **Machine Learning** | Logistic Regression     | Zeyrek   | 0.7148     | **0.8448**   |
| **Machine Learning** | Logistic Regression     | Zemberek | **0.7260** | 0.8374       |
| **Machine Learning** | Naive Bayes             | Zeyrek   | 0.6978     | 0.8191       |
| **Machine Learning** | Naive Bayes             | Zemberek | 0.7225     | 0.8149       |
| **Machine Learning** | Random Forest             | Zeyrek   | 0.6475     | 0.7519       |
| **Machine Learning** | Random Forest             | Zemberek | 0.6740     | 0.7708       |
| **Machine Learning** |  Support Vector Machine   | Zeyrek   | 0.7051     | 0.8445       |
| **Machine Learning** |  Support Vector Machine   | Zemberek | 0.7200     | 0.8444       |
| **Deep Learning**  | MLP                        | Zeyrek   | 0.7179     | 0.8334       |
| **Deep Learning** |  MLP                        | Zemberek | **0.7375** | **0.8401**   |
| **Deep Learning**  | CNN                        | Zeyrek   | 0.6931     | 0.8013       |
| **Deep Learning** | CNN                        | Zemberek | 0.7212     | 0.8190       |
| **Deep Learning**  | LSTM                       | Zeyrek   | 0.6321     | 0.7383       |
| **Deep Learning** | LSTM                       | Zemberek | 0.6683     | 0.7403       |
| **Deep Learning**  | Bi-LSTM                    | Zeyrek   | 0.6719     | 0.7597       |
| **Deep Learning**  | Bi-LSTM                    | Zemberek | 0.6953     | 0.7714       |
| **Deep Learning**  | GRU                        | Zeyrek   | 0.6427     | 0.7367       |
| **Deep Learning**  | GRU                        | Zemberek | 0.6730     | 0.7321       |
| **Pre Trained**   | BERT                       | Zeyrek   | 0.7427     | 0.8525       |
| **Pre Trained**   | BERT                       | Zemberek | **0.7760** | **0.8558**   |


## Results
- **Logistic Regression** achieved the highest accuracy among machine learning models.
- **Multi-Layer Perceptron (MLP)** was the most successful deep learning model.
- **BERT** outperformed all models in both datasets, achieving the highest accuracy.

## Future Work
- Exploring more sophisticated preprocessing techniques
- Testing additional machine learning and deep learning models
- Expanding the dataset to include more occupations and larger tweet samples
- Integrating other social media platforms for a broader analysis

The dataset of size 25.000 can be found [here](https://github.com/imayda/occupation-dataset-in-turkish).

The dataset of size 30.000, created by adding new data, can be found [here](https://www.kaggle.com/datasets/tolgaizdas/turkish-tweet-dataset-for-occupation-prediction).

## Contact
For any inquiries, please contact:
- Tolga İzdaş: [tolga.izdas@std.yildiz.edu.tr](mailto:tolga.izdas@std.yildiz.edu.tr)
- Hikmet İskifoğlu: [hikmet.iskifoglu@std.yildiz.edu.tr](mailto:hikmet.iskifoglu@std.yildiz.edu.tr)

