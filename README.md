# Sentiment Analysis of Movie Reviews

This project served as my introduction to practical NLP. I explored multiple approaches to sentiment analysis—starting with classic machine-learning baselines and progressing toward modern deep-learning and transformer models. The goal was to understand how different methods behave, what their strengths are, and how they compare across datasets.

#### The work was carried out in three main phases:

#### 1. Classical Machine Learning

Implemented baseline models using TF-IDF representations combined with Logistic Regression and Linear SVM. This provided a strong, interpretable reference point for evaluating more advanced models.

#### 2. Recurrent Neural Networks

Built a Bi-LSTM model and conducted a structured experiment comparing several embedding strategies:

Randomly initialized embeddings

Pretrained GloVe vectors

Custom Word2Vec embeddings

I also examined how performance changes when embeddings are fine-tuned versus kept frozen during training.

#### 3. Transformer Models

Fine-tuned a RoBERTa-based classifier to establish a benchmark using a state-of-the-art architecture.

All models were tested on two datasets with different characteristics: long IMDB movie reviews and short, dense snippets from Rotten Tomatoes.

#### Tech Stack

Core: Python, PyTorch, Scikit-learn, Pandas, NumPy
NLP: Hugging Face Transformers, Gensim, NLTK / spaCy (tokenization)
Tools: Jupyter Notebook, Git, GitHub

#### Reproducibility & Experimental Setup

All experiments use consistent train/validation/test splits.

Randomness is fully controlled through a fixed seed.

Hyperparameters were tuned through iterative experimentation with the goal of illustrating conceptual differences between approaches-rather than exhaustively optimizing each model.

## Project Structure

```
├── datasets/
│   ├── imdb.csv
│   ├── imdb_unsupervised.csv
│   ├── imdb_unsupervised_clean.csv
│   └── rt.csv
│
├── imdb_split/
│   ├── imdb_train.csv
│   └── imdb_test.csv
│
├── rt_split/
│   ├── rt_train.csv
│   └── rt_test.csv
│
├── notebooks/
│   ├── 00_Preprocessing_and_EDA.ipynb       # Data cleaning, EDA, and train/test splits
│   ├── 01_ML_Models.ipynb                   # TF-IDF, n-grams, Logistic Regression, Linear SVM
│   ├── 02_Word2Vec_Train.ipynb              # Training custom Word2Vec embeddings
│   ├── 03_LSTM.ipynb                        # Bi-LSTM experiments (random/GloVe/W2V embeddings)
│   └── 04_Roberta.ipynb                     # Fine-tuning RoBERTa sentiment model
│
├── src/
│   ├── __init__.py                          # Marks this directory as a Python package
│   └── config.py                            # Centralized config: paths, seeds, constants
│
├── w2v_model/
│   ├── w2v_model                            # Trained gensim Word2Vec model
│   ├── w2v_model.syn1neg.npy                # Negative sampling weights
│   └── w2v_model.wv.vectors.npy             # Word vectors matrix
│
├── README.md                                # Project documentation
├── requirements.txt                         # Python dependencies
```
