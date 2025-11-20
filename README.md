## Sentiment Analysis of Movie Reviews

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
