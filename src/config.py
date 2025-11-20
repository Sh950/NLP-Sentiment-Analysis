from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

IMDB_PATH = PROJECT_ROOT / "datasets" / "imdb.csv"
RT_PATH = PROJECT_ROOT / "datasets" / "rt.csv"

IMDB_UNSUPERVISED_PATH = PROJECT_ROOT / "datasets" / "imdb_unsupervised.csv"
IMDB_UNSUPERVISED_CLEAN_PATH = PROJECT_ROOT / "datasets" / "imdb_unsupervised_clean.csv"

IMDB_TRAIN_PATH = PROJECT_ROOT / "imdb_splited" / "imdb_train.csv"
IMDB_TEST_PATH  = PROJECT_ROOT / "imdb_splited" / "imdb_test.csv"

RT_TRAIN_PATH = PROJECT_ROOT / "rt_splited" / "rt_train.csv"
RT_TEST_PATH  = PROJECT_ROOT / "rt_splited" / "rt_test.csv"

W2V_MODEL_PATH = PROJECT_ROOT / "w2v_model"