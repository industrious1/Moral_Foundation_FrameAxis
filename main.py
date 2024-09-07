import argparse
import pandas as pd
from gensim.models import KeyedVectors
from frameAxis import FrameAxis

def parse_arguments():
    parser = argparse.ArgumentParser(description='Calculate Moral Foundation Scores Using FrameAxis Approach.')

    parser.add_argument('--input_file',
                        help='Path to the dataset .csv file containing input text documents in a column.')

    parser.add_argument('--dict_type', type=str, default='mfd',
                        help='Dictionary for calculating FrameAxis Scores. Possible values are: emfd, mfd, mfd2')

    parser.add_argument('--word_embedding_model',
                        help='Path to the word embedding model used to map words to a vector space.')

    parser.add_argument('--output_file',
                        help='The path for saving the MF scored output CSV file.')

    parser.add_argument('--docs_colname',
                        help='The column containing the text docs to score with moral foundations.')
    args = parser.parse_args()
    return args

print("Running FrameAxis Moral Foundations scores")
args = parse_arguments()

# 打印解析的参数
print("Parsed arguments:", args)

IN_PATH = args.input_file
DICT_TYPE = args.dict_type

# 打印输入文件路径和字典类型
print(f"Input file path: {IN_PATH}, Dictionary type: {DICT_TYPE}")

if DICT_TYPE not in ["emfd", "mfd", "mfd2", "customized"]:
    raise ValueError(
        f'Invalid dictionary type received: {DICT_TYPE}, dict_type must be one of \"emfd\", \"mfd\", \"mfd2\", \"customized\"')

OUT_CSV_PATH = args.output_file
DOCS_COL = args.docs_colname

# 打印输出文件路径和文档列名
print(f"Output file path: {OUT_CSV_PATH}, Document column name: {DOCS_COL}")

if args.word_embedding_model is not None:
    W2V_PATH = args.word_embedding_model
    print(f"Loading word embedding model from {W2V_PATH}")
    model = KeyedVectors.load_word2vec_format(W2V_PATH, binary=False)
else:
    print('Downloading word embedding model: glove-twitter-200')
    import gensim.downloader
    model = gensim.downloader.load('glove-twitter-200')

print("Word embedding model loaded successfully.")

try:
    data = pd.read_csv(IN_PATH, lineterminator='\n')
    print(f"Data loaded successfully. Data shape: {data.shape}")
except Exception as e:
    print(f"Error loading data: {e}")
    raise e

# 打印加载的数据的列名
print("Columns in the loaded data:", data.columns)

try:
    fa = FrameAxis(mfd=DICT_TYPE, w2v_model=model)
    print("FrameAxis instance created successfully.")
    
    mf_scores = fa.get_fa_scores(df=data, doc_colname=DOCS_COL, tfidf=False, format="virtue_vice",
                                 save_path=OUT_CSV_PATH)
    print("Moral Foundation scores calculated and saved successfully.")
except KeyError as ke:
    print(f"KeyError: {ke} - Check if the column name {DOCS_COL} exists in the data.")
except Exception as e:
    print(f"Error during FrameAxis processing: {e}")
    raise e
