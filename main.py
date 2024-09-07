import argparse
import pandas as pd
from frameAxis import FrameAxis
from transformers import DistilBertTokenizer, DistilBertModel
import torch

def parse_arguments():
    parser = argparse.ArgumentParser(description='Calculate Moral Foundation Scores Using FrameAxis Approach.')

    parser.add_argument('--input_file',
                        help='Path to the dataset .csv file containing input text documents in a column.')

    parser.add_argument('--dict_type', type=str, default='emfd',
                        help='Dictionary for calculating FrameAxis Scores. Possible values are: emfd, mfd, mfd2')

    parser.add_argument('--output_file',
                        help='The path for saving the MF scored output CSV file.')

    parser.add_argument('--docs_colname',
                        help='The column containing the text docs to score with moral foundations.')
    
    args = parser.parse_args()
    return args

print("Running FrameAxis Moral Foundations scores")
args = parse_arguments()

IN_PATH = args.input_file
DICT_TYPE = args.dict_type

if DICT_TYPE not in ["emfd", "mfd", "mfd2", "customized"]:
    raise ValueError(
        f'Invalid dictionary type received: {DICT_TYPE}, dict_type must be one of \"emfd\", \"mfd\", \"mfd2\", \"customized\"')

OUT_CSV_PATH = args.output_file
DOCS_COL = args.docs_colname

# 加载 DistilBERT 模型和 tokenizer
print("Loading DistilBERT model and tokenizer...")
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# 确保模型在 GPU 上运行（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("DistilBERT model loaded successfully.")

# 读取数据
try:
    data = pd.read_csv(IN_PATH, lineterminator='\n')
    print(f"Data loaded successfully. Data shape: {data.shape}")
except Exception as e:
    print(f"Error loading data: {e}")
    raise e

print("Columns in the loaded data:", data.columns)

try:
    # 创建 FrameAxis 实例，传入 BERT 模型和 tokenizer
    fa = FrameAxis(mfd=DICT_TYPE, bert_model=model, bert_tokenizer=tokenizer, device=device)
    print("FrameAxis instance created successfully.")
    
    # 计算道德基础分数
    mf_scores = fa.get_fa_scores(df=data, doc_colname=DOCS_COL, tfidf=False, format="virtue_vice",
                                 save_path=OUT_CSV_PATH)
    print("Moral Foundation scores calculated and saved successfully.")
except KeyError as ke:
    print(f"KeyError: {ke} - Check if the column name {DOCS_COL} exists in the data.")
except Exception as e:
    print(f"Error during FrameAxis processing: {e}")
    raise e
