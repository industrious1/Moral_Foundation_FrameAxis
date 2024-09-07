import os
import torch
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from transformers import DistilBertTokenizer, DistilBertModel

class FrameAxis:
    def __init__(self, mfd=None, bert_model=None, bert_tokenizer=None, device=None):
        self.model = bert_model
        self.tokenizer = bert_tokenizer
        self.device = device
        current_dir_path = os.path.dirname(os.path.realpath(__file__))
        
        if mfd == "emfd":
            words_df = pd.read_csv(f'{current_dir_path}/moral_foundation_dictionaries/eMFD_wordlist.csv')
        elif mfd == "mfd":
            words_df = pd.read_csv(f'{current_dir_path}/moral_foundation_dictionaries/MFD_original.csv')
        elif mfd == "mfd2":
            words_df = self.read_mfd2_into_dataframe(current_dir_path)
        elif mfd == "customized":
            words_df = pd.read_csv(f'{current_dir_path}/moral_foundation_dictionaries/customized.csv')
        else:
            raise ValueError(f'Invalid mfd value: {mfd}')
        
        self.axes, categories = self._get_emfd_axes(words_df)
        print('axes names: ', categories)

    def get_bert_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

    def cos_sim(self, a, b):
        # 计算余弦相似度
        dot = np.dot(a, b)
        norma = np.linalg.norm(a)
        normb = np.linalg.norm(b)
        cos = dot / (norma * normb)
        return cos

    def _get_emfd_axes(self, eMFD):
        # 构建 eMFD 的 moral foundation 轴
        axes = {}
        mfs = set()
        mf_p = []
        for col in eMFD.columns:
            if col.endswith('_p'):
                mfs.add(col.split('_')[0])
                mf_p.append(col)

        centroids = {}
        for index, row in eMFD.iterrows():
            mf = pd.to_numeric(row[mf_p]).idxmax().split('_')[0]
            word_embedding = self.get_bert_embedding(row['word'])
            if row[mf + '_sent'] > 0:
                mf_virtue = mf + '.virtue'
                if mf_virtue not in centroids:
                    centroids[mf_virtue] = [word_embedding]
                else:
                    centroids[mf_virtue].append(word_embedding)
            else:
                mf_vice = mf + '.vice'
                if mf_vice not in centroids:
                    centroids[mf_vice] = [word_embedding]
                else:
                    centroids[mf_vice].append(word_embedding)

        for mf in mfs:
            mf_vice = mf + '.vice'
            mf_virtue = mf + '.virtue'
            centroids[mf_virtue] = np.mean(np.array(centroids[mf_virtue]), axis=0)
            centroids[mf_vice] = np.mean(np.array(centroids[mf_vice]), axis=0)
            axes[mf] = centroids[mf_virtue] - centroids[mf_vice]

        return axes, mfs

    # 其他方法保持不变...
