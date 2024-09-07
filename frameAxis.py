import os
import torch
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import DistilBertTokenizer, DistilBertModel


class FrameAxis:
    def __init__(self, mfd=None, bert_model=None, bert_tokenizer=None, device=None):
        self.model = bert_model
        self.tokenizer = bert_tokenizer
        self.device = device
        current_dir_path = os.path.dirname(os.path.realpath(__file__))
        
        # 读取道德基础词典
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

    def read_mfd2_into_dataframe(self, current_dir_path):
        # 读取 MFD2 格式词典
        num_to_mf = {}
        mfs_df = []
        with open(f'{current_dir_path}/moral_foundation_dictionaries/mfd2.txt', 'r') as mfd2:
            reading_keys = False
            for line in mfd2:
                line = line.strip()
                if line == '%' and not reading_keys:
                    reading_keys = True
                    continue
                if line == '%' and reading_keys:
                    reading_keys = False
                    continue
                if reading_keys:
                    num, mf = line.split()
                    num_to_mf[num] = mf
                else:
                    mf_num = line.split()[-1]
                    mf = num_to_mf[mf_num]
                    phrase = '_'.join(line.split()[0:-1])
                    mfs_df.append({'word': phrase, 'category': mf.split('.')[0], 'sentiment': mf.split('.')[1]})
        return pd.DataFrame(mfs_df)

    def get_bert_embedding(self, text):
        # 使用 DistilBERT 获取文本嵌入
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

    def cos_sim(self, a, b):
    # 确保向量是 1D 的
        a = np.squeeze(a)  # 将 (1, 768) 转换为 (768,)
        b = np.squeeze(b)  # 将 (1, 768) 转换为 (768,)
        
        # 计算余弦相似度
        dot = np.dot(a, b)
        norma = np.linalg.norm(a)
        normb = np.linalg.norm(b)
        return dot / (norma * normb)


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

    def get_fa_scores(self, df, doc_colname, save_path=None, tfidf=False, format="virtue_vice"):
        # 处理输入的 DataFrame，获取每个文档的 BERT 嵌入
        df = df.reset_index(drop=True)
        docs = df[doc_colname]
        print(f'Preprocessing column {doc_colname}')
        
        # 获取每个文档的 BERT 嵌入
        embeddings = []
        for doc in docs:
            embedding = self.get_bert_embedding(doc)
            embeddings.append(embedding)
        
        # 初始化存储偏向性和强度分数的 DataFrame
        biases = pd.DataFrame()
        intensities = pd.DataFrame()
        
        # 遍历道德基础轴，计算每个文档的分数
        for mf in self.axes.keys():
            print(f"Processing moral foundation: {mf}")
            mf_bias_scores = []
            mf_intensity_scores = []
            
            for embedding in embeddings:
                # 计算偏向性和强度
                axis_vector = self.axes[mf]
                bias_score = self.cos_sim(embedding, axis_vector)
                intensity_score = abs(bias_score)
                
                mf_bias_scores.append(bias_score)
                mf_intensity_scores.append(intensity_score)
            
            biases[f'bias_{mf}'] = mf_bias_scores
            intensities[f'intensity_{mf}'] = mf_intensity_scores

        # 将结果合并到原始数据
        fa_scores = pd.concat([df, biases, intensities], axis=1)
        
        # 保存结果到 CSV 文件
        if save_path:
            fa_scores.to_csv(save_path, index=False)
            print(f"Moral Foundations FrameAxis scores saved to {save_path}")
        else:
            print("No output file specified, results not saved.")

        return fa_scores
