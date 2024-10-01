#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
from typing import List, Tuple
from module.embeddings import BaseEmbeddings, OpenAIEmbedding, JinaEmbedding, ZhipuEmbedding
from tqdm import tqdm
import faiss

class VectorStore:
    def __init__(self, document: List[str] = [''], vectors: List[str] = ['']):
        self.document = document
        self.index = None
        self.vectors = vectors

    def get_vector(self, EmbeddingModel: BaseEmbeddings) -> List[List[float]]:
        self.vectors = []
        for doc in tqdm(self.document, desc="Calculating embeddings"):
            embedding = EmbeddingModel.get_embedding(text = doc)
            normalized_embedding = self.normalize_vector(embedding)
            # 归一化向量
            self.vectors.append(self.normalize_vector(embedding))

        return self.vectors

    def normalize_vector(self, vector: List[float]) -> List[float]:
        # 计算向量的L2范数并归一化
        norm = np.linalg.norm(vector)
        if norm > 0:
            return [float(v / norm) for v in vector]
        else:
            return [float(v) for v in vector]

    def build_index(self):
        if self.vectors:
            # 使用faiss.IndexFlatIP创建余弦相似度索引
            self.index = faiss.IndexFlatIP(len(self.vectors[0]))
            self.index.add(np.array(self.vectors).astype('float32'))

    def persist(self, path: str = 'storage'):
        if not os.path.exists(path):
            os.makedirs(path)
        with open(f"{path}/document.json", 'w', encoding='utf-8') as f:
            json.dump(self.document, f, indent=2, ensure_ascii=False)
        if self.index is not None:
            faiss.write_index(self.index, f"{path}/vectors.index")

    def load_index(self, path: str = 'storage'):
        if os.path.exists(f"{path}/vectors.index"):
            self.index = faiss.read_index(f"{path}/vectors.index")
        with open(f"{path}/document.json", 'r', encoding='utf-8') as f:
            self.document = json.load(f)
        if os.path.exists(f"{path}/Memory_Vectors.json"):
            with open(f"{path}/Memory_Vectors.json", 'r', encoding='utf-8') as f:
                vectors_list = json.load(f)
            self.vectors = [np.array(vector, dtype=np.float32) for vector in vectors_list]

    def get_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        # 余弦相似度已经通过内积计算，所以直接返回内积结果
        return np.dot(vector1, vector2)

    def query(self, query: str, EmbeddingModel: BaseEmbeddings, k: int = 2) -> List[str]:
        # query_vector = np.array([EmbeddingModel.get_embedding(query)]).astype('float32')
        query_vector = np.array([EmbeddingModel.get_embedding(query)]).astype('float32')
        # 归一化查询向量
        query_vector = self.normalize_vector(query_vector[0])
        query_vector = np.array([query_vector]).astype('float32')
        distances, indices = self.index.search(query_vector, k)
        distances = (distances + 1) / 2
        # 确保返回k个结果
        return [self.document[i] for i in indices[0] if i < len(self.document)]

    def query_with_vector(self, query: str, EmbeddingModel: BaseEmbeddings, k: int = 2) -> List[Tuple[str, np.ndarray]]:
        query_vector = np.array([EmbeddingModel.get_embedding(query)]).astype('float32')
        # 归一化查询向量
        query_vector = self.normalize_vector(query_vector[0])
        query_vector = np.array([query_vector]).astype('float32')
        distances, indices = self.index.search(query_vector, k)
        # 内积结果需要转换成余弦相似度
        distances = (distances + 1) / 2
        # 确保返回k个结果，并返回文档及其对应的向量
        results = []
        for i in indices[0]:
            if i < len(self.document):
                doc_content = self.document[i]
                doc_vector = self.vectors[i]
                results.append((doc_content, doc_vector))
        return results

    def set_index(self, index: faiss.Index):
        self.index = index

    def merge_index(self, temp_index: faiss.Index):
        if self.index is None:
            self.index = temp_index
        else:
            self.index.add(temp_index.reconstruct_n(0, temp_index.ntotal))
    
    def remove_ids(self, ids_to_remove):
        if self.index is not None:
            # 创建ID选择器对象
            selector = faiss.IDSelectorBatch(np.array(ids_to_remove))
            # 调用remove_ids方法移除向量
            self.index.remove_ids(selector)
        else:
            print("Index not built yet.")

    def save_merged_index(self, path: str = 'storage', index_name: str = 'Memory_Vectors.index', vector_name: str = 'Memory_Vectors.json'):
        if self.index is not None:
            # 确保存储路径存在
            if not os.path.exists(path):
                os.makedirs(path)
            # 将合并后的索引保存到磁盘
            faiss.write_index(self.index, f"{path}/{index_name}")
        if self.vectors:
            vectors_list = [list(map(float, vector)) if isinstance(vector, np.ndarray) else vector for vector in self.vectors]
            with open(f"{path}/{vector_name}", 'w', encoding='utf-8') as f:
                json.dump(vectors_list, f, indent=2, ensure_ascii=False)
        else:
            print("No index to save.")