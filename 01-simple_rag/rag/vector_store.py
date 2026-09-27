"""
离线流水线 Step 4：向量索引（Vector Store）
使用 FAISS 构建 ANN（近似最近邻）索引，替代暴力余弦相似度
支持持久化 save/load，实现增量更新
"""
import json
import pickle
from pathlib import Path
from typing import cast

import faiss
import numpy as np

from .chunker import Chunk


class VectorStore:
    """
    基于FAISS的向量存储与检索
    index_path: 索引文件保存目录
    """

    def __init__(self, index_path: str = "./index") -> None:
        self.index_path: Path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)

        self._index: faiss.Index | None = None # FAISS index对象
        self._chunks: list[Chunk] = [] # 与向量一一对应的文本块
        self._dim: int = 0


    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """将文本块及其向量写入索引"""

        # 把嵌套list转成Numpy的二维数组（矩阵）, vectors.shape = (num_chunks, dim) = (块数, 向量维度)
        vectors = np.array(embeddings, dtype="float32")
        # 向量维度：取第一个块向量的长度（len 返回 int，避开 numpy shape 推断出的 Any）
        dim = len(embeddings[0])

        if self._index is None: # 第一次写入
            self._dim = dim
            # 创建FAISS索引对象
            # IndexFlatIP: 内积（等价于 L2 归一化后的余弦相似度）
            self._index = faiss.IndexFlatIP(dim)
        elif dim != self._dim: # 之后每次写入校验维度
            raise ValueError(f"向量维度不一致: 期望 {self._dim}, 实际 {dim}")
        
        # L2 归一化，使内积 = 余弦相似度
        faiss.normalize_L2(vectors)

        self._index.add(vectors)
        self._chunks.extend(chunks)

        print(f"[VectorStore] 已写入 {len(chunks)} 个块，总计 {len(self._chunks)} 个块")

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[tuple[float, Chunk]]:
        """
        返回 top_k 个最相关块，格式：[(score, Chunk), ...]
        score 越大越相关（余弦相似度，范围 [-1, 1]）
        """

        if self._index is None or self._index.ntotal == 0:
            return []

        # 查询归一化
        vec = np.array([query_embedding], dtype="float32")
        faiss.normalize_L2(vec)

        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(vec, k)

        results: list[tuple[float, Chunk]] = []
        for score, idx in zip(
            cast(list[float], scores[0]), cast(list[int], indices[0])
        ):
            if idx >= 0:
                results.append((float(score), self._chunks[idx]))
        return results

    def save(self) -> None:
        """保存索引与文本块到磁盘"""
        if self._index is None:
            return

        faiss.write_index(self._index, str(self.index_path / "faiss.index"))
        with open(self.index_path / "chunks.pkl", "wb") as f:
            pickle.dump(self._chunks, f)
        with open(self.index_path / "meta.json", "w", encoding="utf-8") as f:
            json.dump({"dim": self._dim, "total": len(self._chunks)}, f)
        
        print(f"[VectorStore] 索引已保存至 {self.index_path}/")

    def load(self) -> bool:
        """从磁盘加载索引，返回是否成功"""
        index_file = self.index_path / "faiss.index"
        chunks_file = self.index_path / "chunks.pkl"

        if not index_file.exists() or not chunks_file.exists():
            print("[VectorStore] 索引文件不存在，无法加载")
            return False
        
        self._index = faiss.read_index(str(index_file))
        with open(chunks_file, "rb") as f:
            self._chunks = pickle.load(f)
        with open(self.index_path / "meta.json", "r", encoding="utf-8") as f:
            meta = cast(dict[str,int], json.load(f))
            self._dim = meta.get("dim", 0)

        print(f"[VectorStore] 加载成功，共 {len(self._chunks)} 个块")
        return True

    @property
    def total(self) -> int:
        return len(self._chunks)