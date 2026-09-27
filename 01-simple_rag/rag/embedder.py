"""
离线流水线 Step 3：文本嵌入（Embedding）
使用本地 SentenceTransformer，无需 API Key，离线可用。
归一化后余弦相似度 = 点积，与 FAISS IndexFlatIP 配合使用。
"""
from sentence_transformers import SentenceTransformer


class Embedder:
    """
    文本向量化器（本地模型）
    model      : HuggingFace 模型名，默认中文 bge-small
    batch_size : 每批编码文本数
    """
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "BAAI/bge-small-zh-v1.5",
        batch_size: int = 32,
    ) -> None:
        self.batch_size: int = batch_size
        print(f"[Embedder] 加载本地模型：{model}")
        self._model: SentenceTransformer = SentenceTransformer(model)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量嵌入，返回 L2 归一化向量列表。"""
        all_embeddings: list[list[float]] = []
        # range(0, n, 32) 产生0, 32, 64...
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i: i + self.batch_size]

            # 调用SentenceTransformer编码这一批，返回形状 (len(batch), dim) 的 numpy 数组
            # astype("float32") : 转成单精度浮点数 （FAISS和大多数向量库都要求float32）
            vecs = self._model.encode(
                batch,
                normalize_embeddings=True, # 输出向量L2长度 = 1
            ).astype("float32")

            # extend把二维数组摊平追加，.tolist()把numpy数组转成Python原生嵌套list
            all_embeddings.extend(vecs.tolist())
        return all_embeddings

    """
    手动分批其实是多余的, encode() 自己就接受 batch_size show_progress_bar
    
    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vecs = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return vecs.astype("float32").tolist()

    """

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]