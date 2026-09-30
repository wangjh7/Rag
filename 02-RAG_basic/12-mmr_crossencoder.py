# =============================================================================
#  MMR(Maximal Marginal Relevance) 去冗余选择  &  Cross-Encoder 精排
# =============================================================================
#
#  ── 完整流程图（对应本文件代码执行顺序）──────────────────────────────────────
#
#       ┌─────────────┐     ┌──────────────────────────────────────────┐
#       │   Query     │     │              CORPUS（语料库）              │
#       │ "员工年假天数"│     │  doc1, doc2, ... doc9（员工手册片段）     │
#       └──────┬──────┘     └──────────────────┬───────────────────────┘
#              │                                 │
#              └──────────────┬──────────────────┘
#                             ▼
#              ┌──────────────────────────────────────────┐
#              │  Bi-Encoder（bge-small-zh-v1.5）          │
#              │  bi_encoder_encode()                     │
#              │    encode(query)  ──► query_vec          │
#              │    encode(corpus) ──► doc_vecs           │
#              │    normalize_embeddings=True（L2 归一化）  │
#              └──────────────────┬───────────────────────┘
#                                 │
#                    sim_to_q = doc_vecs @ query_vec
#                                 │
#              ┌──────────────────┴───────────────────────┐
#              │                                          │
#              ▼                                          ▼
#  ┌───────────────────────────┐          ┌───────────────────────────────┐
#  │  Part 1: MMR 去冗余选择    │          │  Part 2: Bi-Encoder → CE 精排  │
#  └─────────────┬─────────────┘          └───────────────┬───────────────┘
#                │                                        │
#                ▼                                        ▼
#  ┌─────────────────────────┐              ┌─────────────────────────────┐
#  │ 路径 A：纯相关性 Top-K   │              │ bi_encoder_retrieve()       │
#  │ argsort(sim_to_q)[:K]  │              │ 按 cos 相似度取 Top-K 候选   │
#  │ → 易选入语义重复文档     │              └──────────────┬──────────────┘
#  └─────────────┬───────────┘                             │
#                │                                          ▼
#                ▼                             ┌─────────────────────────────┐
#  ┌─────────────────────────┐                  │ cross_encoder_rerank()      │
#  │ 路径 B：MMR 贪心选择     │                  │ CrossEncoder（bge-reranker） │
#  │ mmr_select(top_k=K)    │                  │ pairs = [[q,d1],[q,d2],...] │
#  │                         │                  │ predict() → CE 分数排序       │
#  │  初始化 selected=[],    │                  └──────────────┬──────────────┘
#  │        candidates=全部   │                                 │
#  │  ┌─ 循环 K 次 ────────┐  │                                 ▼
#  │  │ 对每个候选 i:       │  │                  ┌─────────────────────────────┐
#  │  │  rel  = sim_to_q[i]│  │                  │ Top-N 精排结果（N < K）      │
#  │  │  red  = max sim(i,  │  │                  │ 精度高于 Bi-Encoder 粗召回   │
#  │  │         selected)  │  │                  └─────────────────────────────┘
#  │  │  score=λ·rel       │  │
#  │  │       -(1-λ)·red   │  │
#  │  │  选最高分 → selected│  │
#  │  └────────────────────┘  │
#  └─────────────┬───────────┘
#                │
#                ▼
#  ┌─────────────────────────┐
#  │ MMR Top-K 多样化结果     │
#  │ 兼顾相关性 + 主题分散    │
#  └─────────────────────────┘
#
#  ── MMR 公式 ────────────────────────────────────────────────────────────────
#
#  score = λ · sim(doc, query) - (1-λ) · max_sim(doc, selected_docs)
#          ↑ 相关性权重 λ=0.5        ↑ 与已选文档的最大相似度（冗余惩罚）
#
#  ── 两阶段检索对比 ───────────────────────────────────────────────────────────
#
#  ┌────────────────────┬──────────────────────┬──────────────────────────────┐
#  │      阶段          │       模型           │           特点               │
#  ├────────────────────┼──────────────────────┼──────────────────────────────┤
#  │ Bi-Encoder 粗召回  │ bge-small-zh-v1.5    │ 快，query/doc 独立编码，ANN  │
#  │ MMR 去冗余         │ 同上（向量点积）      │ 在粗召回结果上提升多样性      │
#  │ Cross-Encoder 精排 │ bge-reranker-base    │ 慢，query-doc 联合编码，更准  │
#  └────────────────────┴──────────────────────┴──────────────────────────────┘
#
#  典型 RAG 串联：Query → Bi-Encoder Top-K →（可选 MMR）→ Cross-Encoder Top-N → LLM
#
# =============================================================================

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

bi_encoder = SentenceTransformer("BAAI/bge-small-zh-v1.5")
cross_encoder = CrossEncoder("BAAI/bge-reranker-base")

CORPUS = [
    "公司成立于2010年，专注于人工智能领域的研发与应用。",
    "标准工作时间为周一至周五，每天9:00-18:00。",
    "公司每月15日发放工资，提供五险一金、带薪年假。",
    "员工应遵守职业道德，保护公司机密。",
    "年假：入职满1年可享受5天带薪年假，最多15天。",
    "本公司年假为15天，入职满1年方可享受。",
    "年假按工龄累计，最长不超过15天。",
    "报销应提交发票原件及审批单。",
    "员工需遵守考勤制度，迟到需补卡。",
]

TOP_K = 4   # Bi-Encoder 粗召回 / MMR 选取数量
TOP_N = 2   # Cross-Encoder 精排后保留数量


# ── Part 1: MMR 贪心选择 ──────────────────────────────────────────────────────
def mmr_select(
    query_vec: np.ndarray,      # 查询向量，形状 (dim,)，需已归一化
    doc_vecs: np.ndarray,       # 候选文档向量，形状 (n, dim)，每行一篇，需已归一化
    top_k: int,                 # 最终要选出的文档篇数
    lambda_mult: float = 0.5    # 相关性 vs 多样性的权衡旋钮：1.0=只看相关性，0.0=只看多样性
) -> list[int]:                 # 返回被选中文档的【下标】，按选中顺序排列（先选的更相关）
    """
    doc_vecs: shape (n, dim)，已归一化。
    向量归一化后余弦相似度 = 点积。

    ## 为什么需要它
    如果语料里有 5 篇内容几乎一样的文档, Top-5 可能全是同一个意思的不同表述, 信息量很低。
    MMR 的思路是一篇一篇地挑，每挑下一篇时同时考虑两件事：
    - 它和查询**越相关越好**
    - 它和已经选中的文档**越不像越好**（惩罚冗余）

    ## 手算小例子
    假设 λ=0.5, 有 3 篇文档, `sim_to_q = [0.9, 0.85, 0.5]`, 其中文档0 和文档1 内容几乎一样(相互相似度 0.95), 文档2 是不同的话题(与文档0 相似度 0.2)。

    **第 1 轮**, selected 为空，冗余度都是 0:
    ```
    文档0: 0.5×0.9  − 0.5×0 = 0.450   ← 最高
    文档1: 0.5×0.85 − 0.5×0 = 0.425
    文档2: 0.5×0.5  − 0.5×0 = 0.250
    ```
    选文档0。

    **第 2 轮**, selected=[0]:
    ```
    文档1: 0.5×0.85 − 0.5×0.95 = −0.050    （和文档0 太像，被重罚）
    文档2: 0.5×0.5  − 0.5×0.20 =  0.150    ← 最高
    ```
    选文档2。

    结果 `[0, 2]`。如果是普通 Top-2, 结果会是 [0, 1], 两篇几乎重复。
    MMR 把位置让给了不同话题的文档2。
    """

    # ---------- 准备：预先算好"每篇文档与查询"的相似度 ----------
    # (n, dim) @ (dim,) -> (n,)，sim_to_q[i] = 第 i 篇文档与查询的余弦相似度。
    # 因为向量已归一化，点积就是余弦相似度。
    # 这个值在整个选择过程中不会变化，所以提前算一次，避免循环里重复计算。
    sim_to_q = doc_vecs @ query_vec

    # selected：已选中文档的下标，按选中先后顺序存放。
    selected: list[int] = []

    # candidates：还没被选中的文档下标集合，初始时是 {0, 1, ..., n-1}。
    # 用 set 是因为后面要反复"删除某个元素"，set.remove 比 list.remove 快。
    candidates = set(range(len(doc_vecs)))

    # ---------- 外层循环：每一轮选出一篇文档 ----------
    # 两个条件必须同时满足才继续：
    #   1) 还没选够 top_k 篇
    #   2) 候选池还没空（防止 top_k 大于文档总数时死循环）
    while len(selected) < top_k and candidates:

        # best_idx：本轮得分最高的文档下标；best_score：对应得分。
        # best_score 初始设为极小值，保证任何真实得分都能把它替换掉。
        best_idx, best_score = None, -1e9

        # ---------- 内层循环：给每个候选文档打 MMR 分，找出最高的 ----------
        for i in candidates:

            # 冗余度 = 文档 i 与"已选文档"的最大相似度。
            #   doc_vecs[i] @ doc_vecs[j]：两个一维向量的点积，得到一个数（文档 i 与已选文档 j 的相似度）
            #   for j in selected：对每一篇已选文档都算一遍
            #   max(...)：取最大值 —— 只要和任意一篇已选文档很像，就视为冗余
            #   float(...)：把 np.float 转成 Python float
            # 第一轮时 selected 为空，没有可比较对象，冗余度记为 0.0，
            # 所以第一轮一定会选出和查询最相似的那篇。
            redundancy = max(float(doc_vecs[i] @ doc_vecs[j]) for j in selected) if selected else 0.0

            # MMR 核心公式：
            #   score = λ × 与查询的相似度 − (1−λ) × 与已选文档的最大相似度
            # 前一项奖励"相关"，后一项惩罚"重复"。
            # λ=1 时退化成普通 Top-K；λ=0 时只追求与已选文档不同。
            score = lambda_mult * sim_to_q[i] - (1 - lambda_mult) * redundancy

            # "打擂台"：遇到更高分就更新当前最优。
            if score > best_score:
                best_score, best_idx = score, i

        # ---------- 本轮结果：选中最优文档，并把它移出候选池 ----------
        # 下一轮计算冗余度时，这篇文档就会成为"已选文档"的一员，
        # 与它相似的文档会被扣分。
        selected.append(best_idx)
        candidates.remove(best_idx)

    # 返回下标列表；需要文档原文时用 corpus[i] 自行取出。
    return selected

def bi_encoder_encode(
    query: str,
    corpus: list[str],
    model: SentenceTransformer,
) -> tuple[np.ndarray, np.ndarray]:
    """Bi-Encoder 编码：返回 (query_vec, doc_vecs)，均已 L2 归一化。"""
    doc_vecs = model.encode(corpus, normalize_embeddings=True)
    query_vec = model.encode([query], normalize_embeddings=True)[0]
    return query_vec, doc_vecs

query = "员工年假天数"
query_vec, doc_vecs = bi_encoder_encode(query, CORPUS, bi_encoder)
sim_to_q = doc_vecs @ query_vec

naive_top_k = np.argsort(sim_to_q)[::-1][:TOP_K].tolist()
mmr_top_k = mmr_select(query_vec, doc_vecs, top_k=TOP_K, lambda_mult=0.5)

print(f"Query: {query}")
print(f"\n【对比：纯相关性 Top-{TOP_K}（易选入语义相近的重复内容）】")
for rank, idx in enumerate(naive_top_k, start=1):
    print(f"  Top-{rank} (cos={sim_to_q[idx]:.4f}): {CORPUS[idx]}")

print(f"\n【MMR Top-{TOP_K}（兼顾相关性与多样性）】")
for rank, idx in enumerate(mmr_top_k, start=1):
    print(f"  Top-{rank} (cos={sim_to_q[idx]:.4f}): {CORPUS[idx]}")

# ── Part 2: Bi-Encoder 召回 → Cross-Encoder 精排 ─────────────────────────────

def bi_encoder_retrieve(
    query: str,
    corpus: list[str],
    model: SentenceTransformer,
    top_k: int,
) -> list[tuple[str, float]]:
    """Bi-Encoder 向量检索：按余弦相似度取 Top-K。"""
    q_vec, doc_vecs = bi_encoder_encode(query, corpus, model)
    scores = doc_vecs @ q_vec
    ranked_idx = np.argsort(scores)[::-1][:top_k]
    return [(corpus[i], float(scores[i])) for i in ranked_idx]

def cross_encoder_rerank(
    query: str,
    candidates: list[str],
    reranker: CrossEncoder,
    top_n: int,
) -> list[tuple[str, float]]:
    """Cross-Encoder 精排：对 Top-K 候选重排取 Top-N。"""
    pairs = [[query, doc] for doc in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates,scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]

bi_results = bi_encoder_retrieve(query, CORPUS, bi_encoder, top_k=TOP_K)
print(f"\n【Step 1】Bi-Encoder 粗召回 Top-{TOP_K}")
for rank, (doc, score) in enumerate(bi_results, start=1):
    print(f"  Top-{rank} (cos={score:.4f}): {doc}")

candidate_docs = [doc for doc, _ in bi_results]
reranked = cross_encoder_rerank(query, candidate_docs, cross_encoder, top_n=TOP_N)
print(f"\n【Step 2】Cross-Encoder 精排 Top-{TOP_N}")
for rank, (doc, score) in enumerate(reranked, start=1):
    print(f"  Top-{rank} (ce={score:.4f}): {doc}")

# ── Part 3·Bi-Encoder 召回 → mmr去冗余 → Cross-Encoder 精排 ─────────────────────────────

def bi_encoder_retrieve_mmr(
    query: str,
    corpus: list[str],
    model: SentenceTransformer,
    top_k: int,
) -> list[tuple[str, float]]:
    """Bi-Encoder 向量检索：按余弦相似度取 Top-K。"""
    q_vec, doc_vecs = bi_encoder_encode(query, corpus, model)
    scores = doc_vecs @ q_vec
    ranked_idx = np.argsort(scores)[::-1][:top_k]
    return [(corpus[i], float(scores[i])) for i in ranked_idx]

def mmr_select(
    query_vec: np.ndarray,
    doc_vecs: np.ndarray,
    top_k: int,
    lambda_mult: float = 0.5,
) -> list[int]:
    """MMR 贪心选择：按余弦相似度取 Top-K。"""
    sim_to_q = doc_vecs @ query_vec
    selected: list[int] = []
    candidates = set(range(len(doc_vecs)))
    while len(selected) < top_k and candidates:
        best_idx, best_score = None, -1e9
        for i in candidates:
            redundancy = max(float(doc_vecs[i] @ doc_vecs[j]) for j in selected) if selected else 0.0
            score = lambda_mult * sim_to_q[i] - (1 - lambda_mult) * redundancy
            if score > best_score:
                best_score, best_idx = score, i
        selected.append(best_idx)
        candidates.remove(best_idx)
    return selected

def cross_encoder_rerank_mmr(
    query: str,
    candidates: list[str],
    reranker: CrossEncoder,
    top_n: int,
) -> list[tuple[str, float]]:
    """Cross-Encoder 精排：对 Top-K 候选重排取 Top-N。"""
    pairs = [[query, doc] for doc in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]

bi_results_mmr = bi_encoder_retrieve_mmr(query, CORPUS, bi_encoder, top_k=TOP_K)
mmr_selected = mmr_select(query_vec, doc_vecs, top_k=TOP_K, lambda_mult=0.5)
candidate_docs_mmr = [doc for doc, _ in bi_results_mmr]
reranked_mmr = cross_encoder_rerank_mmr(query, candidate_docs_mmr, cross_encoder, top_n=TOP_N)

print(f"\n【Step 1】Bi-Encoder 粗召回 Top-{TOP_K}")
for rank, (doc, score) in enumerate(bi_results_mmr, start=1):
    print(f"  Top-{rank} (cos={score:.4f}): {doc}")

print(f"\n【Step 2】MMR 去冗余 Top-{TOP_K}")
for rank, idx in enumerate(mmr_selected, start=1):
    print(f"  Top-{rank} (cos={sim_to_q[idx]:.4f}): {CORPUS[idx]}")

print(f"\n【Step 3】Cross-Encoder 精排 Top-{TOP_N}")
for rank, (doc, score) in enumerate(reranked_mmr, start=1):
    print(f"  Top-{rank} (ce={score:.4f}): {doc}")