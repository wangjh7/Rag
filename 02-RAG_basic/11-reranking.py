# =============================================================================
#  Step-back 检索  &  重排序（Reranking）
# =============================================================================
#
#  ── Step-back（两次检索拼上下文）────────────────────────────────────────────
#
#  原理：对用户原始问题生成一个更抽象的"退一步"问题，两路检索后合并上下文，
#        让 LLM 同时拥有具体细节与背景知识。
#
#       用户问题 ──────────────────────────────► 具体检索结果 (concrete_ctx)
#            │                                          │
#            │ LLM 生成抽象问题                           │ merge()
#            ▼                                          │
#       抽象问题 ──────────────────────────────► 背景检索结果 (background_ctx)
#                                                       │
#                                                       ▼
#                                               final_prompt → LLM
#
#  ── 重排序（Reranking）───────────────────────────────────────────────────────
#
#  原理：向量检索（Bi-Encoder）为速度对 query-doc 独立编码，交互信息不足；
#        Cross-Encoder 把 query 与 doc 拼在一起打分，精度更高但慢，
#        故放在 Top-K 之后做小范围重排。
#
#       ┌──────────────────────────────────────────────────┐
#       │                  Bi-Encoder                      │
#       │  encode(query) ──► q_vec                         │  快，可 ANN
#       │  encode(doc)   ──► d_vec   score = q_vec · d_vec │  精度低于 CE
#       └──────────────────────────────────────────────────┘
#
#       ┌──────────────────────────────────────────────────┐
#       │                 Cross-Encoder                    │
#       │  encode([query, doc]) ──► score                  │  精度高
#       │  query 与 doc 深度交互（Self-Attention）           │  慢，小批量
#       └──────────────────────────────────────────────────┘
#
#  流程：
#       Query ──► Bi-Encoder 向量检索 ──► Top-K 候选
#                                              │
#                                   Cross-Encoder 重排序
#                                              │
#                                         Top-N 结果（N < K）
#
# =============================================================================

from datetime import datetime, timezone

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

# ── Part 1: Step-back 查询生成 ────────────────────────────────────────────────

def step_back_queries(user_question: str) -> tuple[str, str]:
    """
    第二步可用 LLM 生成 abstract_q；此处用占位演示结构。
    返回 (具体问题, 抽象退步问题)
    """
    abstract_q = f"与下列问题相关的背景原理与定义是什么？{user_question}"
    return user_question, abstract_q

question = "公司每月几号发工资？"
q1, q2 = step_back_queries(question)

print("【Step-back 查询】")
print(f"  具体问题：{q1}")
print(f"  抽象问题：{q2}")

# ── Part 2: Bi-Encoder 召回 → Cross-Encoder 重排 ─────────────────────────────
TOP_K = 4   # Bi-Encoder 粗召回数量
TOP_N = 2   # Cross-Encoder 精排后保留数量

print(" load bi-encoder and cross-encoder start time:")
time_start = datetime.now(timezone.utc)
bi_encoder = SentenceTransformer("BAAI/bge-small-zh-v1.5")
cross_encoder = CrossEncoder("BAAI/bge-reranker-base")
print(" load bi-encoder and cross-encoder end time:")
time_end = datetime.now(timezone.utc)
print(f" load bi-encoder and cross-encoder time: {time_end - time_start}")

def bi_encoder_retrieve(
    query: str,                  # 用户的查询文本，例如 "猫喜欢吃什么"
    corpus: list[str],           # 候选文档列表，例如 ["猫喜欢吃鱼", "今天下雨了", ...]
    model: SentenceTransformer,  # 已加载的句向量模型，负责把文本变成向量
    top_k: int,                  # 最终返回最相关的前 top_k 篇
) -> list[tuple[str, float]]:    # 返回 [(文档原文, 相似度分数), ...]，分数从高到低
    """Bi-Encoder 向量检索：query 与 doc 独立编码，按余弦相似度取 Top-K。"""

    # ---------- 第 1 步：把所有文档编码成向量 ----------
    # model.encode 接收字符串列表，对每个字符串输出一个长度为 d 的向量，
    # 并把它们按行叠成二维数组。
    # 假设 N 篇文档、向量维度 d，则 doc_vecs 形状为 (N, d)：
    #     第 i 行 = 第 i 篇文档的向量
    # normalize_embeddings=True：把每一行缩放到长度（L2 范数）为 1，方向不变。
    # 这是为了让第 3 步的"点积"直接等于"余弦相似度"。
    doc_vecs = model.encode(corpus, normalize_embeddings=True)

    # ---------- 第 2 步：把查询编码成向量 ----------
    # 用【同一个模型】编码，这样 query 和 doc 才在同一个向量空间里，才有可比性。
    # 注意 query 和 doc 是各自独立编码的，模型编码 query 时看不到任何文档，
    # 这正是 "Bi-Encoder（双塔）" 名字的由来。
    #
    # 为什么写 [query]：encode 期望接收"一批文本"（列表）。
    #   传 [query]  -> 返回形状 (1, d) 的二维数组（1 行 d 列）
    # 末尾的 [0]：取第 0 行，把 (1, d) 变成 (d,) 的一维数组，
    #   这样第 3 步的矩阵乘法形状才能直接对上。
    q_vec = model.encode([query], normalize_embeddings=True)[0]

    # ---------- 第 3 步：计算每篇文档与查询的相似度 ----------
    # @ 是矩阵乘法。形状变化： (N, d) @ (d,) -> (N,)
    # 含义：doc_vecs 的每一行都与 q_vec 做点积（对应位置相乘再求和），
    #       所以 scores[i] 就是"第 i 篇文档与查询的相似度"。
    #
    # 为什么点积就是余弦相似度：
    #   余弦相似度 = 点积 / (|A| * |B|)
    #   两个向量的长度都已经是 1，分母 = 1，于是余弦相似度 = 点积。
    # 分数范围 [-1, 1]，越接近 1 表示语义越相近。
    scores = doc_vecs @ q_vec

    # ---------- 第 4 步：取分数最高的 top_k 个下标 ----------
    # 以 scores = [0.30, 0.90, 0.10, 0.60]，top_k = 2 为例：
    #
    # np.argsort(scores)：返回"按分数从小到大排序后的【下标】"，而不是分数本身
    #                     -> [2, 0, 3, 1]
    # [::-1]            ：整个列表反转，变成从大到小
    #                     -> [1, 3, 0, 2]
    # [:top_k]          ：只保留前 top_k 个
    #                     -> [1, 3]
    #
    # 要下标而不是分数值，是因为下一步要用下标回到 corpus 中取文档原文。
    # 若 top_k 大于文档总数，切片不会报错，只会返回全部文档。
    ranked_idx = np.argsort(scores)[::-1][:top_k]

    # ---------- 第 5 步：组装返回结果 ----------
    # 列表推导式：遍历每个下标 i，生成 (文档原文, 分数) 二元组。
    # 因为 ranked_idx 已按分数从高到低排列，所以结果也是从高到低。
    # float(...)：把 numpy 的 np.float32 转成 Python 原生 float，
    #             方便后续打印、序列化成 JSON 等（np.float32 无法直接被 json 序列化）。
    return [(corpus[i], float(scores[i])) for i in ranked_idx]


def cross_encoder_rerank(
    query: str,                 # 用户的查询文本，例如 "猫喜欢吃什么"
    candidates: list[str],      # 候选文档文本列表，通常是 bi_encoder_retrieve 召回的 Top-K（只要文本，不要分数）
    reranker: CrossEncoder,     # 已加载的交叉编码器模型，负责给 (query, doc) 对直接打相关性分数
    top_n: int,                 # 精排后最终保留的篇数，通常远小于 len(candidates)
) -> list[tuple[str, float]]:   # 返回 [(文档原文, 相关性分数), ...]，分数从高到低
    """Cross-Encoder 精排：query 与 doc 联合编码，对 Top-K 候选重排取 Top-N。"""

    # ---------- 第 1 步：构造 (query, doc) 对 ----------
    # Cross-Encoder 的输入不是单段文本，而是"一对文本"，模型会把两段拼在一起处理。
    # 所以要为每个候选文档配上同一个 query：
    #   query = "猫喜欢吃什么"，candidates = ["猫喜欢吃鱼", "今天下雨了"]
    #   -> pairs = [["猫喜欢吃什么", "猫喜欢吃鱼"],
    #               ["猫喜欢吃什么", "今天下雨了"]]
    # 长度与 candidates 相同，pairs[i] 对应 candidates[i]。
    pairs = [[query, doc] for doc in candidates]

    # ---------- 第 2 步：给每一对打分 ----------
    # 与 Bi-Encoder 的本质区别：query 和 doc 被拼接后【一起】送进模型，
    # 模型内部的注意力机制可以让两段文字的每个词互相对照（token 级交互），
    # 因此对"是否真正相关"的判断比只比较两个向量更精细。
    # 输出是每一对一个分数，返回 numpy 数组，形状 (len(candidates),)，
    # 顺序与 pairs 一一对应，例如 [8.2, -3.1]。
    # 注意：这个分数不是余弦相似度，范围取决于模型（可能是任意实数，也可能在 0~1），
    # 只适合在【同一个 query 的候选之间】比较大小，不同 query 之间不可比。
    scores = reranker.predict(pairs)

    # ---------- 第 3 步：配对并按分数降序排序 ----------
    # zip(candidates, scores)：把文档和分数像拉链一样一一配对，
    #   -> ("猫喜欢吃鱼", 8.2), ("今天下雨了", -3.1), ...
    # key=lambda x: x[1]：告诉 sorted 按每个元组的第 2 项（也就是分数）来排序。
    #   lambda x: x[1] 是一个匿名小函数，等价于 def f(x): return x[1]
    # reverse=True：sorted 默认升序，加上后变成降序（分数高的在前）。
    # 结果 ranked 是列表，元素是 (文档原文, 分数) 二元组。
    # 这里不需要像上一个函数那样用 argsort 取下标，
    # 因为候选数量少（几十篇），直接对配好对的数据排序更直观。
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

    # ---------- 第 4 步：截取前 top_n 个 ----------
    # 列表切片，取排名最靠前的 top_n 个。
    # 若 top_n 大于候选总数，切片不会报错，只会返回全部。
    # 注意：这里返回的分数仍是 np.float32，而不是签名里标注的 float；
    # 如果要序列化成 JSON，需要改成：
    #   return [(doc, float(s)) for doc, s in ranked[:top_n]]
    return ranked[:top_n]


corpus: list[str] = [
    "公司成立于2010年，专注于人工智能领域的研发与应用。",
    "公司每月15日发放工资，提供五险一金、带薪年假。",
    "标准工作时间为周一至周五，每天9:00-18:00。",
    "年假：入职满1年可享受5天带薪年假，最多15天。",
    "员工应遵守职业道德，保护公司机密。",
    "报销应提交发票原件及审批单。",
    "年假按工龄累计，最长不超过15天。",
    "员工需遵守考勤制度，迟到需补卡。",
    "公司每月15日发放工资，提供五险一金、带薪年假。",
    "标准工作时间为周一至周五，每天9:00-18:00。",
    "年假：入职满1年可享受5天带薪年假，最多15天。",
    "员工应遵守职业道德，保护公司机密。",
    "报销应提交发票原件及审批单。",
    "年假按工龄累计，最长不超过15天。",
    "员工需遵守考勤制度，迟到需补卡。",
]

query = "发工资"

# Step 1: Bi-Encoder 粗召回
print(" bi-encoder start time:")
time_start = datetime.now(timezone.utc)
bi_results = bi_encoder_retrieve(query, corpus, bi_encoder, top_k=TOP_K)
print(" bi-encoder end time:")
time_end = datetime.now(timezone.utc)
print(f" bi-encoder time: {time_end - time_start}")
print(f"\n【Step 1】Bi-Encoder 粗召回 Top-{TOP_K}")
for rank, (doc, score) in enumerate(bi_results, start=1):
    print(f"  Top-{rank} (cos={score:.4f}): {doc}")

# Step 2: Cross-Encoder 精排
print(" cross-encoder start time:")
time_start = datetime.now(timezone.utc)
candidate_docs = [doc for doc, _ in bi_results]
reranked = cross_encoder_rerank(query, candidate_docs, cross_encoder, top_n=TOP_N)
print(" cross-encoder end time:")
time_end = datetime.now(timezone.utc)
print(f" cross-encoder time: {time_end - time_start}")
print(f"\n【Step 2】Cross-Encoder 精排 Top-{TOP_N}")
for rank, (doc, score) in enumerate(reranked, start=1):
    print(f"  Top-{rank} (ce={score:.4f}): {doc}")