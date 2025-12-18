import numpy as np
import pandas as pd
import json
import re
import torch
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import math

from chat_utils import get_chat_result
from config import config_mapping
from utils.tool_utils import Embedder


class TableRAGPipeline:
    """
    集成了：表格重构、BGE 向量检索、Schema Pruning (列筛选) 和 子表生成。
    """
    def __init__(self,
                 df: pd.DataFrame,
                 external_text_list: List[str],  # 核心改动：直接输入字符串列表
                 llm_backbone: str = "qwen2.5:32b",
                 llm_path: str = "./models/bge-m3"):  # 或者使用本地路径

        self.backbone = llm_backbone
        self.df = df
        self.raw_text_list = external_text_list
        # 1. 加载 LLM 配置
        self.llm_config = config_mapping.get(llm_backbone)
        if not self.llm_config:
            raise ValueError(f"Backbone {llm_backbone} not found in config_mapping")

        # 预处理：转字符串，填充空值
        self.df = self.df.astype(str).replace('nan', '')

        # 3. 加载 BGE Embedding 模型
        self.embedder = Embedder(llm_path)

        # 4. 内部状态存储
        self.documents = []  # 存储转化后的实体文档
        self.table_embeddings = None  # 表格行向量 (Tensor)
        self.text_embeddings = None  # 文本块向量
        self.template = ""  # 存储生成的通用模板
        self.pk_col = self.df.columns[0] # 默认第一列为主键

    def _clean_json_response(self, content: str) -> Dict:
        """Helper: 鲁棒的 JSON 提取器"""
        content = content.strip()
        match = re.search(r'(\{.*\})', content, re.DOTALL)
        json_str = match.group(1) if match else content
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            print(f"❌ JSON Parse Failed. Raw:\n{content}")
            return {}

    # =========================================================================
    # PHASE 1: 离线索引构建 (Offline Indexing)
    # =========================================================================

    def _generate_generic_template(self) -> Dict:
        """让 LLM 看表头，生成一个通用的、中立的行描述模板"""
        columns = self.df.columns.tolist()
        prompt = """
You are a Data-to-Text Template Generator.
Input Columns: {columns}

Goal: Create a python f-string template to convert a table row into a natural language sentence.
Rules:
1. **Neutrality**: Do NOT infer or hallucinate. Just describe the data structure.
2. **Completeness**: You MUST include placeholders for ALL columns provided.
3. **Format**: Use {{Column Name}} for placeholders.
4. **Primary Key**: Identify the main entity column to start the sentence.

Output JSON only:
{{
  "primary_key": "<best identifier column>",
  "template": "<sentence template>"
}}
"""
        formatted_prompt = prompt.format(columns=', '.join(columns))
        print(f"🤖 [LLM] Generating generic row template...")
        response = get_chat_result(
            messages=[{"role": "user", "content": formatted_prompt}],
            tools=None,
            llm_config=self.llm_config
        )
        return self._clean_json_response(response.content)

    def build_index(self):
        """核心流程：执行离线建库"""
        print("\n=== 🏗️ Phase 1: Building Offline Index ===")

        # 1. 生成模板
        template_info = self._generate_generic_template()
        self.template = template_info.get("template", "")
        self.pk_col = template_info.get("primary_key", self.df.columns[0])
        print(f"✅ Template: {self.template}")

        # 2. 行转文本 (Entity Documents)
        py_template = self.template.replace("{{", "{").replace("}}", "}")
        self.documents = []
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Rows to Docs"):
            row_dict = row.to_dict()
            try:
                text = py_template.format(**row_dict)
                self.documents.append({
                    "row_id": idx,
                    "text": text,
                    "entity": row_dict.get(self.pk_col, "Unknown")
                })
            except Exception:
                continue

        # 3. BGE 向量化 (Vectorization)
        print("⚡ Encoding with BGE...")
        table_texts = [d["text"] for d in self.documents]
        self.table_embeddings = torch.tensor(self.embedder.encode(table_texts))

        # 对外部文本列表进行向量化
        if self.raw_text_list and len(self.raw_text_list) > 0:
            print(f"⚡ Encoding {len(self.raw_text_list)} External Text Blocks...")
            self.text_embeddings = torch.tensor(self.embedder.encode(self.raw_text_list))
        else:
            print("⚠️ Warning: external_text_list is empty, text indexing skipped.")

    # =========================================================================
    # PHASE 2: 在线推理 (Online Inference)
    # =========================================================================

    def _get_top_k_indices(self, query: str, embeddings: torch.Tensor, top_k: int) -> List[int]:
        """统一检索核心：处理 Query 编码与相似度计算"""
        if embeddings is None: return []
        query_emb = torch.tensor(self.embedder.encode(query)).squeeze()
        # 计算点积相似度
        scores = torch.matmul(embeddings, query_emb)
        top_results = torch.topk(scores, k=min(top_k, embeddings.shape[0]))
        return top_results.indices.tolist()

    def _filter_columns(self, question: str) -> Dict[str, Any]:
        """让 LLM 根据问题筛选列，并判断是否需要表外知识"""
        all_cols = self.df.columns.tolist()
        prompt = """
    You are a Column Selector.
    Question: "{question}"
    Available Columns: {columns}

    Goal: Select columns strictly necessary to answer the question.
    Rules:
    1. Include the Entity Name column.
    2. Include columns for filtering conditions in the question.
    3. CRITICAL: Include the column containing the Answer value.
    4. If the column that should contain the answer is missing, or the question asks for details not usually in a table (like "why", "how", or specific historical descriptions), set "answer_in_table" to true.

    Output JSON only:
    {{
      "selected_columns": ["<col1>", "<col2>", ...],
      "answer_in_table": true/false,
      "reasoning": "<brief explanation>"
    }}
    """
        formatted_prompt = prompt.format(question=question, columns=', '.join(all_cols))
        print(f"🤖 [LLM] Filtering columns & Checking self-sufficiency...")

        response = get_chat_result(
            messages=[{"role": "user", "content": formatted_prompt}],
            tools=None,
            llm_config=self.llm_config
        )

        result = self._clean_json_response(response.content)

        # 校验选出的列是否真的在表中
        result["selected_columns"] = [c for c in result.get("selected_columns", []) if c in all_cols]
        if not result["selected_columns"]:
            result["selected_columns"] = all_cols

        print(f"🏷️ answer_in_table: {result['answer_in_table']}")
        print(f"💡 Reasoning: {result['reasoning']}")

        return result

    def _analyze_query_intent(self, question: str) -> Dict[str, bool]:
        """
        分析问题意图：是简单的查值，还是复杂的聚合/排序
        """
        signals = {
            "is_complex": False,
            "has_agg": any(w in question.lower() for w in ["how many", "sum", "average", "total", "percentage"]),
            "has_rank": any(w in question.lower() for w in ["most", "highest", "second", "rank", "top", "compare"])
        }
        if signals["has_agg"] or signals["has_rank"]:
            signals["is_complex"] = True
        return signals

    def _expand_context_radius(self, anchor_ids: List[int], intent: Dict[str, bool]) -> List[int]:
        """
        根据意图和分布情况，自适应分配上下文预算
        """
        final_ids = set(anchor_ids)

        # 1. 计算分布集中度 (索引的标准差)
        std_dist = np.std(anchor_ids) if len(anchor_ids) > 1 else 0

        # 2. 策略：查值型 (集中且简单) -> 局部邻域扩展 (上下各1行)
        if not intent["is_complex"] and std_dist < 5:
            print("🎯 Strategy: Compact Lookup (Expanding Local Neighborhood)")
            for rid in anchor_ids:
                if rid > 0: final_ids.add(rid - 1)
                if rid < len(self.df) - 1: final_ids.add(rid + 1)

        # 3. 策略：复杂型 (聚合/排序) -> 属性共享扩展
        else:
            print("📊 Strategy: Analytical (Expanding by Key Attributes)")
            # 找到锚点行中最重要的属性（比如同一个 Engine）
            for rid in anchor_ids:
                # 假设我们拉入与锚点行共享 'Current layout engine' 的所有行
                # 这能帮助 LLM 在比较时看到“同类”数据
                shared_val = self.df.iloc[rid].get('Current layout engine', '')
                if shared_val and shared_val != 'Unknown':
                    # 找到具有相同引擎的所有行索引
                    shared_rows = self.df[self.df['Current layout engine'] == shared_val].index.tolist()
                    final_ids.update(shared_rows)

        # 限制最大上下文预算，防止 Token 溢出 (比如最多 15 行)
        sorted_ids = sorted(list(final_ids))
        return sorted_ids[:15]

    # =========================================================================
    # PHASE 2: 文本侧精简 (Textual Pruning - KV Focused)
    # =========================================================================

    def _retrieve_and_prune_text(self, question: str, anchor_entities: List[str], retrieved_texts: List[str]) -> str:
        """
        2. 自动判定 KV 结构与句子结构
        3. 基于 BGE 相似度与实体锚定打分
        4. 动态保留前 50% 的高价值信息单元
        """

        all_units = []
        query_emb = self.embedder.encode(question, convert_to_tensor=True, normalize_embeddings=True)


        for text in retrieved_texts:
            # 自动判定 KV vs 纯文本结构
            is_kv = len(re.findall(r'[:：|]', text)) > len(text) / 50
            units = re.split(r'[\n;]', text) if is_kv else re.split(r'(?<=[。？！?.])\s+', text)
            units = [u.strip() for u in units if len(u.strip()) > 5]

            if not units: continue

            # 批量计算单元相似度
            unit_embs = self.embedder.encode(units, convert_to_tensor=True, normalize_embeddings=True)
            scores = torch.matmul(unit_embs, query_emb).cpu().numpy()

            for i, score in enumerate(scores):
                text_unit = units[i]
                # 实体锚定加分：如果提到了表格里的 Top-K 实体，增加权重
                entity_bonus = 0.2 if any(ent.lower() in text_unit.lower() for ent in anchor_entities) else 0.0
                all_units.append({"text": text_unit, "score": score + entity_bonus})

        # 动态比例裁剪：保留前 50%
        all_units.sort(key=lambda x: x["score"], reverse=True)
        keep_count = math.ceil(len(all_units) * 0.5)
        top_units = all_units[:keep_count]

        print(f"✂️  Text Pruned: {len(all_units)} units -> Kept {len(top_units)} (Top 50%)")
        return "\n".join([f"- {u['text']}" for u in top_units])


    # =========================================================================
    # PHASE 3: 最终融合推理 (Hybrid Inference)
    # =========================================================================
    def query(self, question: str, top_k_rows: int = 5) -> str:
        """
        推理入口：结合自适应子表与精简 KV 文本
        """
        print(f"\n=== 🚀 Hybrid Query: {question} ===")

        # 1. 意图分析与锚点检索
        intent = self._analyze_query_intent(question)
        anchor_ids = self._get_top_k_indices(question, self.table_embeddings, top_k=top_k_rows)
        anchor_entities = [self.df.iloc[rid][self.pk_col] for rid in anchor_ids]

        # 2. 自适应行半径扩展
        expanded_ids = self._expand_context_radius(anchor_ids, intent)

        # 3. 动态列精简
        col_info = self._filter_columns(question)
        print(f"🏷️  Ext Knowledge Required: {col_info.get('answer_in_table')}")

        # 4. 构建精简子表
        sub_table_md = self.df.loc[expanded_ids, col_info["selected_columns"]].to_markdown(index=False)

        # 5. 文本侧检索与 50% 精简
        pruned_text = ""
        if self.text_embeddings is not None:
            top_text_ids = self._get_top_k_indices(question, self.text_embeddings, top_k=3)
            retrieved_raw = [self.raw_text_list[i] for i in top_text_ids]
            print("self.raw_text_list",self.raw_text_list)
            print("retrieved_raw",retrieved_raw)

            pruned_text = self._retrieve_and_prune_text(question, anchor_entities, retrieved_raw)

        # 6. 生成
        final_prompt = f"""
    You are a factual reasoning assistant. Answer the question based on the two types of evidence provided below.

    ### 1. Structured Table Evidence (Key Rows & Columns)
    {sub_table_md}

    ### 2. Supporting Textual Evidence (Extracted Facts)
    {pruned_text}

    ### Task:
    - Combine the Table and Text to find the answer.
    - If the Table lacks a specific value (e.g., a version number), look for it in the Textual Evidence.
    - Question: {question}

    Answer:"""

        print("\n📝 [Final Prompt Context Preview]:")
        print(f"--- Table ---\n{sub_table_md}\n--- Text ---\n{pruned_text}\n")

        # 4. 生成答案
        response = get_chat_result(
            messages=[{"role": "user", "content": final_prompt}],
            llm_config=self.llm_config
        )
        return response.content
