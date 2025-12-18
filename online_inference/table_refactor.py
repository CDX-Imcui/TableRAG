import pandas as pd
import json
from typing import List, Dict, Optional, Tuple


class TableRefactorer:
    """
    实现 Entity-Centric TableRAG 的第一阶段：
    1. 识别主键 (Primary Key)
    2. 将二维表格重构为以实体为中心的文档 (Entity Documents)
    """

    def __init__(self, df: pd.DataFrame, table_name: str):
        self.df = df
        self.table_name = table_name
        # 预处理：移除空列和全空行
        self.df.dropna(how='all', axis=0, inplace=True)
        self.df.dropna(how='all', axis=1, inplace=True)
        # 转换为字符串，防止序列化错误
        self.df = self.df.astype(str)

    def _detect_primary_key(self) -> str:
        """
        轻量级表结构识别：确定主键列。
        策略：
        1. 优先找包含 'name', 'id', 'title', 'model' 的列
        2. 必须满足 Uniqueness (唯一性) 较高
        3. 优先选 Text 类型而不是纯数字 (纯数字可能是索引，语义较弱)
        """
        candidates = []

        for col in self.df.columns:
            # 1. 检查唯一性
            unique_ratio = self.df[col].nunique() / len(self.df)

            # 2. 检查列名特征
            name_score = 0
            col_lower = col.lower()
            if 'name' in col_lower: name_score += 3
            if 'title' in col_lower: name_score += 3
            if 'model' in col_lower: name_score += 2
            if 'id' in col_lower: name_score += 1

            # 3. 检查内容长度 (主键通常不宜过长，也不宜过短)
            avg_len = self.df[col].apply(len).mean()
            len_score = 1 if 2 < avg_len < 50 else 0

            # 综合打分
            total_score = (unique_ratio * 10) + name_score + len_score
            candidates.append((col, total_score))

        # 按分数排序，取最高
        if candidates:
            best_col = sorted(candidates, key=lambda x: x[1], reverse=True)[0][0]
            print(f"[{self.table_name}] Detected Primary Key: {best_col}")
            return best_col
        else:
            # Fallback: 如果实在找不到，默认第一列
            return self.df.columns[0]

    def _row_to_natural_language(self, row: pd.Series, pk_col: str) -> str:
        """
        将一行转化为自然语言描述 (用于检索)。
        格式: "Entity [PK] has [Attribute] of [Value]..."
        """
        entity_name = row[pk_col]
        description_parts = []

        # 强调用途：这是关于哪个实体的描述
        description_parts.append(f"Entity: {entity_name}")

        for col, val in row.items():
            if col == pk_col:
                continue
            # 跳过空值或无效值
            if val.lower() in ['nan', 'none', '', 'null', '-']:
                continue

            # 构建陈述句，增加语义连接词，便于向量模型理解
            description_parts.append(f"The {col} is {val}")

        return ". ".join(description_parts) + "."

    def process(self) -> List[Dict]:
        """
        执行重构流程，返回文档列表
        """
        pk_col = self._detect_primary_key()
        documents = []

        for idx, row in self.df.iterrows():
            # 1. 构造语义文本 (用于向量检索)
            text_content = self._row_to_natural_language(row, pk_col)

            # 2. 构造结构化元数据 (用于后续精确过滤和SQL验证)
            # 我们把整行数据保留在 metadata 里，而不是像旧代码那样丢弃
            metadata = row.to_dict()
            metadata.update({
                "table_name": self.table_name,
                "row_id": idx,
                "is_primary_key": pk_col,
                "entity_name": row[pk_col]
            })

            doc = {
                "page_content": text_content,  # 这部分进向量库 embedding
                "metadata": metadata  # 这部分存库 payload
            }
            documents.append(doc)

        return documents