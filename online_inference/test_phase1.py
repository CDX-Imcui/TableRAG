import json
import os
from table_pipeline import TableRAGPipeline
import pandas as pd

def main():
    # 1. 读取表格
    df = pd.read_excel("data/dev_excel/Mobile_browser_0.xlsx")

    # 2. 读取 JSON 并转化为字符串列表 List[str]
    with open("data/dev_doc/Mobile_browser_0.json", 'r') as f:
        json_data = json.load(f)
    # 将字典的值提取出来，形成一个 List[str]
    text_list = list(json_data.values())


    # 这里的 embedding_model_name 可以换成你本地 BGE 模型的路径，或者 HuggingFace Hub ID
    pipeline = TableRAGPipeline(
        df=df,
        external_text_list=text_list,
        llm_backbone="qwen2.5:7b",
        llm_path="./models/bge-m3"
    )

    pipeline.build_index()

    # 3. 在线问答 (Phase 2)
    # Case 1: 之前的 Android 版本问题
    q1 = "Of the free and open source software browsers, which is currently on stable version 10?"
    ans1 = pipeline.query(q1, top_k_rows=5)
    print(f"\n📝 Final Answer 1: {ans1}")

    print("-" * 50)

    # Case 2: 测试列筛选能力
    q2 = "What engine does the Blackberry Browser use?"
    ans2 = pipeline.query(q2, top_k_rows=3)
    print(f"\n📝 Final Answer 2: {ans2}")


if __name__ == "__main__":
    main()