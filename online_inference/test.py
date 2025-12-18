from config import config_mapping
from chat_utils import get_chat_result
import json

extract_column_prompt = """
You are a Table Column Selector for table question answering.

Goal: Select a MINIMALLY SUFFICIENT set of columns to answer the question using the table.
Minimally sufficient means:
- Sufficiency first: the selected columns must be enough to (A) identify the target row(s), (B) perform any needed comparison/ranking/sorting/filtering, and (C) extract the final answer value.
- Minimality second: you may remove a column ONLY if removing it would still keep (A)(B)(C) possible without introducing ambiguity or unverifiable assumptions.

Rules:
1) Decompose the question into: target selection criteria + required operation(s) + final answer field.
2) Select columns that support each part:
   - Row identification columns (filter keys / entity identifiers)
   - Operation columns (the metric used for max/second most/ranking, tie-breaking if needed)
   - Answer extraction column(s) (where the requested value is located)
3) Robustness constraints:
   - If the table has a "rank" column AND the question depends on ordering, include BOTH the rank and the metric column unless you are certain rank is computed from the same metric.
   - If ties are possible for the metric, include a tie-breaker column if available; otherwise note tie risk via answer_in_table=false if it affects determinism.
   - If the requested value is not explicitly represented by any column (e.g., asks for middle name but only "player" exists), still select columns to identify the correct player, and set answer_in_table=false.
4) Primary key retention: Always include at least one entity identifier (primary key-like) column in required_columns (e.g., name/player/id). This is mandatory when missing_info=true or need_external_knowledge=true, because downstream retrieval requires an entity anchor.
Provide the final answer in the following format:
{{
  "required_columns": ["<exact column name>", ...],
  "answer_in_table": true/false,
  "missing_info": true/false,
  "need_external_knowledge": true/false,
  "external_query_hint": "<brief retrieval hint if needed>",
  "reasoning": "<brief explanation of why these columns were selected>"
}}

Instance:
Question: {question}
Columns: {columns}
"""

Template_Generator_prompt = """
You are a “Table Row-to-Sentence Template Generator”.

Input:
Question (optional, only to adapt domain wording): {question}
Columns (a list of exact column names): {columns}

Goal:
Generate one reusable sentence template that can convert each table row into a retrieval-friendly natural-language sentence. The template will later be rendered by replacing placeholders with the row’s cell values.

Rules:
You must select one primary key column from Columns that best identifies the entity in a row (e.g., player/name/id). If uniqueness may be uncertain, still choose the best candidate and mention the risk in notes.

Placeholders:
You may ONLY reference the provided column names using placeholders in this exact form:
{{<exact column name>}}
The placeholder column name must match exactly, including spaces and parentheses. Do not invent any columns.

Template requirements:
The template must be a single sentence and must start with the primary key placeholder.
Prefer information-dense “Field: Value” phrasing for numeric fields (e.g., “Yards: {{Yards}}”).
If there is a rank column, write “rank {{rank}}” but do not assume what the rank is based on.

Output:
Return ONLY valid JSON, with no extra text, matching this exact schema:
{
"primary_key": "<one column name from Columns>",
"template": "<a single-sentence template with multiple placeholders>",
"notes": "<one short sentence about style and/or primary key uniqueness risk>"
}
"""

question = "What is the middle name of the player with the second most National Football League career rushing yards?"
columns = ["rank", "player", "Team ( s ) by season", "Carries", "Yards", "Average"]

input_text = [{"role": "user", "content": extract_column_prompt.format(question=question, columns=', '.join(columns))}]
backbone = "qwen2.5:32b"
select_config = config_mapping[backbone]
response = get_chat_result(messages=input_text, tools=None, llm_config=select_config)
response = json.loads(response.content)
print(response)
print(type(response))
