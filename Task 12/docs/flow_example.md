# End-to-End Flow Example — Day 12

## Query: "If I have 1500 tokens and each chunk is 300 tokens, how many chunks?"

---

### Step 1 — Embed the Question
The question is tokenized and converted to a TF-IDF vector using the
LocalEmbedder (fitted on all ingested chunks).

### Step 2 — Vector Search
Cosine similarity is computed between the query vector and all chunk vectors.
Top-4 chunks are retrieved. In this case, chunks about "chunking strategy"
from the RAG document score highest (~0.4–0.6).

### Step 3 — Build Prompt
System prompt + retrieved chunks (with scores and source titles) + user
question are assembled into the messages array.

### Step 4 — LLM Round 1 (Groq)
Model receives context + question + tool schemas.
Model decides the question has a math component → outputs a tool_call:

```json
{
  "name": "calculate",
  "arguments": {"expression": "1500 / 300"}
}
```

### Step 5 — Tool Execution
`execute_tool("calculate", {"expression": "1500 / 300"})` runs locally.
Returns: `"Result of '1500 / 300' = 5.0"`

### Step 6 — Feed Result Back
Tool result appended as a `tool` role message. API called again (Round 2).

### Step 7 — LLM Round 2 — Final Answer
Model has context + tool result. No further tool calls needed.
Generates final answer:

> "Based on the retrieved documents, chunking splits documents into
> overlapping windows. For your specific case: 1500 ÷ 300 = **5 chunks**
> (assuming no overlap). With 50-token overlap, you would get slightly
> more chunks since each chunk advances fewer than 300 tokens."

---

## When Is a Tool Called vs Not?

| Scenario | Tool Used? | Reason |
|---|---|---|
| "What is RAG?" | ❌ No | Fully covered in retrieved chunks |
| "What is today's date?" | ✅ get_current_date | Not in any document |
| "Calculate 256 * 4" | ✅ calculate | Math operation |
| "Tell me about ML in detail" | ✅ fetch_additional_doc | Not in docs |
| "Latest vector DB news 2025" | ✅ search_web_summary | Recent/external info |
| "What chunk size is recommended?" | ❌ No | Specific answer in RAG doc |

---

## Key Decision Rule
The model reads the tool descriptions and decides:
- If context chunks fully answer → respond directly (RAG only)
- If date/time needed → `get_current_date`
- If math needed → `calculate`
- If topic missing from docs → `fetch_additional_doc`
- If recent/external info needed → `search_web_summary`