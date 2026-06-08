import os
import time
import json
import logging
from typing import List, Dict, Any, Tuple
import PyPDF2
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Structured Telemetry Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("Enterprise_RAG")

# Core Data Directories
DATA_DIR = "./uploaded_documents"
RESULTS_FILE = "./e2e_test_results.json"
os.makedirs(DATA_DIR, exist_ok=True)

class EnterpriseRAGSystem:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.chunks: List[str] = []
        self.vector_matrix = None
        self.is_indexed = False
        
    # ==========================================
    # 1. DYNAMIC DOCUMENT PROCESSING ENGINE
    # ==========================================
    def load_and_index_documents(self) -> int:
        """Scans the upload directory, parses files dynamically, and builds the vector store."""
        self.chunks = []
        supported_files = [f for f in os.listdir(DATA_DIR) if f.endswith(('.txt', '.pdf', '.docx'))]
        
        if not supported_files:
            self.is_indexed = False
            return 0
            
        for file_name in supported_files:
            file_path = os.path.join(DATA_DIR, file_name)
            try:
                if file_name.endswith('.txt'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        self._chunk_text(f.read())
                elif file_name.endswith('.pdf'):
                    with open(file_path, 'rb') as f:
                        reader = PyPDF2.PdfReader(f)
                        text = "".join([page.extract_text() or "" for page in reader.pages])
                        self._chunk_text(text)
                elif file_name.endswith('.docx'):
                    doc = Document(file_path)
                    text = "".join([p.text for p in doc.paragraphs])
                    self._chunk_text(text)
            except Exception as e:
                logger.error(f"Error parsing file {file_name}: {str(e)}")

        if self.chunks:
            self.vector_matrix = self.vectorizer.fit_transform(self.chunks)
            self.is_indexed = True
            
        return len(supported_files)

    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50):
        """Slices raw text down into cohesive chunk tokens for vector alignment."""
        words = text.split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                self.chunks.append(chunk)

    # ==========================================
    # 2. VECTOR RETRIEVAL ENGINE
    # ==========================================
    def retrieve_context(self, query: str, top_k: int = 2) -> List[str]:
        if not self.is_indexed or self.vector_matrix is None:
            return []
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.vector_matrix).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        return [self.chunks[idx] for idx in top_indices if similarities[idx] > 0.1]

    # ==========================================
    # 3. STRUCTURED MOCK INTERFERENCE ENGINE
    # ==========================================
    def execute_query_mock(self, query: str) -> Dict[str, Any]:
        """ Simulates Groq JSON Mode inference execution instantly for cost-free QA testing """
        start_time = time.time()
        retrieved_contexts = self.retrieve_context(query)
        context_str = " ".join(retrieved_contexts)
        
        # Build out a programmatic response schema matching our strict JSON generation requirement
        simulated_json_string = json.dumps({
            "answer": f"Simulated programmatic answer matching the query context guidelines.",
            "source_citations": retrieved_contexts,
            "confidence_score": 0.91 if retrieved_contexts else 0.42
        })
        
        latency = time.time() - start_time
        prompt_tokens = len(query.split()) + len(context_str.split()) + 100
        completion_tokens = len(simulated_json_string.split())
        
        # Cost Metrics Calculations
        cost = ((prompt_tokens / 1000) * 0.005) + ((completion_tokens / 1000) * 0.015)
        
        # Evaluate Hallucination / Groundedness Heuristics
        groundedness = 0.85 if retrieved_contexts else 0.10
        
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "query": query,
            "structured_response": json.loads(simulated_json_string),
            "telemetry": {
                "latency_seconds": round(latency, 3),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "calculated_cost_usd": round(cost, 6)
            },
            "eval_metrics": {
                "groundedness_score": groundedness,
                "hallucination_flag": groundedness < 0.40
            }
        }
        
        self._write_telemetry_to_disk(payload)
        return payload

    def _write_telemetry_to_disk(self, payload: Dict[str, Any]):
        results = []
        if os.path.exists(RESULTS_FILE):
            try:
                with open(RESULTS_FILE, 'r') as f:
                    results = json.load(f)
            except: pass
        results.append(payload)
        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=4)

# ==========================================
# 4. INTERACTIVE USER TERMINAL INTERFACE
# ==========================================
def main():
    rag = EnterpriseRAGSystem()
    print("====================================================")
    print("🚀 ENTERPRISE RAG PIPELINE CONSOLE — PHASE 2 ACTIVATED")
    print("====================================================")
    
    # Run initial index scan
    print("[*] Initializing automated background document ingest...")
    count = rag.load_and_index_documents()
    print(f"[✓] Data Synchronization complete. Indexed {count} source files from '{DATA_DIR}'.\n")

    while True:
        print("--- MAIN CONTROL INTERFACE ---")
        print("1) Re-scan and Refresh Upload Folder Documents")
        print("2) Execute Interactive Dynamic Q&A Loop (JSON Output Mode)")
        print("3) Exit Console Interface")
        choice = input("Select System Directive Menu Path [1-3]: ").strip()

        if choice == '1':
            print("\n[*] Re-indexing files...")
            count = rag.load_and_index_documents()
            print(f"[✓] Refreshed processing completed. {count} target files mapped to memory.\n")
            
        elif choice == '2':
            if not rag.is_indexed:
                print("\n⚠️  Warning: No source files are currently indexed. Please drop documents into the folder first.\n")
                continue
                
            print("\nEntering Interactive Q&A Mode. Type 'exit' to return to the main control layer.")
            while True:
                query = input("\nEnter User Context Query: ").strip()
                if query.lower() == 'exit' or not query:
                    break
                    
                print("[*] Streaming processing vectors... Computing JSON Payload response...")
                transaction_result = rag.execute_query_mock(query)
                
                # Print entire formatted system telemetry payload out to display
                print("\n=== SYSTEM RESPONSE STRING (JSON FORMAT) ===")
                print(json.dumps(transaction_result, indent=4))
                print("============================================")
                
        elif choice == '3':
            print("\nShutting down terminal process engine loop. Operational logs saved safely to disk.")
            break
        else:
            print("\n⚠️ Invalid directive. Select a standard menu operation index pathway.\n")

if __name__ == "__main__":
    main()