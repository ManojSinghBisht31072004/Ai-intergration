import pytest
import os
import json
from rag_system import EnterpriseRAGSystem, DATA_DIR, RESULTS_FILE

@pytest.fixture
def initialized_rag():
    # Setup dummy data file for isolated test pipeline verification
    test_file_path = os.path.join(DATA_DIR, "test_sla.txt")
    with open(test_file_path, "w", encoding="utf-8") as f:
        f.write("The corporate critical data security response threshold SLA limit is fixed at 2 hours.")
    
    system = EnterpriseRAGSystem()
    system.load_and_index_documents()
    yield system
    
    # Teardown Cleanup
    if os.path.exists(test_file_path):
        os.remove(test_file_path)

def test_production_readiness_metrics(initialized_rag):
    query = "What is the critical response SLA threshold?"
    payload = initialized_rag.execute_query_mock(query)
    
    # Validation Asserts mapping back to core criteria metrics
    assert payload["telemetry"]["latency_seconds"] < 3.0, "Latency breach threshold"
    assert "structured_response" in payload, "Payload did not output structured response field"
    assert "answer" in payload["structured_response"], "JSON structure mapping missing direct response string"
    assert payload["eval_metrics"]["hallucination_flag"] is False, "Heuristic evaluation identified system hallucination hazard"
    assert os.path.exists(RESULTS_FILE), "Telemetry logging failed to output transaction payload record to disk"