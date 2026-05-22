"""
test_20qa.py
------------
Run 20 questions against your running API and generate an accuracy report.

Usage:
  1. Start the app: uvicorn app.main:app --reload
  2. Upload your document via POST /upload
  3. Edit the TEST_CASES list below with your questions
  4. Run: python tests/test_20qa.py

Output: test_results.json + printed summary table
"""

import json
import time
import requests
from dataclasses import dataclass, asdict
from datetime import datetime

API_BASE = "http://localhost:8000"

# ─────────────────────────────────────────────────────────────────────────────
# EDIT THESE 20 QUESTIONS TO MATCH YOUR UPLOADED DOCUMENT
# 15 should be answerable, 5 should NOT be in the document
# ─────────────────────────────────────────────────────────────────────────────
TEST_CASES = [
    # ── ANSWERABLE (15) ──────────────────────────────────────────────────────
    {
        "q_id": "Q01",
        "question": "What is the main topic of the document?",
        "answerable": True,
        "expected_keywords": [],   # add keywords you expect in the answer
    },
    {
        "q_id": "Q02",
        "question": "Who are the key people or authors mentioned?",
        "answerable": True,
        "expected_keywords": [],
    },
    {
        "q_id": "Q03",
        "question": "What is the first major section about?",
        "answerable": True,
        "expected_keywords": [],
    },
    {
        "q_id": "Q04",
        "question": "What problem does this document try to solve?",
        "answerable": True,
        "expected_keywords": [],
    },
    {
        "q_id": "Q05",
        "question": "What are the key findings or conclusions?",
        "answerable": True,
        "expected_keywords": [],
    },
    {
        "q_id": "Q06",
        "question": "What methodology or approach is described?",
        "answerable": True,
        "expected_keywords": [],
    },
    {
        "q_id": "Q07",
        "question": "What data or evidence is presented?",
        "answerable": True,
        "expected_keywords": [],
    },
    {
        "q_id": "Q08",
        "question": "What are the limitations mentioned?",
        "answerable": True,
        "expected_keywords": [],
    },
    {
        "q_id": "Q09",
        "question": "What recommendations are made?",
        "answerable": True,
        "expected_keywords": [],
    },
    {
        "q_id": "Q10",
        "question": "What tools or technologies are referenced?",
        "answerable": True,
        "expected_keywords": [],
    },
    {
        "q_id": "Q11",
        "question": "What is the timeline or date range discussed?",
        "answerable": True,
        "expected_keywords": [],
    },
    {
        "q_id": "Q12",
        "question": "What organizations or institutions are mentioned?",
        "answerable": True,
        "expected_keywords": [],
    },
    {
        "q_id": "Q13",
        "question": "What are the main categories or types described?",
        "answerable": True,
        "expected_keywords": [],
    },
    {
        "q_id": "Q14",
        "question": "What examples are given in the document?",
        "answerable": True,
        "expected_keywords": [],
    },
    {
        "q_id": "Q15",
        "question": "What future work or next steps are suggested?",
        "answerable": True,
        "expected_keywords": [],
    },
    # ── NOT IN DOCUMENT (5) — model must say "I don't know" ─────────────────
    {
        "q_id": "Q16",
        "question": "What is the price of Bitcoin today?",
        "answerable": False,
        "expected_keywords": [],
    },
    {
        "q_id": "Q17",
        "question": "Who won the FIFA World Cup in 2022?",
        "answerable": False,
        "expected_keywords": [],
    },
    {
        "q_id": "Q18",
        "question": "What is the capital of Mars?",
        "answerable": False,
        "expected_keywords": [],
    },
    {
        "q_id": "Q19",
        "question": "What is the recipe for chocolate cake?",
        "answerable": False,
        "expected_keywords": [],
    },
    {
        "q_id": "Q20",
        "question": "How many stars are in the Andromeda galaxy?",
        "answerable": False,
        "expected_keywords": [],
    },
]


@dataclass
class TestResult:
    q_id: str
    question: str
    answerable: bool
    answer: str
    sources: list
    grounded: bool
    hallucination_detected: bool
    correct: bool
    hallucination: bool
    notes: str
    latency_ms: int


def evaluate(case: dict, api_result: dict) -> tuple[bool, bool, str]:
    """
    Returns (correct, hallucination, notes).

    Correct:
    - If answerable=True  → model should be grounded (not say "I don't know")
    - If answerable=False → model should say "I don't know..."

    Hallucination:
    - hallucination_detected=True from the API (invalid chunk_id cited)
    - OR answerable=False but model gave a confident grounded answer
    """
    answer_lower = api_result["answer"].lower()
    grounded = api_result["grounded"]
    hal_detected = api_result["hallucination_detected"]

    notes = []
    hallucination = False

    if case["answerable"]:
        correct = grounded  # should have answered from context
        if not correct:
            notes.append("Model said 'I don't know' but answer should exist")
        if hal_detected:
            hallucination = True
            notes.append("Invalid chunk_id cited")
    else:
        # unanswerable — model should NOT be grounded
        correct = not grounded
        if not correct:
            # model answered confidently for something not in docs → hallucination
            hallucination = True
            notes.append("Model hallucinated an answer for unanswerable question")
        if hal_detected:
            hallucination = True
            notes.append("Invalid chunk_id cited")

    return correct, hallucination, "; ".join(notes) if notes else "OK"


def run_tests():
    results: list[TestResult] = []
    print(f"\nRunning 20 Q&A tests against {API_BASE}\n{'='*60}")

    for case in TEST_CASES:
        print(f"[{case['q_id']}] {case['question'][:60]}...")

        start = time.time()
        try:
            resp = requests.post(
                f"{API_BASE}/ask",
                json={"question": case["question"], "top_k": 5},
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  ERROR: {e}")
            data = {
                "answer": f"REQUEST FAILED: {e}",
                "sources": [],
                "grounded": False,
                "hallucination_detected": False,
                "retrieved_chunks": 0,
            }

        latency = int((time.time() - start) * 1000)
        correct, hallucination, notes = evaluate(case, data)

        r = TestResult(
            q_id=case["q_id"],
            question=case["question"],
            answerable=case["answerable"],
            answer=data["answer"][:200],
            sources=data["sources"],
            grounded=data["grounded"],
            hallucination_detected=data["hallucination_detected"],
            correct=correct,
            hallucination=hallucination,
            notes=notes,
            latency_ms=latency,
        )
        results.append(r)

        status = "PASS" if correct else "FAIL"
        hal_flag = " [HALLUCINATION]" if hallucination else ""
        print(f"  {status}{hal_flag} — {notes} ({latency}ms)")
        time.sleep(0.5)  # gentle rate limiting

    # ── summary ───────────────────────────────────────────────────────────────
    total = len(results)
    correct_count = sum(1 for r in results if r.correct)
    hallucination_count = sum(1 for r in results if r.hallucination)

    accuracy = correct_count / total * 100
    hallucination_rate = hallucination_count / total * 100

    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total questions  : {total}")
    print(f"Correct          : {correct_count} / {total}")
    print(f"Accuracy         : {accuracy:.1f}%")
    print(f"Hallucinations   : {hallucination_count} / {total}")
    print(f"Hallucination %  : {hallucination_rate:.1f}%  (target: < 10%)")
    print(f"{'='*60}")

    target_met = "PASS" if hallucination_rate < 10 else "FAIL"
    print(f"Hallucination target (<10%): {target_met}")

    # ── save JSON log ─────────────────────────────────────────────────────────
    report = {
        "timestamp": datetime.now().isoformat(),
        "total": total,
        "correct": correct_count,
        "accuracy_pct": round(accuracy, 1),
        "hallucinations": hallucination_count,
        "hallucination_rate_pct": round(hallucination_rate, 1),
        "target_met": target_met == "PASS",
        "results": [asdict(r) for r in results],
    }

    out_path = "tests/test_results.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nFull log saved to: {out_path}")
    return report


if __name__ == "__main__":
    run_tests()