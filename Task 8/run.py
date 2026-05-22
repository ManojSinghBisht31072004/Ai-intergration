# run.py
from db import init_db
from pipeline import run_pipeline
from tickets import TEST_TICKETS
from save_results import save_to_json 
import time # ← ADD THIS

def main():
    print("=" * 55)
    print("   DAY 8 — Multi-step AI Workflow")
    print("   Extract → Classify → Summarize → Store")
    print("=" * 55)

    init_db()
    results = []

    for i, ticket in enumerate(TEST_TICKETS, start=1):
        result = run_pipeline(ticket, ticket_num=i)
        results.append(result)
        time.sleep(3)

    print("\n" + "=" * 55)
    print("   TEST LOG")
    print("=" * 55)
    print(f"{'#':<5} {'STATUS':<10} DETAIL")
    print("-" * 55)

    success_count = 0
    for r in results:
        num    = r["ticket_num"]
        status = r["status"]
        if status == "success":
            detail = f"category={r.get('category', '?')}"
            success_count += 1
            flag = "✅"
        else:
            detail = f"error={r.get('error', '?')[:40]}"
            flag = "❌"
        print(f"{num:<5} {flag} {status:<10} {detail}")

    total = len(results)
    rate  = (success_count / total) * 100

    print("-" * 55)
    print(f"\n📊 RESULTS SUMMARY")
    print(f"   Total   : {total}")
    print(f"   Success : {success_count}")
    print(f"   Failed  : {total - success_count}")
    print(f"   Rate    : {rate:.1f}%  (target >90%)")

    if rate >= 90:
        print(f"\n🎉 PASSED — {rate:.1f}% meets the >90% target!")
    else:
        print(f"\n⚠️  BELOW TARGET — {rate:.1f}%. Check logs above.")

    print("\n✅ All results saved to tickets.db")
    save_to_json(results, success_count, total)   # ← ADD THIS

if __name__ == "__main__":
    main()