# save_results.py
import json
from datetime import datetime

def save_to_json(results, success_count, total):
    output = {
        "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "total": total,
            "success": success_count,
            "failed": total - success_count,
            "success_rate": f"{(success_count / total) * 100:.1f}%"
        },
        "tickets": results
    }

    filename = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w") as f:
        json.dump(output, f, indent=4)

    print(f"💾 Results saved to: {filename}")
    return filename