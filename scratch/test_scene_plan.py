import requests
import json

payload = {
    "topic": "Urban tree canopies and microclimate cooling in sustainable cities",
    "research_mode": "script_plus_research"
}

r = requests.post("http://localhost:8000/api/v1/scene-plan", json=payload)
print("Status:", r.status_code)
data = r.json()
print("Scenes count:", len(data.get("scenes", [])))
if data.get("scenes"):
    first = data["scenes"][0]
    print("\nScene 1 All 13 Fields:")
    for k, v in first.items():
        print(f"  {k}: {v}")
