"""
Smoke test — sends a sample case to the running server.

Usage:
    1. Start the server:  python main.py
    2. In another terminal: python scripts/test_api.py
"""

import requests
import json

payload = {
    "patient_age": 14,
    "gender": "female",
    "case_type": "pocso",
    "symptoms": "bruising trauma brought by neighbor",
}

print("Sending request to http://localhost:8000/api/analyze ...")
res = requests.post("http://localhost:8000/api/analyze", json=payload)

print(f"Status: {res.status_code}")

if res.status_code == 200:
    data = res.json()
    print(f"\n✅ Session ID : {data['session_id']}")
    print(f"   Is minor   : {data['is_minor']}")
    print(f"   Laws       : {data['laws_retrieved']}")
    print("\n   Checklist:")
    for key, items in data["checklist"].items():
        print(f"\n   {key}:")
        for item in items:
            print(f"     • {item}")
elif res.status_code == 429:
    print(f"\n⚠️  Rate limited: {res.json()['detail']}")
else:
    print(f"\n❌ Error: {res.text}")
