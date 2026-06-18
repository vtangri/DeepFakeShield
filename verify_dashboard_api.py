import requests
import json

BASE_URL = "http://localhost:8000/api/v1"
LOGIN_URL = f"{BASE_URL}/auth/login"
REPORTS_URL = f"{BASE_URL}/reports/"

def verify_dashboard_data():
    # 1. Login
    print(f"Logging in to {LOGIN_URL}...")
    login_data = {"email": "test@example.com", "password": "password"}
    try:
        response = requests.post(LOGIN_URL, json=login_data)
        response.raise_for_status()
        token = response.json()["access_token"]
        print("Login successful.")
    except Exception as e:
        print(f"Login failed: {e}")
        return

    # 2. Fetch Reports
    headers = {"Authorization": f"Bearer {token}"}
    print(f"Fetching reports from {REPORTS_URL}...")
    try:
        response = requests.get(REPORTS_URL, headers=headers)
        response.raise_for_status()
        reports = response.json()
        print(f"Fetched {len(reports)} reports.")
    except Exception as e:
        print(f"Failed to fetch reports: {e}")
        return

    # 3. Simulate Frontend Logic
    if not reports:
        print("No reports found to verify.")
        return

    print("\n--- Verifying Data Structure ---")
    sample = reports[0]
    required_fields = ["id", "created_at", "overall_score", "media_type", "summary"]
    missing = [f for f in required_fields if f not in sample]
    if missing:
        print(f"❌ Missing fields in report object: {missing}")
    else:
        print("✅ All required fields present in report object.")
        print(f"Sample Report: ID={sample['id']}, Score={sample['overall_score']}, Type={sample['media_type']}")

    print("\n--- Simulating Dashboard Statistics ---")
    # Total Scans
    total_scans = len(reports)
    print(f"Total Scans (frontend 'statTotalScans'): {total_scans}")

    # Fake Detection (Threat Level)
    # Logic in app.js: const fakes = reports.filter(r => (r.overall_score || 0) > 0.7).length;
    # const fakePercent = Math.round((fakes / reports.length) * 100) || 0;
    fakes_count = len([r for r in reports if (r.get("overall_score") or 0) > 0.7])
    fake_percent = round((fakes_count / total_scans) * 100) if total_scans > 0 else 0
    print(f"Threat Level (frontend 'statFakePercent'): {fake_percent}%")

    # Average Confidence
    # Logic in app.js: const avgConfidence = Math.round(reports.reduce((acc, r) => acc + (r.overall_score || 0), 0) / reports.length * 100) || 0;
    total_score = sum([(r.get("overall_score") or 0) for r in reports])
    avg_confidence = round((total_score / total_scans) * 100) if total_scans > 0 else 0
    print(f"Avg Confidence (frontend 'statConfidence'): {avg_confidence}%")

    # Recent Alerts
    # Logic: const recentFakes = reports.filter(r => (r.overall_score || 0) > 0.5).slice(0, 5);
    recent_fakes = [r for r in reports if (r.get("overall_score") or 0) > 0.5][:5]
    print(f"Recent Alerts Count: {len(recent_fakes)}")

    print("\n✅ API Verification Complete. Data is ready for Dashboard.")

if __name__ == "__main__":
    verify_dashboard_data()
