#!/usr/bin/env python3
"""
DeepFakeShield Live VPS Verification Script
Tests all live API endpoints against https://deepshield.cloud/
"""

import sys
import time
import requests

BASE_URL = "https://deepshield.cloud/api/v1"
TEST_USER_EMAIL = f"test_vps_{int(time.time())}@example.com"
TEST_USER_PASS = "TestPassword123!"

print(f"🚀 Starting Live VPS API Verification against {BASE_URL}...")

session = requests.Session()
session.verify = False  # Disable SSL warnings if needed

# 1. Health Check
print("\n1. Testing Health Endpoint...")
try:
    r = session.get("https://deepshield.cloud/health", timeout=10)
    print(f"   Status Code: {r.status_code}")
    print(f"   Response: {r.json()}")
    assert r.status_code == 200, f"Health check failed with status {r.status_code}"
except Exception as e:
    print(f"   ⚠️ Health check failed: {e}")

# 2. Register & Login User
print("\n2. Testing Authentication (Register & Login)...")
try:
    reg_data = {
        "email": TEST_USER_EMAIL,
        "password": TEST_USER_PASS,
        "full_name": "VPS Test User"
    }
    r_reg = session.post(f"{BASE_URL}/auth/register", json=reg_data, timeout=10)
    print(f"   Registration Status: {r_reg.status_code}")
    
    r_login = session.post(f"{BASE_URL}/auth/login", json={"email": TEST_USER_EMAIL, "password": TEST_USER_PASS}, timeout=10)
    print(f"   Login Status: {r_login.status_code}")
    token_data = r_login.json()
    token = token_data.get("access_token")
    assert token, "Failed to receive access_token"
    print("   ✅ Auth token generated successfully!")
    session.headers.update({"Authorization": f"Bearer {token}"})
except Exception as e:
    print(f"   ❌ Auth test failed: {e}")
    sys.exit(1)

# 3. Media Upload
print("\n3. Testing Media Upload API...")
try:
    sample_file_path = "test_samples/sample_real_with_audio.mp4"
    with open(sample_file_path, "rb") as f:
        files = {"file": (sample_file_path, f, "video/mp4")}
        r_upload = session.post(f"{BASE_URL}/media/upload", files=files, timeout=30)
    print(f"   Upload Status: {r_upload.status_code}")
    upload_res = r_upload.json()
    media_id = upload_res.get("id")
    print(f"   Media ID: {media_id}")
    assert media_id, "Upload failed to return media ID"
    print("   ✅ Media upload successful!")
except Exception as e:
    print(f"   ❌ Media upload test failed: {e}")
    sys.exit(1)

# 4. Start Analysis Task
print("\n4. Testing Start Analysis API...")
try:
    r_start = session.post(f"{BASE_URL}/analysis/start", json={"media_id": media_id}, timeout=10)
    print(f"   Start Analysis Status: {r_start.status_code}")
    start_res = r_start.json()
    job_id = start_res.get("job_id")
    print(f"   Job ID: {job_id}")
    assert job_id, "Failed to start analysis job"
    print("   ✅ Analysis task started!")
except Exception as e:
    print(f"   ❌ Start analysis failed: {e}")
    sys.exit(1)

# 5. Poll Job Status
print("\n5. Polling Job Status until completion...")
completed = False
for i in range(60):
    time.sleep(2)
    r_status = session.get(f"{BASE_URL}/analysis/{job_id}/status", timeout=10)
    status_res = r_status.json()
    stage = status_res.get("stage", "unknown")
    status_val = status_res.get("status", "unknown")
    progress = status_res.get("progress", 0.0)
    print(f"   [Poll {i+1}] Status: {status_val} | Stage: {stage} | Progress: {progress*100:.0f}%")
    
    if status_val.lower() == "done" or stage.lower() == "done":
        completed = True
        break
    elif status_val.lower() == "failed":
        print(f"   ❌ Job failed: {status_res.get('error_message')}")
        break

if not completed:
    print("   ⚠️ Job did not complete within 60s")

# 6. Retrieve Result
print("\n6. Retrieving Analysis Result...")
try:
    r_result = session.get(f"{BASE_URL}/analysis/{job_id}/result", timeout=10)
    print(f"   Result Status: {r_result.status_code}")
    result_data = r_result.json()
    print(f"   Verdict: {result_data.get('label')}")
    print(f"   Overall Score: {result_data.get('overall_score')}")
    print(f"   Video Score: {result_data.get('video_score')}")
    print(f"   Audio Score: {result_data.get('audio_score')}")
    print("   ✅ Results retrieved successfully!")
except Exception as e:
    print(f"   ❌ Result retrieval failed: {e}")

# 7. Download PDF Report
print("\n7. Testing PDF Report Download API...")
try:
    r_pdf = session.get(f"{BASE_URL}/reports/{job_id}/report.pdf", timeout=15)
    print(f"   PDF Status: {r_pdf.status_code}")
    assert r_pdf.status_code == 200, "PDF report download failed"
    assert len(r_pdf.content) > 1000, "PDF report file size too small"
    print("   ✅ PDF report generated and downloaded successfully!")
except Exception as e:
    print(f"   ❌ PDF download test failed: {e}")

print("\n🎉 Live VPS API Verification Complete!")
