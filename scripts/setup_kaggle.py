"""
Kaggle CLI Setup Utility.
Configures ~/.kaggle/kaggle.json with user credentials and sets permissions.
"""
import os
import json
import sys
import subprocess
from pathlib import Path

def setup_kaggle():
    print("=" * 50)
    print("           KAGGLE CLI SETUP UTILITY")
    print("=" * 50)
    
    # 1. Get credentials from environment or prompt user
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    
    if not username or not key:
        print("\nTo connect with Kaggle, you need an API Token.")
        print("Go to: https://www.kaggle.com -> Your Profile -> Settings -> 'Create New Token'")
        print("This will download a 'kaggle.json' file containing your username and key.\n")
        
        username = input("Enter your Kaggle Username: ").strip()
        key = input("Enter your Kaggle API Key: ").strip()
        
    if not username or not key:
        print("[ERROR] Username or Key cannot be empty.")
        sys.exit(1)
        
    # 2. Setup ~/.kaggle directory
    home_dir = Path.home()
    kaggle_dir = home_dir / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    
    config_file = kaggle_dir / "kaggle.json"
    
    # 3. Write credentials
    credentials = {
        "username": username,
        "key": key
    }
    
    with open(config_file, "w") as f:
        json.dump(credentials, f, indent=2)
        
    print(f"\n[INFO] Created credential file: {config_file}")
    
    # 4. Set permissions (chmod 600)
    try:
        if os.name == 'posix':
            os.chmod(config_file, 0o600)
            print("[INFO] Set file permissions to Owner Read/Write (chmod 600).")
        else:
            # On Windows, standard file creation is usually sufficient
            pass
    except Exception as e:
        print(f"[WARNING] Failed to set permissions: {e}")
        
    # 5. Verify installation and connectivity
    print("\n[INFO] Verifying Kaggle API connectivity...")
    try:
        result = subprocess.run(
            ["kaggle", "competitions", "list"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("\n[SUCCESS] Connected to Kaggle API successfully!")
            print("You can now download trained models or datasets using the Kaggle CLI.")
        else:
            print(f"\n[ERROR] Connection failed. Stderr: {result.stderr}")
            print("Please check your credentials and try again.")
    except FileNotFoundError:
        print("\n[ERROR] 'kaggle' executable not found in PATH.")
        print("Try running: python3 -m pip install kaggle")

if __name__ == "__main__":
    setup_kaggle()
