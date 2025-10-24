#!/usr/bin/env python3
"""
Launcher for Chlorophyll-a Prediction API
"""

import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = ['fastapi', 'uvicorn', 'earthengine-api', 'catboost']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_dependencies():
    """Install missing dependencies"""
    print("Installing dependencies...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "-r", "chlorophyll_requirements.txt"
        ])
        return True
    except subprocess.CalledProcessError:
        return False

def check_model_files():
    """Check if required model files exist"""
    required_files = ["catboost_chlor_a.pkl", "scaler.pkl"]
    missing_files = []
    
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    return missing_files

def check_credentials():
    """Check if Google Earth Engine credentials exist"""
    credentials_path = "secrets/ee-ulikhasanah16-02d7612d178d.json"
    return Path(credentials_path).exists()

def main():
    print("🌊 Chlorophyll-a Prediction API Launcher")
    print("=" * 50)
    
    # Check dependencies
    missing_packages = check_dependencies()
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        print("Installing dependencies...")
        if not install_dependencies():
            print("❌ Failed to install dependencies")
            return
        print("✅ Dependencies installed successfully!")
    
    # Check model files
    missing_files = check_model_files()
    if missing_files:
        print(f"❌ Missing model files: {', '.join(missing_files)}")
        print("Please ensure the following files are in the current directory:")
        for file in missing_files:
            print(f"  - {file}")
        return
    
    # Check credentials
    if not check_credentials():
        print("❌ Google Earth Engine credentials not found")
        print("Please ensure 'secrets/ee-ulikhasanah16-02d7612d178d.json' exists")
        return
    
    # Check if API file exists
    if not Path("chlorophyll_api.py").exists():
        print("❌ chlorophyll_api.py not found")
        return
    
    print("✅ All checks passed!")
    print("🚀 Starting Chlorophyll-a Prediction API...")
    print("📱 API will be available at: http://localhost:8000")
    print("📖 Documentation at: http://localhost:8000/docs")
    print()
    print("Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Start the API
        subprocess.run([
            sys.executable, "-m", "uvicorn", "chlorophyll_api:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n👋 Shutting down API...")
    except Exception as e:
        print(f"❌ Error starting API: {e}")

if __name__ == "__main__":
    main()