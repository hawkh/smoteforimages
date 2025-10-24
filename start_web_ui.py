#!/usr/bin/env python3
"""
Launcher script for SMOTE Image Synthesis Web UI
"""

import subprocess
import sys
import os
from pathlib import Path

def check_streamlit():
    """Check if streamlit is installed"""
    try:
        import streamlit
        return True
    except ImportError:
        return False

def install_streamlit():
    """Install streamlit if not present"""
    print("Installing Streamlit...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    print("🖼️ SMOTE Image Synthesis Web UI Launcher")
    print("=" * 50)
    
    # Check if streamlit is installed
    if not check_streamlit():
        print("Streamlit not found. Installing...")
        if not install_streamlit():
            print("❌ Failed to install Streamlit. Please install manually:")
            print("   pip install streamlit")
            return
        print("✅ Streamlit installed successfully!")
    
    # Check if web_ui.py exists
    web_ui_path = Path("web_ui.py")
    if not web_ui_path.exists():
        print("❌ web_ui.py not found in current directory")
        return
    
    print("🚀 Starting Web UI...")
    print("📱 The interface will open in your default browser")
    print("🔗 URL: http://localhost:8501")
    print()
    print("Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Start streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "web_ui.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Shutting down Web UI...")
    except Exception as e:
        print(f"❌ Error starting Web UI: {e}")

if __name__ == "__main__":
    main()