#!/usr/bin/env python3
"""
Launcher script for the Honeywell Anomaly Detection Dashboard
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit dashboard"""
    print("🚀 Launching Honeywell Anomaly Detection Dashboard...")
    print("📊 This will open a web browser with the interactive dashboard")
    print("💡 Make sure you have generated scored data with main.py first!")
    print()
    
    try:
        # Check if streamlit is installed
        import streamlit
        print("✅ Streamlit is available")
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "streamlit", "plotly"])
        print("✅ Dependencies installed")
    
    # Launch the dashboard
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard.py")
    
    print(f"🌐 Starting dashboard at: {dashboard_path}")
    print("🔗 The dashboard will open in your default web browser")
    print("⏹️  Press Ctrl+C to stop the dashboard")
    print()
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", dashboard_path,
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped. Goodbye!")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        print("💡 Try running: streamlit run dashboard.py")

if __name__ == "__main__":
    main()


