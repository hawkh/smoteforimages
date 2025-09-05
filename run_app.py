#!/usr/bin/env python3
"""
Simple startup script for SMOTE Image Synthesis Application
"""

import sys
import os
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def run_app():
    """Run the SMOTE image synthesis application."""
    
    print("=" * 60)
    print("     SMOTE IMAGE SYNTHESIS APPLICATION")
    print("=" * 60)
    print()
    
    print("This application demonstrates the SMOTE technique for generating")
    print("synthetic images to address class imbalance in image datasets.")
    print()
    
    print("Available options:")
    print("1. Run basic demo (small dataset, CPU)")
    print("2. Run full demo (larger dataset)")
    print("3. Run with custom settings")
    print("4. Show configuration options")
    print("5. Exit")
    print()
    
    while True:
        try:
            choice = input("Select an option (1-5): ").strip()
            
            if choice == "1":
                print("\nRunning basic demo...")
                print("This will create synthetic images using a small dataset on CPU.")
                print("Expected runtime: 2-5 minutes")
                print()
                
                cmd = [
                    sys.executable, "demo_pipeline.py", 
                    "--n-samples", "20",
                    "--cpu",
                    "--output-dir", "./basic_demo_output"
                ]
                
                print(f"Executing: {' '.join(cmd)}")
                print("-" * 50)
                
                try:
                    result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
                    print("✅ Demo completed successfully!")
                    print("📁 Results saved to: ./basic_demo_output")
                    print("📊 Check the output directory for generated files.")
                    if result.stdout:
                        print("\n📋 Demo output:")
                        print(result.stdout[-500:])  # Show last 500 chars
                    if result.stderr:
                        print("\n⚠️  Warnings:")
                        print(result.stderr[-200:])  # Show last 200 chars
                except subprocess.TimeoutExpired:
                    print("⏰ Demo is taking longer than expected...")
                    print("📁 Check ./basic_demo_output for partial results.")
                except subprocess.CalledProcessError as e:
                    print(f"❌ Demo failed with error code {e.returncode}")
                    print(f"Error: {e.stderr}")
                    if e.stdout:
                        print(f"Output: {e.stdout}")
                break
                
            elif choice == "2":
                print("\nRunning full demo...")
                print("This will create a larger synthetic dataset.")
                print("Expected runtime: 5-15 minutes")
                print()
                
                cmd = [
                    sys.executable, "demo_pipeline.py",
                    "--n-samples", "100", 
                    "--train-decoder",
                    "--generate-report",
                    "--output-dir", "./full_demo_output"
                ]
                
                print(f"Executing: {' '.join(cmd)}")
                print("-" * 50)
                
                try:
                    result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
                    print("✅ Full demo completed successfully!")
                    print("📁 Results saved to: ./full_demo_output")
                    print("📊 Check the output directory for generated files and reports.")
                    if result.stdout:
                        print("\n📋 Demo output:")
                        print(result.stdout[-500:])  # Show last 500 chars
                    if result.stderr:
                        print("\n⚠️  Warnings:")
                        print(result.stderr[-200:])  # Show last 200 chars
                except subprocess.TimeoutExpired:
                    print("⏰ Demo is taking longer than expected...")
                    print("📁 Check ./full_demo_output for partial results.")
                except subprocess.CalledProcessError as e:
                    print(f"❌ Demo failed with error code {e.returncode}")
                    print(f"Error: {e.stderr}")
                    if e.stdout:
                        print(f"Output: {e.stdout}")
                break
                
            elif choice == "3":
                print("\nCustom settings:")
                n_samples = input("Number of samples (default 50): ").strip() or "50"
                use_gpu = input("Use GPU if available? (y/n, default n): ").strip().lower() != 'y'
                output_dir = input("Output directory (default ./custom_output): ").strip() or "./custom_output"
                
                cmd = [sys.executable, "demo_pipeline.py", "--n-samples", n_samples, "--output-dir", output_dir]
                if use_gpu:
                    cmd.append("--cpu")
                
                print(f"\nExecuting: {' '.join(cmd)}")
                print("-" * 50)
                
                try:
                    result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
                    print("✅ Custom demo completed successfully!")
                    print(f"📁 Results saved to: {output_dir}")
                    print("📊 Check the output directory for generated files.")
                    if result.stdout:
                        print("\n📋 Demo output:")
                        print(result.stdout[-500:])  # Show last 500 chars
                    if result.stderr:
                        print("\n⚠️  Warnings:")
                        print(result.stderr[-200:])  # Show last 200 chars
                except subprocess.TimeoutExpired:
                    print(f"⏰ Demo is taking longer than expected...")
                    print(f"📁 Check {output_dir} for partial results.")
                except subprocess.CalledProcessError as e:
                    print(f"❌ Demo failed with error code {e.returncode}")
                    print(f"Error: {e.stderr}")
                    if e.stdout:
                        print(f"Output: {e.stdout}")
                break
                
            elif choice == "4":
                print("\nAvailable demo_pipeline.py options:")
                try:
                    result = subprocess.run([sys.executable, "demo_pipeline.py", "--help"], 
                                          capture_output=True, text=True, check=True, timeout=30)
                    print("\n📖 Available demo_pipeline.py options:")
                    print(result.stdout)
                except subprocess.TimeoutExpired:
                    print("⏰ Help command timed out")
                except subprocess.CalledProcessError as e:
                    print(f"❌ Error showing help: {e.stderr}")
                    
            elif choice == "5":
                print("Exiting...")
                break
                
            else:
                print("Invalid choice. Please select 1-5.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            break

if __name__ == "__main__":
    run_app()