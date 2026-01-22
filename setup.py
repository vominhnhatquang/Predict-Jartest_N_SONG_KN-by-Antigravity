"""
Setup script to install Python dependencies
Run this after creating virtual environment
"""
import subprocess
import sys
import os


def create_venv():
    """Create virtual environment"""
    print("Creating virtual environment...")
    subprocess.run([sys.executable, "-m", "venv", "venv"])
    print("✅ Virtual environment created!")


def install_dependencies():
    """Install dependencies from requirements.txt"""
    print("\nInstalling dependencies...")
    
    # Determine pip executable based on OS
    if sys.platform == "win32":
        pip_path = os.path.join("venv", "Scripts", "pip.exe")
    else:
        pip_path = os.path.join("venv", "bin", "pip")
    
    # Upgrade pip
    subprocess.run([pip_path, "install", "--upgrade", "pip"])
    
    # Install requirements
    subprocess.run([pip_path, "install", "-r", "backend/requirements.txt"])
    
    print("✅ Dependencies installed!")


def display_next_steps():
    """Display next steps"""
    print("\n" + "="*80)
    print("SETUP COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("1. Activate virtual environment:")
    if sys.platform == "win32":
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    print("\n2. Train the model:")
    print("   jupyter notebook notebooks/03_model_training.ipynb")
    print("\n3. Start the backend server:")
    print("   cd backend && python app.py")
    print("\n4. Open frontend/index.html in your browser")
    print("="*80)


if __name__ == "__main__":
    print("="*80)
    print("AI MODEL WEB INTEGRATION - SETUP")
    print("="*80)
    
    try:
        create_venv()
        install_dependencies()
        display_next_steps()
    except Exception as e:
        print(f"\n❌ Error during setup: {str(e)}")
        print("Please check the error and try again.")
        sys.exit(1)
