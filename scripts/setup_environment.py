#!/usr/bin/env python3
"""
Setup script for Diabetic Retinopathy Classification project
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, cwd=None):
    """Run a command and return the result"""
    try:
        result = subprocess.run(cmd, shell=True, check=True, cwd=cwd, 
                              capture_output=True, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def setup_backend():
    """Set up the backend environment"""
    print("🔧 Setting up backend environment...")

    backend_dir = Path("backend")
    if not backend_dir.exists():
        print("❌ Backend directory not found!")
        return False

    # Create virtual environment
    print("Creating virtual environment...")
    success, output = run_command("python -m venv venv", cwd=backend_dir)
    if not success:
        print(f"❌ Failed to create virtual environment: {output}")
        return False

    # Activate venv and install requirements
    if sys.platform == "win32":
        pip_cmd = "venv\\Scripts\\pip"
    else:
        pip_cmd = "venv/bin/pip"

    print("Installing Python dependencies...")
    success, output = run_command(f"{pip_cmd} install -r requirements.txt", cwd=backend_dir)
    if not success:
        print(f"❌ Failed to install dependencies: {output}")
        return False

    print("✅ Backend setup completed!")
    return True

def setup_frontend():
    """Set up the frontend environment"""
    print("🔧 Setting up frontend environment...")

    frontend_dir = Path("frontend")
    if not frontend_dir.exists():
        print("❌ Frontend directory not found!")
        return False

    # Install npm dependencies
    print("Installing Node.js dependencies...")
    success, output = run_command("npm install", cwd=frontend_dir)
    if not success:
        print(f"❌ Failed to install dependencies: {output}")
        return False

    print("✅ Frontend setup completed!")
    return True

def setup_docker():
    """Set up Docker environment"""
    print("🔧 Setting up Docker environment...")

    # Check if docker-compose exists
    success, _ = run_command("docker-compose --version")
    if not success:
        success, _ = run_command("docker compose version")
        if not success:
            print("❌ Docker Compose not found! Please install Docker Desktop.")
            return False

    print("Building Docker images...")
    success, output = run_command("docker-compose build", cwd="docker")
    if not success:
        print(f"❌ Failed to build Docker images: {output}")
        return False

    print("✅ Docker setup completed!")
    return True

def main():
    parser = argparse.ArgumentParser(description="Setup DR Classification project")
    parser.add_argument("--mode", choices=["manual", "docker"], default="docker",
                       help="Setup mode: manual (backend+frontend) or docker")
    parser.add_argument("--backend-only", action="store_true", 
                       help="Setup backend only (manual mode)")
    parser.add_argument("--frontend-only", action="store_true",
                       help="Setup frontend only (manual mode)")

    args = parser.parse_args()

    print("🚀 Starting Diabetic Retinopathy Classification setup...")

    if args.mode == "docker":
        if setup_docker():
            print("\n✅ Setup completed! Run 'docker-compose up' to start the application.")
        else:
            print("\n❌ Setup failed!")
            sys.exit(1)

    elif args.mode == "manual":
        success = True

        if not args.frontend_only:
            success &= setup_backend()

        if not args.backend_only:
            success &= setup_frontend()

        if success:
            print("\n✅ Setup completed!")
            print("Backend: Run 'uvicorn api.main:app --reload' in backend/")
            print("Frontend: Run 'npm start' in frontend/")
        else:
            print("\n❌ Setup failed!")
            sys.exit(1)

if __name__ == "__main__":
    main()
