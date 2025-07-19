#!/usr/bin/env python3
"""
Code formatting script for Email Marketing Agent.

This script runs Black and isort to format all Python code in the project.
"""

import subprocess
from pathlib import Path


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"Running {description}...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(e.stderr)
        return False


def main():
    """Format all Python code in the project."""
    project_root = Path(__file__).parent

    # Change to project directory
    import os
    os.chdir(project_root)

    commands = [
        (["uv", "run", "isort", "app/", "--check-only", "--diff"], "isort check"),
        (["uv", "run", "black", "app/", "--check", "--diff"], "black check"),
        (["uv", "run", "isort", "app/"], "isort format"),
        (["uv", "run", "black", "app/"], "black format"),
    ]

    for cmd, description in commands:
        run_command(cmd, description)

if __name__ == "__main__":
    main()
