import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Define run_folder relative to the project root
run_folder = os.path.join(PROJECT_ROOT, "run/") # Ensures it's always Harmonies/run/
run_archive_folder =  os.path.join(PROJECT_ROOT, "run_archive/")
