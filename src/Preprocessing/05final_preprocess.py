import subprocess
import os

def run_script(script_name):
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    result = subprocess.run(["python3", script_path], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {script_name}:")
        print(result.stderr)
    else:
        print(f"Successfully ran {script_name}")
        print(result.stdout)

if __name__ == "__main__":
    scripts = ["01break_images.py", "02augment.py", "03rename_files.py", "04blur_images.py"]
    
    for script in scripts:
        run_script(script)

