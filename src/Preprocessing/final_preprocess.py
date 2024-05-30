import subprocess

def run_script(script_name):
    result = subprocess.run(["python", script_name], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {script_name}:")
        print(result.stderr)
    else:
        print(f"Successfully ran {script_name}")
        print(result.stdout)

if __name__ == "__main__":
    scripts = ["break_images.py", "augment.py", "rename_files.py", "blur_images.py"]
    
    for script in scripts:
        run_script(script)
