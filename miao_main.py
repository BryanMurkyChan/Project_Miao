import subprocess
import time
from datetime import datetime, timedelta
import os

today = datetime.now().strftime("%Y-%m-%d")
CHAT_HISTORY_FILE = f"memory_storage/miao_memory/chat_history/{today}_chat_history.txt"

def run_first_script():
    return subprocess.Popen(["streamlit", "run", "./main.py"])

def run_second_script():
    subprocess.run(["python", "./realtime_ma.py"])

def check_file_updated(file_path):
    if not os.path.exists(file_path):
        return False
    current_time = time.time()
    modified_time = os.path.getmtime(file_path)
    return abs(current_time - modified_time) < 1800

def main():
    first_script_process = run_first_script()
    first_script_finished = False

    while not first_script_finished:
        first_script_process.poll()
        if first_script_process.returncode is not None:
            first_script_finished = True
        else:
            if check_file_updated(CHAT_HISTORY_FILE):
                run_second_script()
            time.sleep(1800)

if __name__ == "__main__":
    main()
