import subprocess
import time
from datetime import datetime, timedelta
import os

today = datetime.now().strftime("%Y-%m-%d")
CHAT_HISTORY_FILE = f"memory_storage/miao_memory/chat_history/{today}_chat_history.txt"

def run_first_script():
    return subprocess.Popen(["streamlit", "run", "./main.py"])

# def run_second_script():
#     subprocess.run(["python", "./daily_ma.py"])

def run_third_script():
    subprocess.run(["python", "./realtime_ma.py"])

def check_file_updated(file_path):
    if not os.path.exists(file_path):
        return False
    current_time = time.time()
    modified_time = os.path.getmtime(file_path)
    return (current_time - modified_time) < 1800

def main():
    first_script_process = run_first_script()
    first_script_finished = False

    # try:
    while not first_script_finished:
        first_script_process.poll()
        if first_script_process.returncode is not None:
            first_script_finished = True
        else:
            if check_file_updated(CHAT_HISTORY_FILE):
                run_third_script()
            time.sleep(1800)
    # except KeyboardInterrupt:
    #     print("Ctrl+C detected. Waiting for the first script to terminate...")
    #     first_script_process.wait()

    # print("First script has terminated or Ctrl+C was pressed. Running the second script...")
    # run_second_script()

    # Check if it's 23:55 and run the second script
    # while True:
    #     now = datetime.now()
    #     if now.hour == 23 and now.minute == 55:
    #         print("It's 23:55. Running the second script...")
    #         run_second_script()
    #         break
    #     time.sleep(60)  # Check every minute

if __name__ == "__main__":
    main()