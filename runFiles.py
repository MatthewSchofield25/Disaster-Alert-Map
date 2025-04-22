import time
import datetime
import subprocess
import logging
import sys
import KenVaModels
import SearchandSendPosts
import os
os.environ["PYTHONUTF8"] = "1"
from dotenv import load_dotenv
load_dotenv() #load the .env file

venv_python = os.getenv("MY_VENV_PYTHON")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def run_scheduled_tasks():
    if(venv_python):
        print(f"venv={venv_python}")
    else:
        print("No virtual environment activated.")

    
    while True:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        logging.info(f"Searching and sending posts at time = {current_time}")
        
        try:
            # SearchandSendPosts before calling models
            logging.info("Running SearchandSearchandSendPosts.py")
            send_posts_result = subprocess.run(
                [venv_python, "SearchandSendPosts.py"],
                stdout=sys.stdout,
                stderr=sys.stderr,
                text=True,
                encoding="utf-8",
                check=False
            )    
            print(f"SearchandSendPosts.py returned code: {send_posts_result.returncode}")

            if send_posts_result.returncode == 0:
                logging.info("SearchandSendPosts.py executed successfully")
            else:
                logging.error(f"SearchandSendPosts.py failed with exit code {send_posts_result.returncode}")
                logging.error(f"Error: {send_posts_result.stderr}")     
        except Exception as e:
            logging.error(f"Error in SearchandSendPosts.py tasks: {str(e)}")
        
        
        try:
            # after sending posts, run through the models
            logging.info("Running KenVaModels.py")
            models_result = subprocess.run(
            [venv_python, "KenVaModels.py"], 
                stdout=sys.stdout,
                stderr=sys.stderr,
                text=True, 
                check=False
            )
            if models_result.returncode == 0:
                logging.info("KenVaModels.py executed successfully")
            else:
                logging.error(f"KenVaModels.py failed with exit code {models_result.returncode}")
                logging.error(f"Error: {models_result.stderr}")
        except Exception as e2:
            logging.error(f"Error in KenVaModels.py tasks: {str(e2)}")

        # Calculate next run time
        next_run = datetime.datetime.now() + datetime.timedelta(hours=1)
        logging.info(f"Next execution scheduled at {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Sleep for one hour
        print("Wait one hour...")
        time.sleep(3600)
        print("One hour passed, running again.")
        

if __name__ == "__main__":
    logging.info("Starting scheduler")
    run_scheduled_tasks()
