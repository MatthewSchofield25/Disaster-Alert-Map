import time
import datetime
import subprocess
import logging
import sys
import KenLSTMModel
import SearchandSendPosts

import os

env = os.environ.copy()
print("env is {env}")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def run_scheduled_tasks():
    while True:
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        
        print(f"SearchandSendPosts is using Python: {sys.executable}")
        print(f"sys.path: {sys.path}")

        python_executable = sys.executable
        logging.info(f"Using Python interpreter: {python_executable}")
        logging.info(f"Searching and sending posts at time = {current_time}")
        
        try:
            # SearchandSendPosts before calling models
            logging.info("Running SearchandSearchandSendPosts.py")
            send_posts_result = subprocess.run(
                [python_executable, "SearchandSendPosts.py"], 
                capture_output=True, 
                text=True, 
                check=False
            )
            
            if send_posts_result.returncode == 0:
                logging.info("SearchandSendPosts.py executed successfully")
                # after sending posts, run through the models
                logging.info("Running KenLSTMModel.py")
                models_result = subprocess.run(
                [python_executable, "KenLSTMModel.py"], 
                capture_output=True, 
                text=True, 
                check=False
                )
                
                if models_result.returncode == 0:
                    logging.info("KenLSTMModel.py executed successfully")
                else:
                    logging.error(f"KenLSTMModel.py failed with exit code {models_result.returncode}")
                    logging.error(f"Error: {models_result.stderr}")

            else:
                logging.error(f"SearchandSendPosts.py failed with exit code {send_posts_result.returncode}")
                logging.error(f"Error: {send_posts_result.stderr}")
                
        except Exception as e:
            logging.error(f"Error in scheduled tasks: {str(e)}")
        
        # Calculate next run time
        next_run = datetime.datetime.now() + datetime.timedelta(hours=1)
        logging.info(f"Next execution scheduled at {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Sleep for one hour
        #time.sleep(3600)
        print("Wait one hour...")
        time.sleep(3600)
        print("One hour passed, running again.")


if __name__ == "__main__":
    logging.info("Starting scheduler")
    run_scheduled_tasks()