import sys, os
notebook_dir = os.getcwd()
project_dir = os.path.abspath(os.path.join(notebook_dir, '..'))
if project_dir not in sys.path:
    sys.path.append(project_dir)
# logging_setup.py
import logging
import time

# Set up logging
logging.basicConfig(filename='logs/pipeline_log.txt', 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def log_execution_time(func):
    """Decorator to log the execution time of a function and handle errors."""
    def wrapper(*args, **kwargs):
        try:
            start_time = time.time()  # Record start time
            result = func(*args, **kwargs)  # Execute the function
            logging.info(f'{func.__name__} completed')
            end_time = time.time()  # Record end time
            elapsed_time = end_time - start_time  # Calculate elapsed time
            logging.info(f"{func.__name__} completed in {elapsed_time:.2f} seconds.")
            return result
        except Exception as e:
            # Log the error message with traceback
            logging.error(f"An error occurred in {func.__name__}: {str(e)}")
            logging.error("Traceback: " + traceback.format_exc())  # Capture full traceback
            raise  # Optionally re-raise the exception if you want the program to crash
    return wrapper
