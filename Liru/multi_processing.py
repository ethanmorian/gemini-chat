# multi_processing.py

import os
from multiprocessing import Pool


def parallel_processing(func, data):
    with Pool(processes=os.cpu_count()) as pool:
        results = pool.map(func, data)
        
    if all(results):
        print("All tasks were completed successfully!")
    else:
        print("Some tasks were not completed successfully.")
        
    return results