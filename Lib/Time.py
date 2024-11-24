# Lib/Time.py

import time

def InitializeTimeCount():
    start_time = time.time()
    return start_time

def StopTime(start_time):
    end_time = time.time()
    return end_time - start_time

def PrintElapsedTime(elapsed_time):
    print(f"\nTempo total de execução: {elapsed_time:.2f} segundos")