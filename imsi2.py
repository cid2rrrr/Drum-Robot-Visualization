# my_module.py

import time

my_arr = 0

def small_func():
    global my_arr
    my_arr += 1

def func():
    while True:
        small_func()
        time.sleep(1)
