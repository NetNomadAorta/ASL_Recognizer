import pyautogui
import time
import datetime
import random


date_time = datetime.datetime.now()

print ("Current date and time = %s" % date_time)

print ("Today's date:  = %s/%s/%s" % (date_time.day, date_time.month, date_time.year))

print ("The time is now: = %s:%s:%s" % (date_time.hour, date_time.minute, date_time.second))

while date_time.hour < 7 or date_time.hour >= (12+7):
    # Move pointer relative to current position
    time.sleep(1)
    pyautogui.press('volumeup')
    time.sleep(1)
    pyautogui.press('volumedown')

    time.sleep(30 + random.randint(0,60))
    date_time = datetime.datetime.now()
    print("\r", "The updated time is: = %s:%s:%s" % (date_time.hour,
                                                 date_time.minute,
                                                 date_time.second),
          end="")

print("\nDone and exiting!")