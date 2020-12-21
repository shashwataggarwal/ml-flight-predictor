import os
from datetime import timedelta, date
import time

flight_paths = [{'from':'BOM','to':'DEL'},
                {'from':'DEL','to':'BOM'},
                {'from':'DEL','to':'GOI'},
                {'from':'BLR','to':'DEL'},
                {'from':'BOM','to':'GOI'},
                {'from':'BOM','to':'BLR'},
                {'from':'DEL','to':'CCU'}]  

startDate = date.today() + timedelta(1)
print(startDate)

for i in range(36):
    for obj in flight_paths:
        os.system('python3 dataFetcher.py {} {} {}'.format(obj['from'],obj['to'],(startDate+timedelta(i)).strftime('%d/%m/%Y')))
        time.sleep(5)
    print('Done Day {}'.format(i+1))
