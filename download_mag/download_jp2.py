import requests
import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import os, sys, csv
import shutil

def download_from_helioviewer():
    """
    This functions download 4k magnetogram jp2s from helioviewer api (https://api.helioviewer.org/v2/getJP2Image/) at a cadence of 12mins as available.
    Inside the basedir/ : downloaded magnetograms are stored creating a heirarchy as:
    basedir/
        year/
            month/
                day/filename.jp2

    The filename are renamed as: HMI.m{year}.{month}.{day}_{hour}.{minute}.{second}.jp2
    """

    start_date = '2010-12-01 00:00:00'
    basedir = '/data/hmi_compressd/'

    #File counter
    counter = 0

    #Start Date
    dt = datetime.datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
    lis = []
    while True:
        hours = datetime.timedelta(minutes=12)
        dt = dt+ hours
        final_date = str(dt.date()) + 'T' + str(dt.time()) + 'Z'
        if dt.year>2021:
            break
        Path(f'{basedir}/{dt.year}/{dt.month:02d}/{dt.day:02d}').mkdir(parents=True, exist_ok=True)
        #Defining name of downloaded images based on the date and time
        filename = 'HMI.m' + str(dt.year) + '.' +  f'{dt.month:02d}' + '.' + f'{dt.day:02d}' + '_'\
            + f'{dt.hour:02d}' + '.' + f'{dt.minute:02d}' + '.' + f'{dt.second:02d}' + '.jp2'
        file_loc_with_name = f'{basedir}/{dt.year}/{dt.month:02d}/{dt.day:02d}/' + filename

        #Using jpip=True gives the uri of the image which we use to parse time_stamp of available image
        #Detail documentation is provided on api.helioviewer.org
        requestString = "https://api.helioviewer.org/v2/getJP2Image/?date=" + final_date + "&sourceId=19&jpip=true"
        #print(requestString, '-->Requested')

        #Parsing date from the recived uri
        response = requests.get(requestString)
        url = str(response.content)
        url_temp = url.rsplit('/', 1)[-1]
        date_recieved = url_temp.rsplit('__', 1)[0][:-4]
        recieved = datetime.datetime.strptime(date_recieved, "%Y_%m_%d__%H_%M_%S")
    #     print(abs(dt-recieved))
        
        #Now comparing the timestamp of available image and requested image
        #Download only if within the window of 12 minutes.
        if(abs(recieved-dt)<=(datetime.timedelta(minutes=12)) or abs(dt-recieved)<=(datetime.timedelta(minutes=12))):
            #print(dt, recieved)
            #This uri provides access to the actual resource ( i.e., images)
            request_uri = "https://api.helioviewer.org/v2/getJP2Image/?date=" + final_date + "&sourceId=19"
            hmidata = requests.get(request_uri)
            open(file_loc_with_name,'wb').write(hmidata.content)
    #         print(final_date, '-->Downloaded')
            lis.append([dt, recieved])
            counter+=1
            if counter%2500==0:
                print(counter, 'Files Downloaded')

    #Total Files Downloaded
    print('Total Files Downloaded: ', counter)

if __name__ == '__main__':
    download_from_helioviewer()
