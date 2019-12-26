from datetime import datetime, timedelta
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, Q
import pandas as pd
import re

df = pd.DataFrame(columns=['timestamp', 'pcname', 'processid', 'image', 'imageloaded'])

dates = '2019-01-21'
# tomorrow = dates + timedelta(days=1)
# nextday =  datetime.strftime(tomorrow, '%Y-%m-%d')

for hh in range(0,23):
    es = Elasticsearch('192.168.2.140:9200')
    s = Search(using=es, index="winlog-2019-01-21")
    s = s[0:10000]
    s = s.query('range', **{'@timestamp':{'gte': dates + 'T'+ str(hh).zfill(2) +':00:00.000Z' , 'lt': dates + 'T' + str(hh).zfill(2) +':59:59.999Z'}})
    q = Q('match', event_id= 7)
    s = s.query(q)
    responses = s.execute()

    if len(responses) > 0:
        for response in responses['hits']['hits']:
            timestamp = response['_source']['@timestamp']
            pcname = response['_source']['computer_name']
            processid = response['_source']['event_data']['ProcessId']
            image = response['_source']['event_data']['Image'].split('\\')[-1]
            imageloaded = response['_source']['event_data']['ImageLoaded'].split('\\')[-1]

            series = pd.Series([timestamp, pcname, processid, image, imageloaded], index=df.columns)
            df = df.append(series, ignore_index = True)

df['timestamp'] = pd.to_datetime(df['timestamp'])
pcnamelist = (df['pcname'].unique())
processidlist = (df['processid'].unique())
imagelist = (df['image'].unique())
dateindex = pd.date_range('2019-01-21','2019-01-22',freq='min')

with open(str(dates) + '_event7.csv', mode='w') as f:
    n = 0
    for pcnameline in pcnamelist:
        for processidline in processidlist:
            for imageline in imagelist:
                if len(df[(df['pcname'] == pcnameline) & (df['processid'] == processidline) & (df['image'] == imageline)]['imageloaded']):
                    line = str(imageline) + ','
                    line = line + str(pcnameline) + ',' + str(processidline) +','
                    line = line + str(df[(df['pcname'] == pcnameline) & (df['processid'] == processidline) & (df['image'] == imageline)]['imageloaded'].to_string(index=False)).strip('\\')
                    line = re.sub('\n', ' ', line)
                    line = re.sub(' {2,}', ' ', line)
                    line = line + '\n'
                    f.write(line)
        n += 1
        print(str(n) + ' / ' + str(len(pcnamelist)))


# with open(str(dates) + '_event7.csv', mode='w') as f:
#     for t in range(len(dateindex) - 1):
#         for pcnameline in pcnamelist:
#             for processidline in processidlist:
#                 if len(df[(df['timestamp'] >= dateindex[t]) & (df['timestamp'] < dateindex[t+1]) & (df['pcname'] == pcnameline) & (df['processid'] == processidline)]['imageloaded']):
#                     line = str(df[(df['processid'] == processidline)]['image'].iloc[0])
#                     line = line + ','
#                     line = line + str(pcnameline) + ',' + str(processidline)
#                     line = line + ','
#                     line = line + str(df[(df['timestamp'] >= dateindex[t]) & (df['timestamp'] < dateindex[t+1]) & (df['pcname'] == pcnameline) & (df['processid'] == processidline)]['imageloaded'].to_string(index=False)).strip('\\')
#                     line = re.sub('\n', ' ', line)
#                     line = re.sub(' {2,}', ' ', line)
#                     line = line + '\n'
#                     f.write(line)
#         print(str(t) + '/' + str(len(dateindex)))