import time, threading, json
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, Q
from flask import Flask, request

dates = '2019-01-21'
pcname = 'client01'
#pcname = 'winserver2008.example.com'

# s = s.query('range', **{'@timestamp':{'gte': '2019-01-21T05:29:03.000Z' , 'lt': '2019-01-21T05:29:04.000Z'}})
# q = Q('match', event_id= 7) & Q('match', computer_name = pcname)
# s = s.query(q)

for hh in range(0,23):
    for mm in range(0,59):
        for ss in range(0,59):
            es = Elasticsearch('192.168.2.140:9200')
            s = Search(using=es, index="winlog-2019-01-21")
            s = s[0:10000]
            # time the carry
            hha = hh
            mma = mm
            ssa = ss + 1
            if ss == 59:
                ssa = 0
                mma = mm + 1
            if mm == 59:
                mma = 0
                hha = hh + 1
            s = s.query('range', **{'@timestamp':{'gte': dates + 'T'+ str(hh).zfill(2) +':' + str(mm).zfill(2) +':' + str(ss).zfill(2) + '.000Z' , 'lt': dates + 'T' + str(hha).zfill(2) +':' + str(mma).zfill(2) +':' + str(ssa).zfill(2) + '.000Z'}})
            q = Q('match', event_id= 7) & Q('match', computer_name = pcname)
            s = s.query(q)
            responses = s.execute()

            if len(responses) > 0:
                with open(dates + '_' + pcname + '.txt', 'a', encoding='utf-8') as f:
                    # modify from zulu to JST
                    timedata = str(hh+9).zfill(2) + ':' + str(mm).zfill(2) + ':' + str(ss).zfill(2)
                    print(timedata)
                    f.write(timedata + ',')
                    for response in responses['hits']['hits']:
                        imageloaded = response['_source']['event_data']['ImageLoaded']
                        dllall = imageloaded.split('\\')
                        f.write(dllall[-1] + ' ')
                    f.write('\n')

