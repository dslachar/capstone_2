import os
API_KEY = os.environ['NYT_API_KEY']

import requests
import pymongo
import pprint
import pandas as pd

def init_mongo_client():
    client = pymongo.MongoClient()  # Initiate Mongo client
    db = client.nyt      # Access database
    coll = db.articles   # Access collection
    return db.articles   # return collection pointer

def make_request():
    endpoint = 'https://travel.state.gov/_res/rss/TAsTWs.xml'
    document_list = []
    for page in range(0,10):
        payload = {'api-key': API_KEY, 'page': page}
        json = _single_query(endpoint, payload)
        documents = _parse_response(json)
        document_list.extend(documents)
    return document_list

def _single_query(endpoint, payload):
    response = requests.get(endpoint, params=payload)
    if response.status_code == 200:
        print('request successful')
        return response.json()
    else:
        print ('WARNING status code {}'.format(response.status_code))

def _parse_response(json):
    document_list = json['response']['docs']
    return document_list

def query_collection(collection, query):
    return collection.find(query)


if __name__ == '__main__':

    nyt_article_collection = init_mongo_client()
    new_documents = make_request()

    print('{} documents received'.format(len(new_documents)))
    print('inserting documents into mongodb...')
    document_count = 0
    for doc in new_documents:
        try:
            nyt_article_collection.insert(doc)
            print(doc['headline']['main'])
            document_count += 1
        except pymongo.errors.DuplicateKeyError:
            print("duplicate record found... skipping...")
            continue
    print('done. {} documents successfully inserted to MongoDB'.format(document_count))

    # query = {'source': 'AP', 'word_count' : {$gt: 140}}
    # select = { snippet: 1, source: 1, type_of_material: 1}
    # cursor = query_collection(nyt_article_collection, (query, select))
    # df =  pd.DataFrame(list(cursor))
    # print(df)
