import pandas as pd
import numpy as np
import glob
import ujson as json
import re
import os
from bs4 import BeautifulSoup

# extract categories for restaurants
def extract_categories(path='restaurant_categories.html'):
    html = open(path, 'r').read()
    soup = BeautifulSoup(html, 'lxml')
    return set([cat.getText(strip=True) for cat in soup.find_all('li')])

## Path to dataset
path_business = "../data/business.tsv"
path_review2017 = "../data/reviews_2017.json"
path_review2017_clean = "../data/reviews_2017_clean.json"

# clean strings
def clean_str(string):
    '''
    Tokenization/string cleaning forn all dataset except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    '''
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"[?]{7,}", " \? ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\n", " ", string)
    return string.strip().lower()

# clean categories
def clean_cat(categories, cate_list):
    categories = categories.split(', ')
    for cat in categories:
        if cat in cate_list:
            return ', '.join(categories)
    return np.NaN

# load json for reviews 2017
if not os.path.isfile(path_review2017_clean):
    print("load raw reviews of 2017")
    with open(path_review2017, 'r') as f:
        json_list = [json.loads(line) for line in f.readlines()]

    # clean str
    print("clean texts in json")
    for num in range(len(json_list)):
        json_list[num]['text'] = clean_str(json_list[num]['text'])

    # save cleaned json to file
    print("save cleaned json to file")
    with open(path_review2017_clean, "w") as f:
        f.writelines([json.dumps(line) + '\n' for line in json_list])
else:
    print("load existing json file")
    with open(path_review2017_clean, "r") as f:
        json_list = [json.loads(line) for line in f.readlines()]

## load data
print("loading data: 'data_business', 'review2017'")
data_business = pd.read_table(path_business, sep='\t')
data_review2017 = pd.DataFrame(json_list)[['text', 'stars', 'business_id', 'date', 'review_id']].\
    sort_values('date', ascending=False).dropna()

## extract cate_list
print("loading categories")
cate_list = extract_categories()

## join reviews to businesses
print("join reviews to business, and clean categories")
business_review2017 = data_business.join(data_review2017.set_index('business_id'), on='business_id', rsuffix='_review').dropna()
business_review2017['categories'] = business_review2017['categories'].apply(lambda x: clean_cat(x, cate_list=cate_list))
business_review2017 = business_review2017.dropna()
business_review2017.index = range(business_review2017.shape[0])


## group by business_id, find mean for review stars, find most recent 5 reviews
print("group by 'business_id' and summarise")
group_review_business = business_review2017.groupby('business_id')
aggregation = {"text": lambda x: ' ; '.join(x[:5]),
               "stars_review": lambda x: x.mean(),
               "categories": "first",
               "state": "first",
               "stars": "first",
               "longitude": "first",
               "latitude": "first"}
tmp = group_review_business.agg(aggregation)
print(max([len(i.split()) for i in tmp['text']]))

## write data out
tmp['class'] = pd.cut(tmp['stars_review'], [0,1.5,2.5,3.5,4.5,5.5],labels=[1,2,3,4,5])
tmp.to_csv("../data/business_reviews2017.tsv", sep='\t')


# oversamplint
