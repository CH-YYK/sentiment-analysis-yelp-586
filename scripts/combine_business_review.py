import pandas as pd
import json

# load business data
with open("../data/yelp_academic_dataset_business.json", 'r') as business:
    business_jsons = [json.loads(line) for line in business.readlines()]

# convert jsons to data
# for business: stars, business_id, latitude, longitude, name, states, categories, review_count
business = pd.DataFrame(business_jsons)[['business_id', 'stars', 'name', 'categories', 'latitude', 'longitude', 'state', 'review_count']]

# load review data
reviews = pd.read_table('../data/reviews_2017.tsv', sep='\t',
                        names=['business_id', 'cool', 'date', 'funny', 'likes', 'review_id', 'review_stars', 'text', 'useful', 'user_id'])

# join two datasets
combined_ = business.set_index('business_id').join(reviews.set_index('business_id'), how='inner')
combined_['business_id'] = combined_.index

# write to tsv
combined_.to_csv('../data/business_review_2017.tsv', sep='\t', index=False)