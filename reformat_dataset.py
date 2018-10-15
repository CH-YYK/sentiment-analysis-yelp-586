import pandas as pd
import json

class Json2Data(object):

    def __init__(self, json_path, to_tsv=True):
        # load texts
        print("loading texts...")
        with open(json_path, 'r') as f:
            self.text_file = f.readlines()

        # concate dataframe
        print("concating dataframes...")
        self.DataFrames = self.json_from_text(self.text_file)

        # dictionary to dataframe
        print("convert dictionary to dataframe")
        self.Data = pd.DataFrame(self.DataFrames, index=range(len(self.DataFrames)))
        print(self.Data.shape)

        # write data to external files
        name = json_path.replace('.json', '.tsv')
        print('write data to', name)
        self.Data.to_csv(name, sep='\t', index=False)

    def json_from_text(self, text):
        return [json.loads(string) for string in text]

if __name__ == '__main__':
    json_path = '/Users/kayleyang/Desktop/sentiment-analysis-yelp-586/data/yelp_academic_dataset_review.json'
    test = Json2Data(json_path)
