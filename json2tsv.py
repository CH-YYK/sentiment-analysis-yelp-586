import pandas as pd
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--json_path", "-j", type=str, help="The path to json file for reviews which will be converted to tsv")
args = vars(parser.parse_args())

class json2csv(object):
    def __init__(self, json_path):
        # load json list from json path
        print("Read json_file at %s" % json_path)
        with open(json_path, 'r') as json_list:
            self.jsons = json_list.readlines()

        # convert each line of json to dictionary
        self.jsons = [json.loads(line.strip()) for line in self.jsons]
        print('converting jsons to data set')
        self.data = pd.DataFrame(self.jsons)
        print("Data converted.")

        # write table out to files
        self.tsv_name = json_path.replace('.json', '.tsv')
        self.data.to_csv(self.tsv_name, sep='\t', index=False, columns=['text', 'stars'])

if __name__ == '__main__':
    json2csv(json_path=args['json_path'])
    print('tsv output.')