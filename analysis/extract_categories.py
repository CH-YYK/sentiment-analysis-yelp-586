from bs4 import BeautifulSoup

class extract_categories(object):
    def __init__(self, path='restaurant_categories.html'):
        self.html = open(path, 'r').read()
        self.soup = BeautifulSoup(self.html, 'lxml')

    def getlist(self):
        return set([cat.strip() for cat in self.soup.find_all('li')])
