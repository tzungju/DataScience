import urllib.request
from bs4 import BeautifulSoup

quote_page = 'http://www.bloomberg.com/quote/SPX:IND'
page = urllib.request.urlopen(quote_page)
soup = BeautifulSoup(page, 'html.parser')
# Take out the <div> of name and get its value
name_box = soup.find('h1', attrs={'class': 'name'})
# strip() is used to remove starting and trailing
name = name_box.text.strip() 
value_box = soup.find('div', attrs={'class': 'price'})
value = value_box.text.strip() 
print (name + ':' + value)


value_box = soup.find('div', attrs={'class': 'change-container'})
print('value_box', value_box)

value = value_box.get_text().strip() 
print('value', value)
arr = value.split(' ')
print(arr[0]+','+arr[-1])

# for item in arr:
    # if item.strip() != '':
        # print (item)