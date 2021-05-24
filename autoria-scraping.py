import requests
import bs4

page = requests.request('GET', 'https://auto.ria.com/uk/auto_mazda_6_29640989.html')
page.raise_for_status()
soup = bs4.BeautifulSoup(page.content, "html.parser")
photo = soup.select('picture')
print(photo[0].getText())
