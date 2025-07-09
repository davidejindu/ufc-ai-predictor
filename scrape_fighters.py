import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

BASE_URL = "http://ufcstats.com/statistics/fighters?char={}&page=all"
FIGHTER_PREFIX = "http://ufcstats.com/fighter-details/"
HEADERS = {"User-Agent": "Mozilla/5.0"}

def get_fighter_links(max_fighters=20):
    links = []
    for char in 'abcdefghijklmnopqrstuvwxyz':
        url = BASE_URL.format(char)
        print(f"Fetching fighter list from: {url}")
        res = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(res.text, 'html.parser')
        anchors = soup.select('td.b-statistics__table-col a')
        for a in anchors:
            href = a.get('href')
            if href and href.startswith(FIGHTER_PREFIX):
                links.append(href)
            if len(links) >= max_fighters:
                return links
        time.sleep(1)
    return links

def parse_fighter_profile(url):
    print(f"Parsing profile: {url}")
    res = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(res.text, 'html.parser')
    fighter = {}

    fighter['url'] = url

    name = soup.select_one('span.b-content__title-highlight')
    fighter['name'] = name.text.strip() if name else "N/A"

    stats = soup.select('li.b-list__box-list-item')
    for stat in stats:
        text = stat.get_text(separator=' ').strip()
        if 'Height:' in text:
            fighter['height'] = text.split(':')[1].strip()
        elif 'Reach:' in text:
            fighter['reach'] = text.split(':')[1].strip()
        elif 'DOB:' in text:
            fighter['dob'] = text.split(':')[1].strip()
        elif 'STANCE:' in text:
            fighter['stance'] = text.split(':')[1].strip()
        elif 'Weight:' in text:
            fighter['weight'] = text.split(':')[1].strip()

    return fighter

# Main pipeline
if __name__ == "__main__":
    fighter_links = get_fighter_links(20)
    fighters = [parse_fighter_profile(url) for url in fighter_links]

    df = pd.DataFrame(fighters)
    df.to_csv("ufc_fighters_20.csv", index=False)
    print("\n✅ Done! Data saved to 'ufc_fighters_20.csv'")
