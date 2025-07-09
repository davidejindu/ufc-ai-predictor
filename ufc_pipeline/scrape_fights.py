import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from datetime import datetime
import re

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}

def standardize_date(date_str):
    # Use regex to extract date patterns like 'Jan. 21, 2023' or 'Jul 01, 2022' or 'Dec. 04, 2010'
    if not isinstance(date_str, str):
        return ''
    # Remove newlines and extra spaces
    cleaned = ' '.join(date_str.split())
    # Regex for dates like 'Jan. 21, 2023' or 'Jul 01, 2022' or 'Dec. 04, 2010'
    match = re.search(r'(Jan\.?|Feb\.?|Mar\.?|Apr\.?|May|Jun\.?|Jul\.?|Aug\.?|Sep\.?|Oct\.?|Nov\.?|Dec\.?)[ ]?\d{1,2},[ ]?\d{4}', cleaned)
    if match:
        date_part = match.group(0).replace('.', '')  # Remove periods from month
        try:
            return pd.to_datetime(date_part, format='%b %d, %Y').strftime('%Y-%m-%d')
        except Exception:
            pass
    # Try fallback: parse any date
    try:
        return pd.to_datetime(cleaned, errors='coerce').strftime('%Y-%m-%d')
    except Exception:
        return ''

def get_fight_history(url, fighter_name, fighter_id, name_to_id):
    print(f"Scraping fight history for: {fighter_name}")
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, 'html.parser')

        fight_table = soup.select_one("table.b-statistics__table")
        if not fight_table:
            fight_table = soup.select_one("table.fight-history")
        if not fight_table:
            fight_table = soup.select_one("table")
        if not fight_table:
            print(f"  ⚠️  No fight table found for {fighter_name}")
            return []

        rows = fight_table.select("tbody tr")
        if not rows:
            rows = fight_table.select("tr")
        fights = []

        for i, row in enumerate(rows):
            cols = row.select("td")
            if len(cols) < 7:
                continue
            try:
                result = cols[0].text.strip()
                opponent = cols[1].text.strip()
                event = cols[2].text.strip()
                date = cols[3].text.strip()
                method = cols[4].text.strip()
                round_ = cols[5].text.strip()
                time_ = cols[6].text.strip()

                # Standardize date - look in both date and event fields
                std_date = standardize_date(date)
                if not std_date:
                    std_date = standardize_date(event)

                # Try to match opponent to fighter_id
                opponent_id = name_to_id.get(opponent, '')

                fights.append({
                    "fighter_id": fighter_id,
                    "result": result,
                    "opponent": opponent,
                    "opponent_id": opponent_id,
                    "event": event,
                    "date": std_date,
                    "method": method,
                    "round": round_,
                    "time": time_
                })
            except Exception as e:
                print(f"  ⚠️  Error parsing row {i} for {fighter_name}: {e}")
                continue
        print(f"  ✅ Successfully parsed {len(fights)} fights for {fighter_name}")
        return fights
    except requests.exceptions.RequestException as e:
        print(f"  ❌ Network error for {fighter_name}: {e}")
        return []
    except Exception as e:
        print(f"  ❌ Error fetching for {fighter_name}: {e}")
        return []

def main():
    # Read and deduplicate fighters
    fighter_data = pd.read_csv("ufc_fighters_20.csv")
    fighter_data = fighter_data.drop_duplicates(subset=['url', 'name']).reset_index(drop=True)
    print(f"Processing {len(fighter_data)} unique fighters...")

    # Assign fighter_id as url
    fighter_data['fighter_id'] = fighter_data['url']
    # Build name to id mapping for opponent matching
    name_to_id = {row['name']: row['url'] for _, row in fighter_data.iterrows()}

    # Save fighters.csv
    fighter_data[['fighter_id', 'name', 'height', 'weight', 'reach', 'stance', 'dob']].to_csv('fighters.csv', index=False)

    all_fights = []
    for idx, row in fighter_data.iterrows():
        fighter_id = row['fighter_id']
        fighter_url = row['url']
        fighter_name = row['name']
        fights = get_fight_history(fighter_url, fighter_name, fighter_id, name_to_id)
        all_fights.extend(fights)
        if (idx + 1) % 10 == 0:
            print(f"Progress: {idx + 1}/{len(fighter_data)} fighters processed")
        time.sleep(1)

    if all_fights:
        df = pd.DataFrame(all_fights)
        df.to_csv("fights.csv", index=False)
        print(f"✅ Fight history saved to 'fights.csv' with {len(df)} fights")
    else:
        print("❌ No fight data was collected. Check the website structure or network connection.")

if __name__ == "__main__":
    main()
