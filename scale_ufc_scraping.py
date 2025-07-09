import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
from datetime import datetime
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import random

# Set up logging
logging.basicConfig(
    filename='scrape_errors.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

BASE_URL = "http://ufcstats.com/statistics/fighters?char={}&page=all"
FIGHTER_PREFIX = "http://ufcstats.com/fighter-details/"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}

# Threading configuration for polite scraping
MAX_WORKERS = 4         # Lowered for rate limit
REQUEST_SLEEP = 0.3     # Increased for politeness
PROFILE_WORKERS = 4     # Lowered for rate limit

def get_fighter_links(max_fighters=10000):
    """Get fighter URLs from all alphabet pages with pagination"""
    links = []
    print(f"Scraping up to {max_fighters} fighters from A-Z pages...")
    
    for char in 'abcdefghijklmnopqrstuvwxyz':
        if len(links) >= max_fighters:
            break
            
        print(f"Fetching fighters from letter '{char.upper()}' ({len(links)}/{max_fighters} so far)")
        
        # Handle pagination for each letter
        page = 1
        while True:
            if page == 1:
                url = BASE_URL.format(char)
            else:
                url = f"http://ufcstats.com/statistics/fighters?char={char}&page={page}"
            
            try:
                res = requests.get(url, headers=HEADERS, timeout=10)
                res.raise_for_status()
                soup = BeautifulSoup(res.text, 'html.parser')
                anchors = soup.select('td.b-statistics__table-col a')
                
                # Check if we got any fighters on this page
                page_fighters = 0
                for a in anchors:
                    href = a.get('href')
                    if href and href.startswith(FIGHTER_PREFIX):
                        if href not in links:  # Avoid duplicates
                            links.append(href)
                            page_fighters += 1
                            if len(links) >= max_fighters:
                                break
                
                # If no fighters found on this page, we've reached the end
                if page_fighters == 0:
                    break
                    
                print(f"  Letter {char.upper()}, Page {page}: Found {page_fighters} fighters")
                
                # Move to next page
                page += 1
                
                # Be respectful to the server
                time.sleep(0.2)  # Reduced sleep time
                
            except Exception as e:
                print(f"  ⚠️  Error fetching letter '{char}' page {page}: {e}")
                break
    
    print(f"✅ Collected {len(links)} unique fighter URLs")
    return links

def parse_fighter_profile(url):
    """Parse individual fighter profile"""
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        res.raise_for_status()
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
    except Exception as e:
        print(f"  ❌ Error parsing {url}: {e}")
        return None

def standardize_date(event_str):
    """Extract and standardize dates from UFC event strings - finds the LAST date pattern"""
    if not isinstance(event_str, str):
        return ''
    
    # Remove newlines and extra spaces
    cleaned = ' '.join(event_str.split())
    
    # Debug: Print first few event strings to see what we're working with
    if hasattr(standardize_date, 'debug_count'):
        standardize_date.debug_count += 1
    else:
        standardize_date.debug_count = 1
    
    if standardize_date.debug_count <= 5:  # Only print first 5 for debugging
        print(f"DEBUG Event string: '{cleaned}'")
    
    # Find ALL date patterns in the string, then take the last one
    # This regex finds the full date string (not just groups)
    date_pattern = r'(Jan\.?|Feb\.?|Mar\.?|Apr\.?|May|Jun\.?|Jul\.?|Aug\.?|Sep\.?|Oct\.?|Nov\.?|Dec\.?)[ ]?\d{1,2},[ ]?\d{4}'
    matches = re.finditer(date_pattern, cleaned)
    
    # Get all matches and take the last one
    all_matches = list(matches)
    if all_matches:
        # Take the last date found (most likely to be the event date)
        last_match = all_matches[-1]
        date_str = last_match.group(0).replace('.', '')  # Remove periods
        try:
            result = pd.to_datetime(date_str, format='%b %d, %Y').strftime('%Y-%m-%d')
            if standardize_date.debug_count <= 5:
                print(f"DEBUG Extracted date: '{date_str}' -> '{result}'")
            return result
        except Exception as e:
            if standardize_date.debug_count <= 5:
                print(f"DEBUG Error parsing date '{date_str}': {e}")
            pass
    else:
        if standardize_date.debug_count <= 5:
            print(f"DEBUG No date pattern found in: '{cleaned}'")
    
    # Fallback: try to parse any date-like string
    try:
        result = pd.to_datetime(cleaned, errors='coerce').strftime('%Y-%m-%d')
        if standardize_date.debug_count <= 5 and result != 'NaT':
            print(f"DEBUG Fallback extracted: '{result}'")
        return result
    except Exception:
        return ''

def get_fight_history(url, fighter_name, fighter_id, name_to_id, max_retries=3):
    """Get fight history for a fighter with improved date extraction and retries"""
    for attempt in range(1, max_retries + 1):
        try:
            res = requests.get(url, headers=HEADERS, timeout=10)
            logging.info(f"FightHistory GET {url} [status={res.status_code}, len={len(res.content)}] (attempt {attempt})")
            if res.status_code == 429:
                logging.warning(f"429 Too Many Requests for {url} (attempt {attempt})")
                time.sleep(2 * attempt)
                continue
            if res.status_code != 200 or len(res.content) < 1000:
                raise Exception(f"Bad response: status={res.status_code}, len={len(res.content)}")
            soup = BeautifulSoup(res.text, 'html.parser')
            fight_table = soup.select_one("table.b-statistics__table")
            if not fight_table:
                fight_table = soup.select_one("table.fight-history")
            if not fight_table:
                fight_table = soup.select_one("table")
            if not fight_table:
                return []
            rows = fight_table.select("tbody tr")
            if not rows:
                rows = fight_table.select("tr")
            fights = []
            for i, row in enumerate(rows):
                cols = row.select("td")
                if len(cols) < 10:
                    continue
                try:
                    result = cols[0].text.strip()
                    opponent = cols[1].text.strip()
                    event = cols[6].text.strip()
                    method = cols[7].text.strip()
                    round_ = cols[8].text.strip()
                    time_ = cols[9].text.strip()
                    result = ' '.join(result.split())
                    opponent = ' '.join(opponent.split())
                    event = ' '.join(event.split())
                    method = ' '.join(method.split())
                    round_ = ' '.join(round_.split())
                    time_ = ' '.join(time_.split())
                    std_date = standardize_date(event)
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
                    logging.error(f"Error parsing fight row for {url}: {e}")
                    continue
            return fights
        except Exception as e:
            logging.error(f"Error getting fight history {url} (attempt {attempt}): {e}")
            if attempt == max_retries:
                print(f"  ❌ Error getting fight history {url} after {max_retries} attempts: {e}")
            time.sleep((0.3 + random.uniform(0, 0.2)) * attempt)
    return []

def scrape_single_fighter_profile(fighter_data):
    """Scrape a single fighter profile - designed for threading"""
    url, fighter_name, fighter_id, name_to_id = fighter_data
    
    try:
        # Get fighter profile data
        fighter = parse_fighter_profile(url)
        if not fighter:
            return None, []
        
        # Get fight history
        fights = get_fight_history(url, fighter_name, fighter_id, name_to_id)
        
        # Polite rate limiting
        time.sleep(REQUEST_SLEEP)
        
        return fighter, fights
        
    except Exception as e:
        print(f"  ❌ Error scraping {url}: {e}")
        return None, []

def parse_single_fighter_profile(url, max_retries=3):
    """Parse a single fighter profile - for threading, with retries and logging"""
    for attempt in range(1, max_retries + 1):
        try:
            res = requests.get(url, headers=HEADERS, timeout=10)
            logging.info(f"Profile GET {url} [status={res.status_code}, len={len(res.content)}] (attempt {attempt})")
            if res.status_code == 429:
                logging.warning(f"429 Too Many Requests for {url} (attempt {attempt})")
                time.sleep(2 * attempt)
                continue
            if res.status_code != 200 or len(res.content) < 1000:
                raise Exception(f"Bad response: status={res.status_code}, len={len(res.content)}")
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
            time.sleep(0.3 + random.uniform(0, 0.2))
            return fighter
        except Exception as e:
            logging.error(f"Error parsing profile {url} (attempt {attempt}): {e}")
            if attempt == max_retries:
                print(f"  ❌ Error parsing profile {url} after {max_retries} attempts: {e}")
            time.sleep((0.3 + random.uniform(0, 0.2)) * attempt)
    return None

def scrape_fighters_and_fights(max_fighters=1000):
    """Main function to scrape fighters and their fight histories with threading"""
    print("🚀 Starting UFC Data Pipeline (Multi-threaded)")
    print("="*50)
    
    # Step 1: Get fighter URLs
    fighter_links = get_fighter_links(max_fighters)
    
    # Step 2: Parse fighter profiles with threading
    print(f"\n📊 Parsing {len(fighter_links)} fighter profiles (using {PROFILE_WORKERS} threads)...")
    fighters = []
    completed_profiles = 0
    
    with ThreadPoolExecutor(max_workers=PROFILE_WORKERS) as pool:
        # Submit all profile parsing tasks
        futures = {pool.submit(parse_single_fighter_profile, url): url for url in fighter_links}
        
        # Process completed tasks
        for future in as_completed(futures):
            completed_profiles += 1
            if completed_profiles % 100 == 0:
                print(f"  Progress: {completed_profiles}/{len(fighter_links)} profiles processed")
            
            try:
                fighter = future.result()
                if fighter:
                    fighters.append(fighter)
            except Exception as e:
                print(f"  ❌ Error in profile thread: {e}")
    
    # Save fighters data
    fighters_df = pd.DataFrame(fighters)
    fighters_df = fighters_df.drop_duplicates(subset=['url', 'name']).reset_index(drop=True)
    fighters_df['fighter_id'] = fighters_df['url']
    
    # Save fighters.csv
    fighters_df[['fighter_id', 'name', 'height', 'weight', 'reach', 'stance', 'dob']].to_csv('fighters.csv', index=False)
    print(f"✅ Saved {len(fighters_df)} fighters to 'fighters.csv'")
    
    # Step 3: Scrape fight histories with threading
    print(f"\n🥊 Scraping fight histories for {len(fighters_df)} fighters (using {MAX_WORKERS} threads)...")
    
    # Build name to id mapping for opponent matching
    name_to_id = {row['name']: row['url'] for _, row in fighters_df.iterrows()}
    
    # Prepare data for threading
    fighter_data_list = []
    for _, row in fighters_df.iterrows():
        fighter_data_list.append((
            row['url'],           # url
            row['name'],          # fighter_name  
            row['fighter_id'],    # fighter_id
            name_to_id            # name_to_id mapping
        ))
    
    # Use ThreadPoolExecutor for parallel scraping
    all_fights = []
    completed_fighters = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        # Submit all tasks
        futures = {pool.submit(scrape_single_fighter_profile, data): data for data in fighter_data_list}
        
        # Process completed tasks
        for future in as_completed(futures):
            completed_fighters += 1
            if completed_fighters % 20 == 0:
                print(f"  Progress: {completed_fighters}/{len(fighter_data_list)} fighters processed ({len(all_fights)} fights collected)")
            
            try:
                fighter, fights = future.result()
                if fighter:
                    all_fights.extend(fights)
            except Exception as e:
                print(f"  ❌ Error in thread: {e}")
    
    # Save fights data with deduplication
    if all_fights:
        fights_df = pd.DataFrame(all_fights)
        # Remove duplicates that might occur from multi-threading
        fights_df = fights_df.drop_duplicates(
            subset=["fighter_id", "opponent", "event", "date"], 
            keep='first'
        )
        fights_df.to_csv("fights.csv", index=False)
        print(f"✅ Saved {len(fights_df)} fights to 'fights.csv'")
    else:
        print("❌ No fight data was collected")
    
    # Summary
    print("\n" + "="*50)
    print("📈 PIPELINE SUMMARY")
    print("="*50)
    print(f"Fighters collected: {len(fighters_df)}")
    print(f"Fights collected: {len(all_fights)}")
    print(f"Average fights per fighter: {len(all_fights)/len(fighters_df):.1f}")
    
    # Check date extraction quality
    if all_fights:
        fights_df = pd.DataFrame(all_fights)
        valid_dates = fights_df['date'].notna() & (fights_df['date'] != '')
        print(f"Fights with valid dates: {valid_dates.sum()}/{len(fights_df)} ({valid_dates.mean()*100:.1f}%)")
    
    print("\n🎯 Ready for modeling!")

if __name__ == "__main__":
    # Scrape the full UFC roster
    scrape_fighters_and_fights(max_fighters=5000)  # Full roster (typically 3000-4000 fighters)  is this not it?