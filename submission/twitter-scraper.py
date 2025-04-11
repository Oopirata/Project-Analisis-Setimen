import snscrape.modules.twitter as sntwitter
import pandas as pd
from datetime import datetime, timedelta
import time
import random

# Definisikan keyword/query pencarian dan jumlah data yang diinginkan
search_queries = [
    "indihome", "telkomsel", "xl axiata", "smartfren", 
    "gojek", "grab", "shopee", "tokopedia", "bukalapak",
    "netflix indonesia", "disney+ hotstar"
]
tweets_per_query = 1000  # Targetkan 1000 tweet per query
total_tweets = 0
max_tweets = 12000  # Target total

# Buat list untuk menyimpan data
tweets_data = []

# Fungsi untuk menunggu secara random untuk menghindari pembatasan
def random_sleep():
    time.sleep(random.uniform(1, 3))

# Definisikan rentang tanggal untuk pencarian (6 bulan terakhir)
end_date = datetime.now()
start_date = end_date - timedelta(days=180)
date_range = f"since:{start_date.strftime('%Y-%m-%d')} until:{end_date.strftime('%Y-%m-%d')}"

# Scraping untuk setiap query
for query in search_queries:
    print(f"Scraping untuk query: {query}")
    count = 0
    
    # Buat query lengkap dengan filter bahasa dan rentang tanggal
    full_query = f"{query} lang:id {date_range}"
    
    try:
        # Gunakan generator scraper
        for tweet in sntwitter.TwitterSearchScraper(full_query).get_items():
            # Ambil atribut tweet yang relevan
            tweets_data.append({
                'id': tweet.id,
                'date': tweet.date,
                'content': tweet.content,
                'username': tweet.user.username,
                'like_count': tweet.likeCount,
                'retweet_count': tweet.retweetCount,
                'reply_count': getattr(tweet, 'replyCount', 0),
                'query': query
            })
            
            count += 1
            total_tweets += 1
            
            # Berhenti jika sudah mencapai target untuk query ini atau total target
            if count >= tweets_per_query or total_tweets >= max_tweets:
                break
                
        random_sleep()
        print(f"Berhasil scraping {count} tweets untuk query: {query}")
        
    except Exception as e:
        print(f"Error pada query {query}: {e}")
        random_sleep()
        continue

# Convert ke dataframe
tweets_df = pd.DataFrame(tweets_data)

# Simpan ke file
tweets_df.to_csv('twitter_sentiment_dataset.csv', index=False)
print(f"Total data berhasil discrape: {len(tweets_df)}")
print(f"Data disimpan ke file 'twitter_sentiment_dataset.csv'")