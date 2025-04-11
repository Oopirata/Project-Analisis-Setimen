import pandas as pd
import time
import random
from google_play_scraper import app, Sort, reviews_all

# Daftar aplikasi populer di Indonesia
app_ids = [
    'com.gojek.app',  # Gojek
    'com.grabtaxi.passenger',  # Grab
    'com.shopee.id',  # Shopee
    'com.tokopedia.tkpd',  # Tokopedia
    'com.bukalapak.android',  # Bukalapak
    'com.telkom.mwallet',  # LinkAja
    'com.dana.id',  # Dana
    'com.ovo.aladin',  # OVO
    'com.netflix.mediaclient',  # Netflix
    'com.mytel.mytelpay',  # Telkomsel MyPoin
    'com.xl.myxl',  # myXL
    'com.indosat.im3',  # myIM3
    'id.co.smartfren.mykuota'  # Smartfren
]

# List untuk menyimpan semua ulasan
all_reviews = []

# Scrape ulasan untuk setiap aplikasi
for app_id in app_ids:
    try:
        print(f"Scraping ulasan untuk aplikasi: {app_id}")
        
        # Dapatkan semua ulasan aplikasi (bahasa Indonesia)
        result = reviews_all(
            app_id,
            lang='id',  # Bahasa Indonesia
            country='id',  # Indonesia
            sort=Sort.NEWEST  # Urutkan dari yang terbaru
        )
        
        print(f"Jumlah ulasan yang berhasil discrape: {len(result)}")
        
        # Tambahkan info aplikasi
        app_reviews = []
        for review in result:
            review_data = {
                'app_id': app_id,
                'user_name': review['userName'],
                'score': review['score'],
                'content': review['content'],
                'thumbs_up_count': review['thumbsUpCount'],
                'review_date': review['at']
            }
            app_reviews.append(review_data)
        
        all_reviews.extend(app_reviews)
        
        # Jeda random untuk menghindari rate limiting
        time.sleep(random.uniform(2, 5))
        
    except Exception as e:
        print(f"Error pada aplikasi {app_id}: {e}")
        continue

# Buat DataFrame
reviews_df = pd.DataFrame(all_reviews)

# Simpan ke file CSV
reviews_df.to_csv('playstore_reviews_dataset.csv', index=False)
print(f"Total ulasan berhasil discrape: {len(reviews_df)}")
print(f"Data disimpan ke file 'playstore_reviews_dataset.csv'")