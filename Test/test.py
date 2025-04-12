# %% [markdown]
# # Import Library

# %%
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from joblib import Parallel, delayed
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from imblearn.over_sampling import SMOTE
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, SpatialDropout1D, Input, concatenate, Attention, Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, BatchNormalization, Layer
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras import initializers, activations
import gensim.downloader as api
import joblib
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import pickle
import re
import seaborn as sns
import tensorflow as tf
import time
import warnings

warnings.filterwarnings('ignore')

# %% [markdown]
# # Sentiment Analysis Workflow
# Notebook ini mencakup:
# - Preprocessing dan pelabelan otomatis dari data ulasan Play Store
# - Pelatihan dan evaluasi model analisis sentimen
# - Inference prediksi sentimen baru

# %%
# Unduh resource NLTK
nltk.download('stopwords')

# Load data
data = pd.read_csv('playstore_reviews_full.csv')
print(f"Jumlah data awal: {len(data)}")

# === CLEANING ===
def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ""

# Stemming Setup
factory = StemmerFactory()
stemmer_id = factory.create_stemmer()
stemmer_en = PorterStemmer()

stop_words_en = set(stopwords.words('english'))
stop_words_id = set([
    'yang', 'dan', 'di', 'dari', 'ke', 'pada', 'untuk', 'dengan', 'adalah',
    'ini', 'itu', 'atau', 'juga', 'saya', 'kamu', 'dia', 'mereka', 'kita',
    'kami', 'ada', 'tidak', 'akan', 'bisa', 'karena', 'jadi'
])
all_stop_words = stop_words_en.union(stop_words_id)

def stem_text(text):
    if isinstance(text, str):
        words = text.split()
        filtered_words = [word for word in words if word not in all_stop_words]
        stemmed_words = [stemmer_id.stem(word) for word in filtered_words]
        return ' '.join(stemmed_words)
    return ""

def label_by_rating(score):
    try:
        score = float(score)
        if score <= 2:
            return 'negative'
        elif score == 3:
            return 'neutral'
        else:
            return 'positive'
    except:
        return None

# === APPLY CLEANING ===
print("Cleaning...")
data['cleaned_text'] = data['content'].apply(clean_text)

# === APPLY STEMMING WITH PARALLEL ===
print("Stemming with parallel processing...")
data['processed_text'] = Parallel(n_jobs=-1)(
    delayed(stem_text)(text) for text in data['cleaned_text']
)

# === LABELING ===
print("Melakukan pelabelan...")
data['sentiment_label'] = data['score'].apply(label_by_rating)
data = data.dropna(subset=['sentiment_label'])

label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
data['sentiment_score'] = data['sentiment_label'].map(label_map)

# === PENANGANAN DATA TIDAK SEIMBANG DENGAN SMOTE ===
print("\nMemeriksa distribusi kelas...")
class_distribution = data['sentiment_score'].value_counts()
print("\nDistribusi kelas sebelum balancing:")
print(class_distribution)

print("\nMenangani ketidakseimbangan data dengan SMOTE...")

# Menyiapkan data
X = data['processed_text']
y = data['sentiment_score']

# Mengubah data teks menjadi fitur numerik menggunakan TF-IDF
print("\nMengkonversi teks ke fitur TF-IDF...")
vectorizer = TfidfVectorizer(max_features=1000)
X_tfidf = vectorizer.fit_transform(X)

# Menerapkan SMOTE
print("\nMenerapkan SMOTE...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_tfidf, y)

print(f"\nBentuk asli: {X_tfidf.shape}, Bentuk setelah resampling: {X_resampled.shape}")
print("\nDistribusi kelas setelah SMOTE:")
unique, counts = np.unique(y_resampled, return_counts=True)
print(dict(zip(unique, counts)))

# Membuat dataset yang seimbang
print("\nMembuat dataset yang seimbang...")

# Menyimpan indeks asli untuk setiap kelas
original_indices = {}
for class_val in np.unique(y):
    original_indices[class_val] = np.where(y == class_val)[0]

# Membuat dataset seimbang dengan memilih sampel asli untuk setiap kelas
balanced_data = pd.DataFrame()

for class_val in np.unique(y_resampled):
    # Menghitung berapa banyak sampel yang dibutuhkan untuk kelas ini
    class_count = sum(y_resampled == class_val)
    
    # Mendapatkan sampel asli untuk kelas ini
    class_samples = data[data['sentiment_score'] == class_val]
    
    # Jika kita butuh lebih banyak sampel daripada yang kita miliki, lakukan oversampling dengan penggantian
    if len(class_samples) < class_count:
        class_samples = class_samples.sample(n=class_count, replace=True, random_state=42)
    # Jika kita punya lebih banyak dari yang dibutuhkan, lakukan undersampling tanpa penggantian
    else:
        class_samples = class_samples.sample(n=class_count, replace=False, random_state=42)
    
    balanced_data = pd.concat([balanced_data, class_samples])

# Acak data yang seimbang
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Periksa distribusi baru
print("\nDistribusi kelas dalam dataset yang telah diseimbangkan:")
print(balanced_data['sentiment_score'].value_counts())

# Ganti data lama dengan data yang telah diseimbangkan
data = balanced_data

# === SIMPAN HASIL ===
data.to_csv('processed_sentiment_dataset.csv', index=False)
print(f"\nDisimpan ke 'processed_sentiment_dataset.csv'")
print(f"\nJumlah data awal: {len(data)}, Jumlah data setelah balancing: {len(balanced_data)}")

# %% [markdown]
# # Analisis Sentimen - Training Model

# %% [markdown]
# ### Menyiapkan Data

# %%
# Unduh resource NLTK yang diperlukan
nltk.download('punkt')
nltk.download('punkt_tab')

# Muat dataset yang telah diproses
print("Memuat dataset...")
data = pd.read_csv('processed_sentiment_dataset.csv')
print(f"Dataset dimuat dengan {len(data)} baris")

# Periksa distribusi label sentimen
print("\nDistribusi label sentimen:")
print(data['sentiment_label'].value_counts())
print(data['sentiment_label'].value_counts(normalize=True) * 100)

# Gambaran data
print("\nContoh Data:")
print(data[['app_name', 'processed_text', 'sentiment_label']].head())

# Periksa nilai yang hilang
nilai_hilang = data.isnull().sum()
print("\nNilai yang hilang:")
print(nilai_hilang[nilai_hilang > 0])

# Hapus nilai yang hilang yang tersisa
data = data.dropna(subset=['processed_text', 'sentiment_label'])
print(f"Dataset setelah menghapus nilai yang hilang: {len(data)} baris")

# Ubah variabel target menjadi numerik
peta_label = {'negative': 0, 'neutral': 1, 'positive': 2}
data['sentiment_score'] = data['sentiment_label'].map(peta_label)

# Buat variabel fitur dan target
X = data['processed_text']
y = data['sentiment_score']

# Bagi data - Kita akan mencoba pembagian yang berbeda sesuai pedoman pengajuan
X_train_80, X_test_20, y_train_80, y_test_20 = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train_70, X_test_30, y_train_70, y_test_30 = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Pembagian 80/20 - Set pelatihan: {len(X_train_80)}, Set pengujian: {len(X_test_20)}")
print(f"Pembagian 70/30 - Set pelatihan: {len(X_train_70)}, Set pengujian: {len(X_test_30)}")

# Tentukan fungsi untuk mengevaluasi dan menampilkan kinerja model
def evaluasi_model(y_true, y_pred, nama_model):
    akurasi = accuracy_score(y_true, y_pred)
    print(f"Akurasi {nama_model}: {akurasi * 100:.2f}%")
    
    print("\nLaporan Klasifikasi:")
    print(classification_report(y_true, y_pred, target_names=list(peta_label.keys())))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=list(peta_label.keys()), 
                yticklabels=list(peta_label.keys()))
    plt.xlabel('Diprediksi')
    plt.ylabel('Sebenarnya')
    plt.title(f'Confusion Matrix - {nama_model}')
    plt.tight_layout()
    plt.show()
    
    return akurasi

# Simpan hasil untuk perbandingan
hasil = []

# %% [markdown]
# ### EKSPERIMEN 1: TF-IDF + SVM dengan pembagian 80/20

# %%
print("\n" + "="*50)
print("EKSPERIMEN 1: TF-IDF + SVM dengan pembagian 80/20")
print("="*50)

# Buat pipeline
tfidf_svm_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
    ('svm', LinearSVC(C=1.0))
])

# Latih model
waktu_mulai = time.time()
tfidf_svm_pipeline.fit(X_train_80, y_train_80)
waktu_pelatihan = time.time() - waktu_mulai
print(f"Waktu pelatihan: {waktu_pelatihan:.2f} detik")

# Prediksi pada set pelatihan dan pengujian
y_train_pred = tfidf_svm_pipeline.predict(X_train_80)
y_test_pred = tfidf_svm_pipeline.predict(X_test_20)

# Evaluasi model
print("\nEvaluasi set pelatihan:")
akurasi_train = evaluasi_model(y_train_80, y_train_pred, "TF-IDF + SVM (Pelatihan)")

print("\nEvaluasi set pengujian:")
akurasi_test = evaluasi_model(y_test_20, y_test_pred, "TF-IDF + SVM (Pengujian)")

# Simpan hasil
hasil.append({
    'eksperimen': 'TF-IDF + SVM (80/20)',
    'ekstraksi_fitur': 'TF-IDF',
    'model': 'SVM',
    'pembagian': '80/20',
    'akurasi_train': akurasi_train,
    'akurasi_test': akurasi_test,
    'waktu_pelatihan': waktu_pelatihan
})

# Simpan model
joblib.dump(tfidf_svm_pipeline, 'tfidf_svm_80_20_model.joblib')
print("Model disimpan sebagai 'tfidf_svm_80_20_model.joblib'")

# %% [markdown]
# ### EKSPERIMEN 2: Word2Vec + Random Forest dengan pembagian 80/20

# %%
print("\n" + "="*50)
print("EKSPERIMEN 2: Word2Vec + Random Forest dengan pembagian 80/20")
print("="*50)

# Muat model Word2Vec yang sudah dilatih sebelumnya
print("Memuat model Word2Vec...")
word2vec_model = api.load('word2vec-google-news-300')
print("Model Word2Vec dimuat")

# Fungsi untuk membuat vektor dokumen dari Word2Vec
def vektor_dokumen(doc, model, ukuran_vektor=300):
    # Inisialisasi vektor nol
    vektor_doc = np.zeros(ukuran_vektor)
    jumlah_kata = 0
    
    # Tokenisasi dokumen
    if isinstance(doc, str):
        kata_kata = word_tokenize(doc.lower())
        
        # Jumlahkan vektor untuk semua kata dalam dokumen
        for kata in kata_kata:
            try:
                # Tambahkan vektor kata jika ada dalam model
                vektor_doc += model[kata]
                jumlah_kata += 1
            except KeyError:
                # Lewati kata yang tidak ada dalam kosakata
                continue
                
        # Hitung vektor rata-rata jika ada kata yang ditemukan
        if jumlah_kata > 0:
            vektor_doc /= jumlah_kata
            
    return vektor_doc

# Buat vektor dokumen untuk set pelatihan dan pengujian
print("Membuat vektor dokumen...")
X_train_w2v = np.array([vektor_dokumen(doc, word2vec_model) for doc in X_train_80])
X_test_w2v = np.array([vektor_dokumen(doc, word2vec_model) for doc in X_test_20])
print("Vektor dokumen telah dibuat")

# Latih Random Forest
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)

waktu_mulai = time.time()
model_rf.fit(X_train_w2v, y_train_80)
waktu_pelatihan = time.time() - waktu_mulai
print(f"Waktu pelatihan: {waktu_pelatihan:.2f} detik")

# Prediksi pada set pelatihan dan pengujian
y_train_pred = model_rf.predict(X_train_w2v)
y_test_pred = model_rf.predict(X_test_w2v)

# Evaluasi model
print("\nEvaluasi set pelatihan:")
akurasi_train = evaluasi_model(y_train_80, y_train_pred, "Word2Vec + RF (Pelatihan)")

print("\nEvaluasi set pengujian:")
akurasi_test = evaluasi_model(y_test_20, y_test_pred, "Word2Vec + RF (Pengujian)")

# Simpan hasil
hasil.append({
    'eksperimen': 'Word2Vec + RF (80/20)',
    'ekstraksi_fitur': 'Word2Vec',
    'model': 'Random Forest',
    'pembagian': '80/20',
    'akurasi_train': akurasi_train,
    'akurasi_test': akurasi_test,
    'waktu_pelatihan': waktu_pelatihan
})

# Simpan model dan vektorisasi
joblib.dump(model_rf, 'w2v_rf_80_20_model.joblib')
print("Model disimpan sebagai 'w2v_rf_80_20_model.joblib'")

# Simpan juga fungsi vektor dokumen untuk inferensi
with open('fungsi_vektor_dokumen_w2v.pkl', 'wb') as f:
    pickle.dump(vektor_dokumen, f)
print("Fungsi vektor dokumen disimpan sebagai 'fungsi_vektor_dokumen_w2v.pkl'")

# %% [markdown]
# ### EKSPERIMEN 3: TF-IDF + Random Forest dengan pembagian 70/30

# %%
print("\n" + "="*50)
print("EKSPERIMEN 3: TF-IDF + Random Forest dengan pembagian 70/30")
print("="*50)

# Buat fitur TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_70)
X_test_tfidf = tfidf_vectorizer.transform(X_test_30)

# Latih Random Forest
model_rf_tfidf = RandomForestClassifier(n_estimators=100, random_state=42)

waktu_mulai = time.time()
model_rf_tfidf.fit(X_train_tfidf, y_train_70)
waktu_pelatihan = time.time() - waktu_mulai
print(f"Waktu pelatihan: {waktu_pelatihan:.2f} detik")

# Prediksi pada set pelatihan dan pengujian
y_train_pred = model_rf_tfidf.predict(X_train_tfidf)
y_test_pred = model_rf_tfidf.predict(X_test_tfidf)

# Evaluasi model
print("\nEvaluasi set pelatihan:")
akurasi_train = evaluasi_model(y_train_70, y_train_pred, "TF-IDF + RF (Pelatihan)")

print("\nEvaluasi set pengujian:")
akurasi_test = evaluasi_model(y_test_30, y_test_pred, "TF-IDF + RF (Pengujian)")

# Simpan hasil
hasil.append({
    'eksperimen': 'TF-IDF + RF (70/30)',
    'ekstraksi_fitur': 'TF-IDF',
    'model': 'Random Forest',
    'pembagian': '70/30',
    'akurasi_train': akurasi_train,
    'akurasi_test': akurasi_test,
    'waktu_pelatihan': waktu_pelatihan
})

# Simpan model dan vektorisasi
joblib.dump(model_rf_tfidf, 'tfidf_rf_70_30_model.joblib')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer_70_30.joblib')
print("Model disimpan sebagai 'tfidf_rf_70_30_model.joblib'")
print("Vektorisasi disimpan sebagai 'tfidf_vectorizer_70_30.joblib'")

# %% [markdown]
# ### EKSPERIMEN 4: Deep Learning - LSTM dengan pembagian 80/20

# %%
print("\n" + "="*50)
print("EKSPERIMEN 4: Deep Learning - LSTM dengan pembagian 80/20")
print("="*50)

# Tokenisasi teks
max_features = 10000  # Kata teratas yang dipertimbangkan
max_len = 100  # Panjang urutan maksimum

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(X_train_80)

X_train_seq = tokenizer.texts_to_sequences(X_train_80)
X_test_seq = tokenizer.texts_to_sequences(X_test_20)

# Pad urutan
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

# Ubah target menjadi kategorikal
y_train_cat = to_categorical(y_train_80, num_classes=3)
y_test_cat = to_categorical(y_test_20, num_classes=3)

# Buat model LSTM menggunakan Functional API
inputs = tf.keras.Input(shape=(max_len,))
x = Embedding(max_features, 300, input_length=max_len)(inputs)
x = SpatialDropout1D(0.25)(x)
x = Bidirectional(LSTM(150, dropout=0.25, recurrent_dropout=0.25, return_sequences=True))(x)

# Add Attention layer with proper inputs
attention_output = Attention()([x, x])  # Pass query and value to the Attention layer

# Add a pooling layer to reduce the sequence dimension
x = GlobalAveragePooling1D()(attention_output)

# Add a pooling layer to reduce the sequence dimension
x = GlobalAveragePooling1D()(attention_output)

x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = Dense(64, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
outputs = Dense(3, activation='softmax')(x)

# Define the model
model_lstm = tf.keras.Model(inputs=inputs, outputs=outputs)

# Kompilasi model dengan learning rate yang disesuaikan
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
model_lstm.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Build model sebelum summary
model_lstm.build((None, max_len))
model_lstm.summary()

# Penghentian dini tetap sama
penghentian_dini = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Tambahkan callback untuk mengurangi learning rate
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

# Latih model dengan batch size yang lebih kecil
waktu_mulai = time.time()
history = model_lstm.fit(
    X_train_pad, y_train_cat,
    epochs=15,  # Lebih banyak epoch
    batch_size=32,  # Batch size lebih kecil
    validation_split=0.1,
    callbacks=[penghentian_dini, reduce_lr],
    verbose=1
)
waktu_pelatihan = time.time() - waktu_mulai
print(f"Waktu pelatihan: {waktu_pelatihan:.2f} detik")

# Plot riwayat pelatihan
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Akurasi Model')
plt.ylabel('Akurasi')
plt.xlabel('Epoch')
plt.legend(['Pelatihan', 'Validasi'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss Model')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Pelatihan', 'Validasi'], loc='upper left')
plt.tight_layout()
plt.show()

# Evaluasi pada set pelatihan
y_train_prob = model_lstm.predict(X_train_pad)
y_train_pred = np.argmax(y_train_prob, axis=1)

# Evaluasi pada set pengujian
y_test_prob = model_lstm.predict(X_test_pad)
y_test_pred = np.argmax(y_test_prob, axis=1)

# Ubah kembali dari kategorikal
y_train_true = np.array(y_train_80)
y_test_true = np.array(y_test_20)

# Evaluasi model
print("\nEvaluasi set pelatihan:")
akurasi_train = evaluasi_model(y_train_true, y_train_pred, "LSTM (Pelatihan)")

print("\nEvaluasi set pengujian:")
akurasi_test = evaluasi_model(y_test_true, y_test_pred, "LSTM (Pengujian)")

# Simpan hasil
hasil.append({
    'eksperimen': 'LSTM (80/20)',
    'ekstraksi_fitur': 'Word Embeddings',
    'model': 'Bidirectional LSTM',
    'pembagian': '80/20',
    'akurasi_train': akurasi_train,
    'akurasi_test': akurasi_test,
    'waktu_pelatihan': waktu_pelatihan
})

# Simpan model dan tokenizer
model_lstm.save('lstm_80_20_model.h5')
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Model disimpan sebagai 'lstm_80_20_model.h5'")
print("Tokenizer disimpan sebagai 'tokenizer.pickle'")

# %% [markdown]
# ### EKSPERIMEN 5: Deep Learning - CNN dengan pembagian 80/20

# %%
print("\n" + "="*50)
print("EKSPERIMEN 5: Deep Learning - CNN dengan pembagian 80/20")
print("="*50)

# Buat model CNN menggunakan Functional API
input_layer = Input(shape=(max_len,))
embedding_layer = Embedding(max_features, 300, input_length=max_len)(input_layer)
embedding_layer = SpatialDropout1D(0.25)(embedding_layer)

# Multiple parallel convolutions
filter_sizes = [2, 3, 4, 5]
conv_blocks = []

for sz in filter_sizes:
    conv = Conv1D(filters=256, kernel_size=sz, padding='valid', activation='relu', strides=1)(embedding_layer)
    conv = GlobalMaxPooling1D()(conv)
    conv_blocks.append(conv)

merged = concatenate(conv_blocks, axis=1)
dense = Dense(256, activation='relu')(merged)
dense = BatchNormalization()(dense)
dense = Dropout(0.4)(dense)
dense = Dense(128, activation='relu')(dense)
dense = BatchNormalization()(dense)
dense = Dropout(0.4)(dense)
output = Dense(3, activation='softmax')(dense)

# Membuat model
model_cnn = Model(inputs=input_layer, outputs=output)

# Kompilasi model dengan learning rate yang dioptimalkan
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
model_cnn.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Summary dari model
model_cnn.summary()

penghentian_dini = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Callback untuk mengurangi learning rate
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

# Latih model
waktu_mulai = time.time()
history = model_cnn.fit(
    X_train_pad, y_train_cat,
    epochs=15,  # Lebih banyak epoch
    batch_size=32,  # Batch size lebih kecil
    validation_split=0.1,
    callbacks=[penghentian_dini, reduce_lr],
    verbose=1
)
waktu_pelatihan = time.time() - waktu_mulai
print(f"Waktu pelatihan: {waktu_pelatihan:.2f} detik")

# %% [markdown]
# # RINGKASAN DAN PERBANDINGAN

# %%
print("\n" + "="*50)
print("RINGKASAN PERBANDINGAN MODEL")
print("="*50)

# Buat DataFrame untuk perbandingan
df_hasil = pd.DataFrame(hasil)
print(df_hasil)

# Urutkan berdasarkan akurasi pengujian
df_hasil_diurutkan = df_hasil.sort_values('akurasi_test', ascending=False)
print("\nModel diurutkan berdasarkan akurasi pengujian:")
print(df_hasil_diurutkan)

# Plot perbandingan model
plt.figure(figsize=(12, 6))
sns.barplot(x='eksperimen', y='akurasi_test', data=df_hasil_diurutkan)
plt.title('Perbandingan Model - Akurasi Pengujian')
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# Plot akurasi pelatihan vs pengujian
plt.figure(figsize=(12, 6))
df_hasil_meleleh = pd.melt(df_hasil_diurutkan, 
                            id_vars=['eksperimen'], 
                            value_vars=['akurasi_train', 'akurasi_test'],
                            var_name='Set', value_name='Akurasi')
sns.barplot(x='eksperimen', y='Akurasi', hue='Set', data=df_hasil_meleleh)
plt.title('Perbandingan Model - Akurasi Pelatihan vs Pengujian')
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# Identifikasi model terbaik
model_terbaik = df_hasil_diurutkan.iloc[0]
print("\nModel Terbaik Berdasarkan Akurasi Pengujian:")
print(f"Eksperimen: {model_terbaik['eksperimen']}")
print(f"Ekstraksi Fitur: {model_terbaik['ekstraksi_fitur']}")
print(f"Model: {model_terbaik['model']}")
print(f"Pembagian: {model_terbaik['pembagian']}")
print(f"Akurasi Pelatihan: {model_terbaik['akurasi_train'] * 100:.2f}%")
print(f"Akurasi Pengujian: {model_terbaik['akurasi_test'] * 100:.2f}%")
print(f"Waktu Pelatihan: {model_terbaik['waktu_pelatihan']:.2f} detik")

# %% [markdown]
# # CONTOH INFERENSI

# %%
print("\n" + "="*50)
print("CONTOH INFERENSI")
print("="*50)

# Fungsi untuk melakukan inferensi dengan model terbaik
def prediksi_sentimen(teks, jenis_model='terbaik'):
    # Praproses teks (disederhanakan untuk demonstrasi)
    teks_diproses = teks.lower()
    
    # Pilih model berdasarkan jenis
    if jenis_model == 'tfidf_svm':
        # Muat model TF-IDF + SVM
        model = joblib.load('tfidf_svm_80_20_model.joblib')
        prediksi = model.predict([teks_diproses])[0]
    
    elif jenis_model == 'word2vec_rf':
        # Muat model Word2Vec + RF
        model = joblib.load('w2v_rf_80_20_model.joblib')
        with open('fungsi_vektor_dokumen_w2v.pkl', 'rb') as f:
            fungsi_vektor_doc = pickle.load(f)
        
        # Buat vektor dokumen
        vektor_doc = fungsi_vektor_doc(teks_diproses, word2vec_model)
        prediksi = model.predict([vektor_doc])[0]
    
    elif jenis_model == 'lstm':
        # Muat model LSTM
        model = tf.keras.models.load_model('lstm_80_20_model.h5')
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        
        # Tokenisasi dan pad
        seq = tokenizer.texts_to_sequences([teks_diproses])
        padded = pad_sequences(seq, maxlen=max_len)
        
        # Prediksi
        pred_prob = model.predict(padded)[0]
        prediksi = np.argmax(pred_prob)
    
    elif jenis_model == 'cnn':
        # Muat model CNN
        model = tf.keras.models.load_model('cnn_80_20_model.h5')
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        
        # Tokenisasi dan pad
        seq = tokenizer.texts_to_sequences([teks_diproses])
        padded = pad_sequences(seq, maxlen=max_len)
        
        # Prediksi
        pred_prob = model.predict(padded)[0]
        prediksi = np.argmax(pred_prob)
    
    else:  # model terbaik (tentukan dari hasil)
        # Gunakan model terbaik berdasarkan hasil
        if model_terbaik['eksperimen'] == 'LSTM (80/20)':
            return prediksi_sentimen(teks, 'lstm')
        elif model_terbaik['eksperimen'] == 'CNN (80/20)':
            return prediksi_sentimen(teks, 'cnn')
        elif model_terbaik['eksperimen'] == 'TF-IDF + SVM (80/20)':
            return prediksi_sentimen(teks, 'tfidf_svm')
        else:
            return prediksi_sentimen(teks, 'word2vec_rf')
    
    # Petakan prediksi kembali ke label
    peta_label_terbalik = {0: 'negatif', 1: 'netral', 2: 'positif'}
    sentimen = peta_label_terbalik[prediksi]
    
    return sentimen

# Contoh teks untuk inferensi
teks_uji = [
    "Aplikasi ini sangat membantu dan mudah digunakan!",
    "Layanan sangat buruk, saya kecewa dengan produk ini.",
    "Biasa saja, tidak ada yang special dari layanan ini."
]

# Uji inferensi
print("\nContoh inferensi:")
for teks in teks_uji:
    sentimen = prediksi_sentimen(teks)
    print(f"Teks: '{teks}'")
    print(f"Sentimen yang diprediksi: {sentimen}")
    print("-" * 50)

print("\nAnalisis Sentimen selesai!")


