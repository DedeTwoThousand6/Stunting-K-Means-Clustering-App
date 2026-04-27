from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import time
import os

app = Flask(__name__)

# 1. Inisialisasi Data & Model saat Server Berjalan
df = pd.read_csv('stunting_wasting_dataset.csv')
le = LabelEncoder()
df['JK_Enc'] = le.fit_transform(df['Jenis Kelamin'])

features = ['Umur (bulan)', 'Tinggi Badan (cm)', 'Berat Badan (kg)', 'JK_Enc']
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Logika Sorting: Memastikan cluster dengan tinggi terendah adalah Stunting
cluster_order = df.groupby('Cluster')['Tinggi Badan (cm)'].mean().sort_values().index
kategori_map = {
    cluster_order[0]: "Stunting",
    cluster_order[1]: "Risiko Stunting",
    cluster_order[2]: "Normal"
}

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    # Timestamp untuk mencegah gambar tidak terupdate di browser (Cache)
    ts = int(time.time())
    
    if request.method == 'POST':
        # Mengambil input dari form
        umur = float(request.form['umur'])
        tinggi = float(request.form['tinggi'])
        berat = float(request.form['berat'])
        jk = int(request.form['jk'])
        
        # Prediksi
        new_data = scaler.transform([[umur, tinggi, berat, jk]])
        cluster_pred = kmeans.predict(new_data)[0]
        prediction = kategori_map.get(cluster_pred)

    # Re-generate grafik scatter plot
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    df['Kategori AI'] = df['Cluster'].map(kategori_map)
    sns.scatterplot(data=df, x='Umur (bulan)', y='Tinggi Badan (cm)', 
                    hue='Kategori AI', palette='viridis', s=100, alpha=0.6)
    
    # Tambahkan marker X merah jika ada input baru
    if request.method == 'POST':
        plt.scatter(umur, tinggi, color='red', marker='X', s=250, label='Data Baru')
        plt.legend()

    plt.title('Visualisasi Pengelompokan Status Gizi Balita')
    plt.savefig('static/cluster.png')
    plt.close()

    sample_data = df.head(10).to_dict(orient='records')
    return render_template('index.html', prediction=prediction, data=sample_data, timestamp=ts)

if __name__ == '__main__':
    app.run(debug=True)