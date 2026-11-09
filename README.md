# Proyek Akhir: Menyelesaikan Permasalahan Institusi Pendidikan (Jaya Jaya Institut)

- Nama: Adam Mukharil Bachtiar
- Email: adam@email.unikom.ac.id
- Id Dicoding: adammb86

## Business Understanding

Jaya Jaya Institut adalah institusi pendidikan tinggi yang berdiri sejak tahun 2000 dan sudah menghasilkan banyak lulusan dengan reputasi baik. Di sisi lain, cukup banyak mahasiswanya yang tidak menyelesaikan studi alias dropout. Dari data yang tersedia, sekitar 32% mahasiswa berakhir dropout, atau hampir satu dari tiga mahasiswa.

Dropout yang tinggi berdampak ke banyak hal: reputasi dan akreditasi institusi turun, pendapatan dari biaya kuliah hilang, dan mahasiswa sendiri kehilangan waktu serta biaya. Karena itu institusi ingin mengenali mahasiswa yang berisiko dropout sedini mungkin supaya bisa segera diberi bimbingan khusus.

### Permasalahan Bisnis

1. Berapa besar tingkat dropout saat ini, dan program studi mana yang paling banyak menyumbang dropout?
2. Faktor apa saja (akademik, finansial, dan demografis) yang paling berkaitan dengan dropout?
3. Bagaimana cara mendeteksi mahasiswa yang berisiko dropout sejak dini, termasuk mahasiswa yang saat ini masih aktif (Enrolled)?
4. Bagaimana manajemen bisa memantau performa dan risiko dropout mahasiswa secara rutin tanpa harus membaca data mentah?

### Cakupan Proyek

1. **Data understanding dan EDA** untuk menemukan faktor yang paling berhubungan dengan dropout.
2. **Data preparation**: target biner (Dropout vs Graduate), pemisahan mahasiswa Enrolled sebagai data yang akan diprediksi, seleksi 17 fitur, dan split data latih/uji secara stratified.
3. **Modeling dan evaluasi**: membandingkan Logistic Regression, Random Forest, dan Gradient Boosting, lalu tuning Random Forest dengan GridSearchCV.
4. **Business dashboard** di Metabase untuk memantau tingkat dropout, faktor penyebabnya, dan daftar mahasiswa aktif yang berisiko tinggi.
5. **Prototype machine learning** berbasis Streamlit yang bisa dipakai petugas akademik untuk memprediksi risiko dropout satu mahasiswa atau banyak mahasiswa sekaligus (unggah CSV), lalu di-deploy ke Streamlit Community Cloud.
6. **Kesimpulan dan rekomendasi action items** untuk Jaya Jaya Institut.

### Persiapan

Sumber data: [Students' Performance - Dicoding Academy](https://github.com/dicodingacademy/dicoding_dataset/tree/main/students_performance). Dataset berisi 4.424 mahasiswa dengan 36 fitur (jalur masuk, demografi, kondisi sosial ekonomi, dan performa akademik semester 1 dan 2) serta kolom target `Status` (Dropout, Enrolled, Graduate). Salinan dataset juga disertakan di `data/data.csv`.

Struktur folder submission:

```
submission
├── model
│   ├── rf_dropout_model.joblib     # model Random Forest
│   └── model_metadata.json         # daftar fitur, threshold risiko, metrik
├── data
│   ├── data.csv                    # dataset asli
│   ├── students.db                 # database SQLite untuk dashboard Metabase
│   └── students_labeled.csv        # data berlabel (hasil notebook)
├── notebook.ipynb
├── app.py
├── README.md
├── adammb86-dashboard.png
├── metabase.db.mv.db
└── requirements.txt
```

Setup environment (Python 3.11 ke atas, diuji dengan Python 3.13):

```
# 1. Buat dan aktifkan virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Opsional) jalankan ulang notebook untuk melatih model dan membuat data dashboard
pip install jupyter
jupyter notebook notebook.ipynb
```

Menjalankan notebook akan menghasilkan ulang `model/rf_dropout_model.joblib`, `model/model_metadata.json`, dan `data/students.db`.

## Business Dashboard

Dashboard dibuat dengan **Metabase** (v0.65) dengan judul **"Dashboard Monitoring Dropout Mahasiswa - Jaya Jaya Institut"**. Screenshot lengkapnya ada di `adammb86-dashboard.png`.

Isi dashboard dibagi menjadi empat bagian:

1. **Ringkasan performa mahasiswa**: total mahasiswa (4.424), jumlah dropout (1.421), tingkat dropout (32,1%), tingkat kelulusan (49,9%), jumlah mahasiswa aktif berisiko tinggi (292), komposisi status, dan tingkat dropout per program studi.
2. **Faktor finansial dan demografis**: tingkat dropout berdasarkan status pembayaran UKT, status utang, status beasiswa, gender, dan kelompok usia saat masuk.
3. **Faktor akademik tahun pertama**: proporsi status berdasarkan jumlah unit kurikuler yang lulus di semester 1 dan 2, serta rata-rata nilai semester 1 dan 2 per status.
4. **Deteksi dini**: hasil prediksi model untuk 794 mahasiswa yang masih Enrolled, berupa sebaran tingkat risiko, jumlah mahasiswa berisiko tinggi per program studi, dan tabel 50 mahasiswa dengan prioritas pendampingan tertinggi.

Dashboard punya filter **Program Studi** sehingga kaprodi bisa melihat kondisi prodinya sendiri.

Akses Metabase:

- Email: `root@mail.com`
- Password: `root123`

Cara menjalankan dashboard secara lokal dengan Docker (jalankan dari folder submission):

```
docker run -d -p 3000:3000 --name metabase \
  -v "$(pwd)/metabase.db.mv.db:/metabase.db/metabase.db.mv.db" \
  -v "$(pwd)/data:/metabase-data" \
  metabase/metabase:v0.65.3.4
```

Lalu buka `http://localhost:3000`, login dengan akun di atas, dan pilih koleksi **Jaya Jaya Institut - Student Dropout**. Mount kedua (`/metabase-data`) diperlukan karena sumber data dashboard adalah database SQLite `data/students.db` yang dibaca Metabase dari path `/metabase-data/students.db`.

## Menjalankan Sistem Machine Learning

Prototype dibuat dengan Streamlit (`app.py`) dan memakai model Random Forest di folder `model`. Aplikasinya punya tiga tab:

1. **Prediksi satu mahasiswa**: petugas mengisi 17 data (program studi, jalur masuk, usia, nilai masuk, status UKT, utang, beasiswa, serta unit diambil, evaluasi, unit lulus, dan nilai semester 1 dan 2). Aplikasi menampilkan probabilitas dropout, tingkat risiko (Tinggi >= 0,7; Sedang 0,4 sampai 0,7; Rendah < 0,4), rekomendasi tindakan, dan sinyal peringatan yang terdeteksi.
2. **Prediksi banyak mahasiswa (CSV)**: unggah CSV dengan format kolom seperti dataset asli (pemisah `;` atau `,`). Aplikasi menampilkan ringkasan risiko, grafik sebaran risiko per program studi, tabel hasil yang diurutkan dari risiko tertinggi, dan tombol unduh hasil prediksi. Template CSV bisa diunduh dari aplikasi.
3. **Tentang model**: algoritma, parameter, threshold, dan metrik model.

Menjalankan prototype secara lokal:

```
pip install -r requirements.txt
streamlit run app.py
```

Aplikasi akan terbuka di `http://localhost:8501`.

Link prototype di Streamlit Community Cloud: **https://GANTI-DENGAN-URL-APP-ANDA.streamlit.app**

## Conclusion

Dari analisis data dan pemodelan, ada beberapa jawaban untuk permasalahan Jaya Jaya Institut:

1. **Tingkat dropout tinggi dan tidak merata antarprodi.** Tingkat dropout keseluruhan 32,1%. Prodi dengan dropout tertinggi (di luar Biofuel Production Technologies yang hanya 12 mahasiswa) adalah Equinculture (55%), Informatics Engineering (54%), Management kelas malam (51%), dan Basic Education (44%). Nursing (15%) dan Social Service (18%) paling rendah.
2. **Performa akademik tahun pertama adalah prediktor terkuat.** Mahasiswa yang tidak meluluskan satu pun unit kurikuler di semester 1 hampir 80% berakhir dropout, sedangkan yang meluluskan 6 unit atau lebih hanya sekitar 10%. Rata-rata nilai semester 2 mahasiswa dropout hanya 5,9, dibanding 12,7 pada lulusan.
3. **Masalah finansial sangat berkaitan dengan dropout.** Mahasiswa yang UKT-nya tidak lunas tepat waktu dropout 86,6% (vs 24,7%), mahasiswa yang punya utang dropout 62% (vs 28%), sementara penerima beasiswa hanya 12,2% (vs 38,7%).
4. **Faktor demografis ikut berperan.** Mahasiswa yang masuk di usia 25 tahun ke atas dropout lebih dari 50%, dan mahasiswa laki-laki dropout 45% dibanding perempuan 25%.
5. **Dropout bisa diprediksi dengan akurat.** Model Random Forest dengan 17 fitur mencapai akurasi 91,7%, precision 0,889, recall 0,901, F1 0,895, dan ROC AUC 0,968 pada data uji. Artinya sekitar 9 dari 10 mahasiswa yang akan dropout bisa terdeteksi. Model yang hanya memakai data sampai semester 1 pun masih memberi F1 0,86, sehingga skrining bisa dimulai setelah semester pertama.
6. **Ada 292 mahasiswa aktif yang perlu segera ditangani.** Dari 794 mahasiswa Enrolled, 292 (sekitar 37%) diprediksi berisiko tinggi dropout, terbanyak di Management (59) dan Informatics Engineering (47).

### Rekomendasi Action Items

- **Jalankan skrining risiko dropout tiap akhir semester.** Gunakan prototype untuk menilai seluruh mahasiswa aktif setelah nilai semester 1 dan semester 2 keluar, lalu teruskan daftar risiko tinggi ke dosen wali. Mulai dari 292 mahasiswa Enrolled berisiko tinggi yang sudah teridentifikasi.
- **Buat program pendampingan akademik untuk mahasiswa tahun pertama.** Mahasiswa yang meluluskan kurang dari setengah unit yang diambil atau nilainya di bawah 10 di semester 1 perlu tutor sebaya, kelas remedial, dan jadwal konsultasi rutin dengan dosen wali sebelum semester 2 berjalan.
- **Tangani kendala finansial secara proaktif.** Bagian keuangan perlu memberi tanda otomatis ketika UKT mahasiswa terlambat atau muncul tunggakan, lalu menawarkan skema cicilan, keringanan, atau beasiswa bantuan. Data menunjukkan beasiswa berkaitan dengan dropout yang jauh lebih rendah, jadi kuota beasiswa sebaiknya diprioritaskan untuk mahasiswa berprestasi yang terkendala biaya.
- **Beri perhatian khusus pada prodi dengan dropout tinggi.** Equinculture, Informatics Engineering, Management kelas malam, dan Basic Education perlu evaluasi kurikulum tahun pertama, beban mata kuliah, dan metode pengajaran. Kaprodi bisa memantau kondisi prodinya lewat filter Program Studi di dashboard.
- **Siapkan dukungan untuk mahasiswa usia dewasa dan kelas malam.** Mahasiswa yang masuk di usia 25 tahun ke atas umumnya sudah bekerja. Jadwal yang lebih fleksibel, materi daring, dan konseling manajemen waktu bisa membantu mereka bertahan.
- **Jadikan dashboard sebagai alat monitoring rutin.** Manajemen sebaiknya meninjau dashboard minimal sekali per bulan, dan data dashboard serta model diperbarui setiap semester agar tetap relevan.
