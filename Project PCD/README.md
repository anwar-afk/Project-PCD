# Peningkatan Kualitas Citra Gelap dan Perbandingan Operator Deteksi Tepi

Proyek ini mengimplementasikan eksperimen untuk meningkatkan kualitas citra low-light menggunakan Histogram Equalization dan CLAHE, serta membandingkan operator deteksi tepi (Sobel, Prewitt, LoG, dan Canny) pada berbagai tingkat noise (Gaussian dan Salt & Pepper). Evaluasi dilakukan menggunakan MSE dan PSNR terhadap citra referensi bersih.

Instruksi singkat:

- Pasang dependensi:

```powershell
python -m pip install -r requirements.txt
```

- Letakkan dataset citra (jpg/png) dalam folder `dataset/` di root proyek. Jika tidak ada, skrip akan menggunakan citra contoh dari scikit-image.

- Jalankan eksperimen (demo mode) dengan:

```powershell
python src/run_experiment.py --dataset dataset --out results
```

Hasil (gambar dan `results/metrics.csv`) akan disimpan di folder `results/`.

Struktur proyek:

- `src/processing.py` — fungsi preprocessing, noise, enhancement
- `src/utils.py` — operator deteksi tepi, thresholding, metrik
- `src/run_experiment.py` — runner untuk eksperimen dan penyimpanan hasil

Silakan lihat file-file di `src/` untuk detail penggunaan fungsi dan parameter.