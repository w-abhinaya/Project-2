Aplikasi web sederhana untuk mendeteksi objek dalam gambar (Pesawat, Mobil, Burung, Kucing, dll) menggunakan **FastAPI**, **TensorFlow**, dan **Docker**.

## Struktur File
main.py`: Backend API menggunakan FastAPI.
index.html`: Tampilan antarmuka (User Interface).
Dockerfile`: Konfigurasi container Docker.
model.keras`: Model AI yang sudah dilatih.

## Cara Menjalankan (Docker)

Pastikan Docker Desktop sudah terinstall, lalu jalankan perintah berikut di terminal:

```bash
# 1. Build Image
docker build -t cifar10-app .

# 2. Jalankan Container
docker run -p 8000:8000 cifar10-app

Setelah berjalan, buka browser dan akses: http://localhost:8000
