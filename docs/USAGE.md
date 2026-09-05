# Panduan Penggunaan Pipeline Spatial Ecoacoustic Analysis (SEA)

Dokumen ini adalah petunjuk operasional teknis untuk menjalankan pipeline pengolahan sinyal akustik multi-kanal dan deteksi spesies burung berbasis unit **MAARU (Multichannel Acoustic Autonomous Recording Unit)** pada data hutan hujan tropis Way Canguk, Lampung, Indonesia.

---

## 1. Ikhtisar Arsitektur Pipeline

Pipeline SEA dirancang modular, praktikal, dan deterministik (*KISS principle*), mengacu pada metodologi publikasi konferensi internasional:  
*“Enhanced birdcall Detection Using Multidirectional Beamforming and Automated Source Selection in Low SNR Soundscapes”*.

Pipeline terdiri dari 4 modul independen di dalam folder `src/`:

```text
[Raw FLAC 6-Channel]
        │
        ▼
[01_render_signals.py]   --> Render Mono, SA, LabIR (19 beam), SPIR (31 beam) ke WAV
        │                    (Onset-aligned 3D impulse response steering vectors)
        ▼
[02_birdnet_infer.py]    --> Batch inference BirdNET multi-processing (min_conf=0.0)
        │                    Output: results.json
        ▼
[03_extract_detections.py] --> Automated Source Selection per (spesies, start_time)
        │                    Output: processed.json (winning beam & max confidence)
        ▼
[04_pair_and_recap.py]   --> Paired comparison & sweep multi-threshold
                             Output: paired_detections.json, summary table (CSV & MD)
```

---

## 2. Struktur Penyimpanan di CX3 HPC

Untuk mengoptimalkan kuota dan mengantisipasi kebijakan penghapusan otomatis 30 hari cluster, alokasi direktori dibagi secara tegas:

| Path | Lokasi Fisik | Karakteristik | Peran |
|---|---|---|---|
| **Raw Audio** | `$EPHEM/monitoring_data/` | Read-only | Berkas audio mentah multi-kanal `.flac` dari unit MAARU di Way Canguk. |
| **Master IR** | `$HOME/MAARU-Impulse-Response/` | Permanen | Berkas kalibrasi *Impulse Response* (`Lab_IR/`, `SP_IR1/`, `SP_IR2/`). |
| **Scratch Audio** | `$EPHEM/sea-scratch/` | Ephemeral (30 hari) | Berkas perantara WAV hasil renderan mono, sa, dan beamformed. Aman terhapus. |
| **Final Output** | `$HOME/spatial-ecoacoustic-analysis/output/` | Permanen | Hasil abstraksi bernilai intelektual (`processed.json`, `paired_detections.json`, tabel rekap, grafik). |

---

## 3. Persiapan Lingkungan

Di cluster CX3 Imperial:
```bash
# Muat environment conda 'sea'
source ~/miniforge3/bin/activate sea

# Pastikan binary ffmpeg dari environment sea terbaca di PATH
export PATH=~/miniforge3/envs/sea/bin:$PATH
```

---

## 4. Cara Penggunaan Mandiri per Modul

### Modul 1: Signal Processing & Beamforming
Mengonversi FLAC 6-kanal menjadi 52 berkas WAV (1 Mono, 1 SA, 19 LabIR, 31 SPIR) dengan koreksi onset *direct arrival*:

```bash
python src/01_render_signals.py \
    --location 2A400 \
    --date 2026-04-22 \
    --max-files 1
```
*Argumen:*
* `--location`: Kode unit (misal `2A400` atau `2D400`).
* `--date`: Tanggal rekaman (`YYYY-MM-DD`).
* `--max-files`: Batas berkas yang dirender (`0` untuk memproses seluruh tanggal).
* `--file-pattern`: Menyaring berkas spesifik (opsional).

### Modul 2: BirdNET Batch Inference
Menjalankan inferensi BirdNET-Analyzer secara paralel pada seluruh berkas WAV di folder rekaman:

```bash
python src/02_birdnet_infer.py \
    $EPHEM/sea-scratch/2A400/2026-04-22/00-02-33_dur=240secs \
    --processes 4
```
*Output:* Menghasilkan berkas `results.json` di dalam direktori rekaman tersebut.

### Modul 3: Automated Source Selection
Mengekstrak deteksi unik per pasangan `(species_name, start_time)`, memilih arah beam pemenang, dan mencatat confidence tertingginya:

```bash
python src/03_extract_detections.py \
    $EPHEM/sea-scratch/2A400/2026-04-22/00-02-33_dur=240secs/results.json \
    --conf-thresh 0.0
```
*Output:* Menghasilkan berkas `processed.json`.

### Modul 4: Paired Comparison & Multi-Threshold Analysis
Memasangkan deteksi Mono vs Beamforming per spesies dan merekap hitungan deteksi pada rentang ambang batas ($\tau = 0{,}30 \dots 0{,}80$):

```bash
python src/04_pair_and_recap.py \
    $EPHEM/sea-scratch/2A400/2026-04-22/00-02-33_dur=240secs/processed.json \
    --out-dir $HOME/spatial-ecoacoustic-analysis/output/2A400/2026-04-22/00-02-33_dur=240secs \
    --thresholds 0.3,0.4,0.5,0.6,0.65,0.7,0.8
```
*Output:* Menghasilkan `paired_detections.json`, `threshold_summary.json`, dan tabel ringkasan `threshold_summary.md`.

---

## 5. Menjalankan Pipeline Skala Penuh per Tanggal

Untuk memproses satu tanggal penuh (atau membatasi jumlah berkas untuk pengujian):

```bash
python src/run_date_pipeline.py \
    --location 2A400 \
    --date 2026-04-22 \
    --processes 8
```

Atau serahkan ke antrean PBS klaster:
```bash
qsub sea-jobs/run_date_pipeline.pbs
```

---

## 6. Struktur Berkas Hasil Evaluasi

Contoh keluaran tabel multi-threshold pada rekaman 4 menit Way Canguk:

| Threshold | Mono Detections | SA Detections | LabIR Detections | SPIR Detections | BF Gain vs Mono (%) |
|---|---|---|---|---|---|
| **0.30** | 8 | 5 | 28 | 55 | **+587.5%** |
| **0.40** | 2 | 3 | 4 | 15 | **+650.0%** |
| **0.50** | 0 | 2 | 1 | 5 | **∞ (Mono = 0)** |
| **0.60** | 0 | 1 | 1 | 2 | **∞ (Mono = 0)** |
| **0.65** | 0 | 0 | 0 | 1 | **∞ (Mono = 0)** |
