# Spatial Ecoacoustic Analysis — Konteks Kerja di CX3

> **Untuk agent/model yang baru masuk:** baca file ini sampai habis sebelum menyentuh apa pun.
> Isinya aturan penyimpanan, cara menjalankan pipeline, status data, dan daftar masalah yang
> masih terbuka. Diperbarui berkala; kalau kamu mengubah struktur atau menyelesaikan salah satu
> item "Masalah Terbuka", perbarui file ini juga.

**Terakhir diperbarui:** 2026-08-31
**Penulis update terakhir:** sesi Claude Code (Opus 5), dilanjutkan ke Pi agent
**Repo:** `git@github.com:ikhwanuddin/spatial-ecoacoustic-analysis.git` — branch `main`, commit `40ebea5`

**Fokus saat ini: `2026-04-21`.** Tanggal itu yang dikerjakan lebih dulu. Tanggal lain
**tidak dihapus** — datanya lengkap dan tetap di tempatnya, hanya ditunda.

---

## 1. Aturan Penyimpanan (paling penting)

Ephemeral **dihapus otomatis setiap 30 hari**. Tiga kategori, tiga tempat:

| Kategori | Lokasi | Alasan |
|---|---|---|
| **Input / source code** | repo → push ke GitHub | backup software |
| **Output penting, kecil** | `HOME` | harus selamat dari wipe |
| **Raw data & proses kerja** | `ephemeral` | besar, bisa dibuat ulang |

```
HOME  /rds/general/user/ri322/home/
  spatial-ecoacoustic-analysis/     # INPUT saja. Jangan pernah taruh output di sini.
  sea-emb/2A400/
    <model>/<date>_<method>_meta.json
    <model>/<date>_summary.json
    <model>/noise_<group>_embeddings.npy + _meta.json
    audits/<date>_comparison.json|md
    run_reports/
  sea-dashboards/2A400/<date>/      # index.html, <model>.html, narrative_<model>.json
  .openrouter_api_key               # chmod 600, dipakai narrative.py

EPHEMERAL  /rds/general/user/ri322/ephemeral/sea-work/2A400/
  <date>/                           # audio hasil beamforming: bf_LabIR, bf_SPIR, sa, mono
  emb/<model>/<date>_<method>.npy   # vektor embedding (bulk)
  emb/<model>/.ckpt/                # checkpoint resume
  noise_references/{dawn,day,dusk,night}/
```

Kuota terakhir: HOME **62 GB / 1 TB (6%)**, Ephemeral **1.83 TB / 11 TB**.
Ukuran: `sea-emb` 16 GB, `sea-dashboards` 72 MB, `emb/` 126 GB.

Path bisa dioverride lewat env: `SEA_RESULTS` (default `~/sea-emb`),
`SEA_DASHBOARDS` (default `~/sea-dashboards`), `ANALYSIS_OUTPUT` (root ephemeral).

**Konsekuensi teknis:** vektor dan metadata-nya ada di dua filesystem berbeda, jadi setiap
pembaca menerima dua direktori. Lihat `embedding_io.load_embeddings_from_dir(emb_dir, meta_dir=...)`.
Semua keputusan path terpusat di `embedding_schema.py` — jangan menyusun path secara manual.

---

## 2. Cara Menjalankan

```bash
module load tools/prod Python/3.11.5-GCCcore-13.2.0
VENV=~/spatial-ecoacoustic-analysis/bacpipe/.venv/bin/python
cd ~/spatial-ecoacoustic-analysis

# ekstraksi embedding (butuh GPU)
$VENV bacpipe/pipeline_bacpipe.py --location 2A400 --date 2026-04-21 --device auto

# noise reference satu tanggal (CPU saja)
$VENV generate_official_noise_references.py --location 2A400 --date 2026-04-21
#   deteksi hanya di beam referensi LabIR(S05_000), lalu interval yang sama
#   diiris ke SELURUH beam LabIR/SPIR + sa + mono, disambung jadi satu file per beam
#   --target-sec 60     ambang peringatan durasi per kondisi
#   --max-sec 120       batasi durasi; 0 = ambil semua interval yang lolos
#   --no-overlap        buang window bertindih (default: overlap dibiarkan)

# dashboard (CPU saja, tidak butuh GPU)
$VENV visualize_bacpipe.py --location 2A400 --date 2026-04-21
#   default output -> ~/sea-dashboards/2A400/2026-04-21
#   --narrative auto|force|off      auto memakai cache, force regenerate
#   --narrative-model <id>          default minimax/minimax-m3:free
#   --models birdnet                batasi ke satu model saat uji

# satu tanggal penuh (signal processing + embedding + dashboard)
./run_full_date.sh 2A400 2026-04-21
```

`run_2A400_all_dates.sh` mengecek kelengkapan per tanggal sebelum menjalankan.

---

## 3. Apa yang Dihitung

Rekaman 6-channel dirender jadi **4 metode** yang dibandingkan di window waktu yang sama:

- `mono` — satu channel mikrofon (**baseline**)
- `sa` — signal averaging antar channel
- `bf_LabIR` — beamforming dengan impulse response ukur laboratorium
- `bf_SPIR` — beamforming dengan impulse response ukur in-situ (SPIR1 + SPIR2)

### Geometri beam — baca ini sebelum mengekstrak ulang apa pun

Set IR mentah jauh lebih besar daripada yang dipakai; hanya sebagian arah yang dipilih.

**LabIR = 19 beam.** Speaker [1, 5, 9, 12] × azimut [0, 60, 120, 180, 240, 300], dengan S12
adalah zenith sehingga hanya punya satu arah:
S01 (elevasi −45°) × 6 + S05 (0°) × 6 + S09 (+45°) × 6 + S12 (zenith) × 1 = **19**.

**SPIR pernah 45, sekarang 31 di config.**
- SPIR1 = jarak [2, 4, 8, 16] × azimut [0, 60, 120, 180, 240, 300] = **24** (tidak berubah)
- SPIR2 = jarak [1, 2, 4, 8, 16, 32, 64] × azimut [180] × repetisi
  - dulu 3 repetisi → 7 × 3 = 21 beam, banyak yang redundan
  - user meminta pilih satu repetisi saja (`rep_values=[2]`) → 7 × 1 = **7**
- Total: dulu 24 + 21 = **45**, sekarang 24 + 7 = **31**

**Sudah diselesaikan lewat allowlist, bukan lewat beamform ulang.** Audio di disk masih memuat
45 beam SPIR karena di-beamform 22 Agustus, sebelum keputusan rep-2. `config.expected_beam_tags()`
menurunkan daftar beam sah langsung dari `PRODUCTION_IR_SUBSETS` (19 LabIR + 24 SPIR1 +
7 SPIR2 = 50), dan penyaringan dilakukan di tiga tempat:

- `generate_official_noise_references.py` — beam di luar daftar tidak pernah jadi referensi
- `embedding_io.load_embeddings_from_dir` — barisnya dibuang saat pemuatan, dengan laporan jumlah
- `pipeline_bacpipe` — referensi noise untuk beam tak sah diabaikan

Hasilnya identik dengan beamform ulang, karena tiap beam adalah operasi filter independen:
SPIR1 dan SPIR2-r2 tidak terpengaruh ada-tidaknya r1/r3. Terverifikasi di 04-21: noise ref
52 file (bukan 66), embedding bf_SPIR 25.200 → 17.360 = 560 × 31, 7.840 baris dibuang.

Berkas SPIR2 r1 dan r3 sengaja **tidak** dihapus dari disk; ia hanya tidak pernah dipakai.
Kalau suatu saat config berubah lagi, daftar sahnya ikut berubah tanpa perlu menyentuh data.

Tiap window dinilai dengan **noise distance**: cosine distance embedding terhadap mean vector
noise reference habitat. Makin jauh dari noise = makin bagus, itu efek yang diharapkan dari
beamforming. Delta dilaporkan terhadap `mono`, diuji one-sided Wilcoxon signed-rank + Cliff's delta.

**Noise distance BUKAN SNR dalam dB.** Jangan pernah menuliskannya seolah-olah dB.

Untuk `bf_*`, arah θ*(t) yang dilaporkan adalah arah steer dengan cosine distance terbesar
dari noise reference — dipilih per window, bukan tetap.

### Bagaimana noise reference dibentuk

Aturannya, dijalankan oleh `generate_official_noise_references.py`:

1. Deteksi **hanya** pada beam referensi `LabIR(S05_000)` dari setiap rekaman di tanggal itu.
   Rekaman dibin ke dawn / day / dusk / night menurut jamnya.
2. **Semua** window 2 detik yang lolos disimpan, bukan cuma yang terbaik. Detektor memakai
   hop 1 detik sehingga window bertindih; pengulangan itu dibiarkan karena tidak merugikan
   untuk profil noise. Manifest mencatat dua angka: panjang tersambung dan panjang unik.
3. Triple `(rekaman, start, end)` hasil deteksi adalah satu-satunya sumber kebenaran, lalu
   diiris **identik** ke seluruh beam LabIR, seluruh beam SPIR, sa, dan mono.
4. Semua irisan satu beam **disambung jadi satu file** berisi background noise saja,
   dengan fade 5 ms di tiap sambungan supaya tidak ada klik.

Hasilnya per kondisi: 19 file LabIR + 45 file SPIR + 1 sa + 1 mono = **66 file**, semuanya
berdurasi sama dan mencakup instan waktu yang sama persis. Angka "1" untuk mono dan sa itu
efek jumlah channel, bukan kekurangan sampel waktu.

**Hasil 2026-04-21:** 7 rekaman, semuanya night. 21 interval dari 5 rekaman (2 rekaman nol
kandidat) → **42,0 detik tersambung / 38,0 detik unik** per stream. Di bawah target 60 detik;
lihat §5.6.

### 3.1 Aturan pemasangan noise reference

Setiap window dinilai terhadap referensi yang berasal dari **metode yang sama dan kondisi
waktu yang sama**. Contoh: rekaman `23-02-27` dengan metode SPIR dipasangkan dengan
`night_SPIR1(...)`, bukan dengan referensi mono dan bukan dengan referensi dawn.

Pemasangannya **per beam**, bukan per metode. `LabIR(S01_000)` dinilai terhadap
`night_noise_LabIR(S01_000)` saja — noise yang ditangkap lewat beam yang sama persis.
Jadi ada 66 prototipe terpisah: 19 LabIR + 45 SPIR + 1 sa + 1 mono.

Nama berkas referensi: `<condition>_noise_<beam>.wav`, mis.
`night_noise_LabIR(S01_000).wav`, `night_noise_SPIR2(64m_180_r2).wav`, `night_noise_mono.wav`.
Tanda kurung dipertahankan supaya arah steer-nya terbaca.

Kondisi tiap window diturunkan dari jam di nama rekamannya (`condition_from_wav`), arah
beam-nya dari `beam_tag_from_name`. Urutan resolusi di `resolve_noise_vector()`:

1. `<condition>_<beam>` — waktu sama, arah sama; ini yang normal dipakai
2. `<condition>_<group>` — waktu sama, metode digabung antar beam (cadangan)
3. `<group>` — referensi tanpa kondisi, cadangan untuk tanggal lama
4. tidak ada → window **tidak diberi skor** (NaN) dan dikeluarkan dari rata-rata

Poin 3 penting: sebelumnya window tanpa referensi diam-diam bernilai 0 sehingga tercatat
sebagai noise distance 1,0 — angka karangan. Sekarang `noise_analysis.per_method` memuat
`n_scored`, `n_unscored`, dan `noise_keys` supaya cakupannya kelihatan.

Penilaian per-window yang otoritatif ada di `visualize_bacpipe`. Ringkasan di
`pipeline_bacpipe` tidak punya metadata per window, jadi kalau satu tanggal memuat lebih
dari satu kondisi ia melaporkan tiap kondisi terpisah dengan status `scored_per_condition`,
bukan satu angka gabungan.

---

## 4. Status Data

12 model: `audioprotopnet avesecho_passt biolingual birdaves_especies birdmae birdnet
birdnet_v3 convnext_birdset perch_bird perch_v2 protoclr vggish`

**Cakupan tanggal tidak rata antar model** — jangan asumsikan semua model punya semua tanggal:

| Model | Jumlah tanggal |
|---|---|
| birdnet | 12 (2026-04-21 … 2026-05-18) |
| perch_bird | 9 |
| vggish | 8 |

**11 dari 12 tanggal konsisten** pada rasio LabIR 19 dan SPIR 45 beam per window.
Pengecualiannya `2026-04-22` — lihat §5.5.

Dashboard yang sudah ada: `2026-04-21` (12 model), `2026-04-26` (12 model).

---

## 5. Masalah Terbuka

### 5.1 Keunggulan beamforming sebagian besar adalah selection bias — TERUKUR

Riwayatnya: mula-mula δ = −1,0 (BF kalah 560/560) karena referensi noise rusak. Setelah
referensi 04-21 dibangun ulang dan dipasangkan per beam per kondisi, angkanya jadi δ ≈ +0,98
(BF menang ~99 %). Dua-duanya ekstrem, dan dua-duanya artefak.

Sekarang seluruh 36.960 embedding dinilai (bukan hanya 2.240 yang lolos seleksi θ*), dan
sebabnya terukur:

| metode | best beam | median beam | mean beam | **best − median** |
|---|---|---|---|---|
| bf_LabIR | +0,0546 / 99 % | +0,0022 / 63 % | +0,0071 / 76 % | **+0,0524** |
| bf_SPIR | +0,0556 / 100 % | +0,0101 / 74 % | +0,0115 / 77 % | **+0,0455** |
| sa | +0,0244 / 82 % | +0,0244 / 82 % | +0,0244 / 82 % | 0,0000 |

**96 % keunggulan bf_LabIR dan 82 % keunggulan bf_SPIR hilang begitu beam tidak boleh dipilih.**
Sebabnya sirkular: θ*(t) dipilih sebagai beam dengan noise distance **tertinggi**, lalu noise
distance itu juga yang dilaporkan sebagai hasil. Metrik yang sama dipakai untuk memilih dan
untuk menilai.

`sa` jadi kontrol yang bagus: ia hanya punya satu sinyal sehingga tidak bisa memilih, dan
ketiga kolomnya identik. Itu sekaligus bukti bahwa perhitungannya benar.

Dinilai atas **semua** beam, urutannya berbalik lagi:

| metode | beam | n | noise distance | Δ vs mono |
|---|---|---|---|---|
| sa | 1 | 560 | 0,0777 | **+0,0244** |
| bf_SPIR | 31 | 17.360 | 0,0648 | +0,0115 |
| bf_LabIR | 19 | 10.640 | 0,0603 | +0,0070 |
| mono | 1 | 560 | 0,0533 | 0 |

Signal averaging mengungguli kedua beamformer. Angka `best beam` **tidak boleh** dikutip
sebagai performa beamforming tanpa menyebut bahwa beam-nya dipilih memakai metrik yang sama.

Yang belum dijawab: apakah ada cara memilih beam yang **tidak** memakai noise distance
(mis. dari arah sumber yang diketahui, atau kriteria independen). Kalau ada, angka best-beam
jadi sah. Selama belum ada, median beam adalah pembanding yang jujur.

### 5.6 Noise reference — produsen sudah diperbaiki, konsumennya belum

**Sudah beres.** Generator lama hanya memakai **satu** interval 2 detik: parameter
`top_n_windows` ada tapi mati, karena setelah sorting kodenya tetap `best_w = selected[0]`.
Daftar rekamannya juga di-hardcode untuk 2026-05-15. Sekarang seluruh interval yang lolos
dipakai dan seluruh rekaman di satu tanggal ikut diproses. Untuk 2026-04-21 hasilnya naik
dari 2 detik menjadi **42 detik** per stream (21× lipat).

**Masih terbuka:**

1. **Durasi di bawah target.** 04-21 night hanya menghasilkan 38 detik audio unik, bukan 60.
   Bukan bug: dua dari tujuh rekaman menghasilkan **nol** kandidat dan tiap rekaman memuat
   300-460 event biofoni, jadi kriteria bersih yang berlaku memang hanya meloloskan sebanyak
   itu. Menaikkannya ke 60 detik berarti melonggarkan kriteria, dan itu keputusan ilmiah,
   bukan keputusan teknis.
2. ~~Pemasangan belum sadar kondisi.~~ **Sudah beres** — lihat §3.1.
3. **Klip pendek vs window model.** Dulu klip 2 detik di-embed sebagai window 3 detik
   (window BirdNET). Dengan file 42 detik masalah ini hilang dengan sendirinya, tapi
   mekanismenya tetap belum dikonfirmasi ke kode.

Poin 1 dan 2 masih menyumbang ke §5.1: prototipe BF adalah centroid 19/45 beam sedangkan
mono/sa hanya 1 beam, jadi asimetri jumlah beam tetap ada meski durasinya sudah setara.

### 5.5 2026-04-22 rusak / tidak lengkap
Diverifikasi dari bentuk array birdnet (mono 17.416, LabIR 147.852, SPIR 391.860):
- SPIR utuh secara struktur (391.860 = 45 × 8.708) tapi **hanya mencakup separuh window mono**
  (8.708 dari 17.416).
- LabIR **147.852 tidak habis dibagi 19** (sisa 13) — array terpotong di tengah grup beam.

Tanggal ini pernah di-resume dari tengah dan tampaknya tidak pernah selesai. Matched-window
alignment akan diam-diam memakai irisannya saja, jadi dashboard 04-22 tidak akan error tapi
juga tidak mewakili satu hari penuh. **Ekstrak ulang sebelum dipakai untuk kesimpulan apa pun**
— tapi baca peringatan 45 vs 31 beam di §3 dulu.

### 5.4 perch_bird masih dipaksa ke CPU padahal bisa GPU
`bacpipe/pipeline_bacpipe.py` memasukkan `perch_bird`, `perch_v2`, `surfperch` ke
`TF_CPU_ONLY_MODELS`. Ini terbukti tidak perlu — lihat §6.1. Efek sampingnya:
`tf.config.set_visible_devices([], "GPU")` bersifat global per proses dan tidak bisa dibalik,
jadi `vggish` (urutan ke-12) ikut jatuh ke CPU padahal terdaftar sebagai model GPU.

---

## 6. Fakta Lapangan CX3 (mahal didapat, jangan diulang)

### 6.1 perch_bird JALAN di GPU
Terukur di RTX 6000 (sm_75): batch 8×5 detik **0.018 s di GPU vs 1.175 s di CPU (~65×)**.
Syaratnya XLA harus menemukan ptxas/libdevice:

```python
CUDA = "/rds/easybuild/noarch/apps/software/CUDA/12.6.0"
os.environ["PATH"] = CUDA + "/bin:" + os.environ["PATH"]
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=" + CUDA
```

Resep ini ada di `bacpipe/run_pilot.py` tapi **tidak** ada di `pipeline_bacpipe.py`.
TF 2.20 di venv adalah CUDA build; gate cuDNN bacpipe lolos (torch cudnn 92400 ≥ 9.3).

### 6.2 Akses LLM dari compute node
Compute node **punya internet keluar** (github 200, api.anthropic.com 401 = tembus).
- `pi -p` **hang** di compute node (butuh TTY) — jangan dipakai di dalam job.
- Yang dipakai sekarang: HTTP langsung ke OpenRouter dari `narrative.py`, key dari
  `~/.openrouter_api_key`. Model default `minimax/minimax-m3:free` (gratis).
- Endpoint free kadang balas **402/429 sporadis**; `narrative.py` retry 3× dengan backoff
  lalu jatuh ke ringkasan rule-based. Laporan tidak pernah gagal karena LLM.
- Kredensial xAI di `~/.pi/agent/auth.json` **mati** di CX3 dan tidak bisa refresh
  non-interaktif. Key API xAI ada tapi team-nya belum punya credit.

### 6.3 SSH ke compute node
`sshnode` di `~/.bashrc` tidak menerima argumen perintah — dia membuka shell interaktif.
Untuk menjalankan perintah langsung:

```bash
PBS_JOBID=<jobid> ssh -T <node> "<perintah>"
```

Antrean `RTX6000` jauh lebih sepi daripada `L40S`. Helper: `qsgpu`, `qsrtx`, `qgpu`, `hpc`.

---

## 7. Aturan Kerja

**Sitasi.** Jangan pernah menempelkan klaim metodologis ke sebuah paper tanpa memverifikasi
paper itu benar-benar memuatnya. Pernah terjadi: template melabeli protokol matched-window
sebagai "(Cobos et al. 2017)", padahal survei tersebut (doi:10.1155/2017/3956282) tidak memuat
istilah itu sama sekali — papernya asli, atribusinya karangan. Klaim itu sempat tercetak di
19 file HTML sebelum dihapus. Kalau butuh sitasi, verifikasi ke DOI dulu; kalau tidak
mendukung, tulis saja apa yang dilakukan kode tanpa sitasi.

**Kartu Interpretation.** Ditulis LLM dari angka laporan itu sendiri, berlabel `AI-written`.
Isinya harus selalu bisa dicek ulang ke tabel di bawahnya.

**Sebelum menghapus atau memindahkan** apa pun milik user: survei dulu, jangan berasumsi dua
folder bernama mirip itu duplikat. Pernah hampir salah — dua run `2026-04-26` ternyata berbeda
jumlah model dan struktur manifest.

**Verifikasi.** `py_compile` tidak menangkap `NameError`. Pakai `pyflakes` untuk undefined name,
dan selalu uji end-to-end minimal satu model setelah mengubah path atau I/O.

**Gaya kode:** sederhana, tanpa fallback/backward-compat kecuali diminta.
**Bahasa chat:** Indonesia praktis, istilah teknis tetap Inggris.
