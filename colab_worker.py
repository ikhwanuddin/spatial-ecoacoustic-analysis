#!/usr/bin/env python3
"""
Colab Worker — distributed processing untuk spatial ecoacoustic analysis.

Dijalankan di dalam Google Colab VM via `colab exec`.
Setiap worker meng-claim 5 file FLAC, memproses beamforming + BirdNET + mono,
lalu menandai selesai. Ulangi terus sampai semua file selesai.

State management: file-based di _state/ folder (hindari JSON merge conflict).

Session ID auto-generated (UUID), saved to _session_id.txt on first run.
Restart worker akan memakai ID yang sama — sehingga claim sebelumnya tetap valid.

Usage (dari Mac Mini, 3 terminal):
    colab exec -s colab-1 --timeout 86400 -f colab_worker.py
"""

import os, sys, json, time, glob, subprocess, uuid
from datetime import datetime
from typing import List, Set, Tuple

# ============================================================
# CONFIG — edit if needed
# ============================================================

GD_BASE       = "/drive/MyDrive"
LOCATION      = "2A400"
RPIID         = "RPiID-0000000091668b26"
IR_TYPES_LIST = ["LabIR", "SPIR1", "SPIR2"]  # 19 + 24 + 7 = 50 arah per file
MAX_FILES     = 5
CLAIM_WAIT    = 30  # detik — tunggu GDrive sync setelah claim

# GDrive paths
MONITORING_DIR = f"{GD_BASE}/monitoring_data/{RPIID}"
OUTPUT_BASE    = f"{GD_BASE}/sea-data/{LOCATION}"
STATE_DIR      = f"{OUTPUT_BASE}/_state"
IR_BASE        = f"{GD_BASE}/MAARU-Impulse-Response"
REPO_DIR       = "spatial-ecoacoustic-analysis"
SESSION_FILE   = f"{STATE_DIR}/_session_id.txt"

# Env vars for config.py
os.environ["MONITORING_DATA"] = os.path.join(GD_BASE, "monitoring_data")
os.environ["ANALYSIS_OUTPUT"]  = os.path.join(GD_BASE, "sea-data")
os.environ["IR_BASE_PATH"]     = IR_BASE


def get_session_id() -> str:
    """Get or create a persistent session ID."""
    os.makedirs(STATE_DIR, exist_ok=True)
    if os.path.isfile(SESSION_FILE):
        with open(SESSION_FILE) as f:
            sid = f.read().strip()
            if sid:
                return sid
    # Generate new unique ID
    sid = f"worker-{uuid.uuid4().hex[:8]}"
    with open(SESSION_FILE, "w") as f:
        f.write(sid)
    return sid


SESSION_ID = get_session_id()  # set after STATE_DIR exists

# ============================================================
# SETUP
# ============================================================

def run(cmd: list, desc: str = "") -> bool:
    if desc:
        print(f"  {desc}...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"  ❌ {desc}: {result.stderr[:300]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"  ❌ {desc}: timeout")
        return False
    except Exception as e:
        print(f"  ❌ {desc}: {e}")
        return False


def setup_environment():
    print("=" * 50)
    print(f"🔧 Colab Worker Setup — {SESSION_ID}")
    print("=" * 50)

    if not os.path.isdir(GD_BASE):
        print("❌ GDrive not mounted at /drive/MyDrive/")
        print("   Run: colab drivemount -s <session> /drive")
        sys.exit(1)
    print(f"  ✅ GDrive mounted at {GD_BASE}")

    os.makedirs(STATE_DIR, exist_ok=True)

    if not os.path.isdir(REPO_DIR):
        print("  Cloning spatial-ecoacoustic-analysis...")
        if not run(["git", "clone", "https://github.com/ikhwanuddin/spatial-ecoacoustic-analysis.git"],
                   "git clone"):
            sys.exit(1)

    print("  Installing Python dependencies (this may take a few minutes)...")
    ok = run([sys.executable, "-m", "pip", "install", "-q",
              "-r", f"{REPO_DIR}/requirements.txt"], "pip install")
    if not ok:
        print("  ⚠ Retrying individual packages...")
        for pkg in ["librosa", "soundfile", "scipy", "numpy", "resampy", "pydub",
                     "birdnetlib", "tensorflow"]:
            run([sys.executable, "-m", "pip", "install", "-q", pkg], f"pip {pkg}")

    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)

    try:
        import config
        print(f"  ✅ Pipeline modules loaded (monitoring: {config.MONITORING_DATA})")
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        sys.exit(1)

    # Precompute IR caches
    from ircache import build_all_caches
    print("  Building IR steering-vector caches...")
    build_all_caches()


# ============================================================
# STATE MANAGEMENT
# ============================================================

def list_all_flac() -> List[Tuple[str, str]]:
    """List all (date_dir, basename) FLAC tuples."""
    flacs = []
    if not os.path.isdir(MONITORING_DIR):
        return []
    for date_dir in sorted(os.listdir(MONITORING_DIR)):
        dp = os.path.join(MONITORING_DIR, date_dir)
        if not os.path.isdir(dp) or date_dir == "logs" or date_dir.startswith("."):
            continue
        for f in sorted(os.listdir(dp)):
            if f.lower().endswith(".flac") and not f.startswith("._"):
                flacs.append((date_dir, f))
    return flacs


def get_flac_full_path(date_dir: str, basename: str) -> str:
    return os.path.join(MONITORING_DIR, date_dir, basename)


def list_state() -> Tuple[Set[str], Set[str]]:
    """Return (done_basenames, claimed_basenames) from _state/ directory."""
    done = set()
    claimed = set()
    if not os.path.isdir(STATE_DIR):
        return done, claimed
    for fname in os.listdir(STATE_DIR):
        if fname.startswith("_session_id"):
            continue
        if fname.endswith(".done"):
            done.add(fname[:-5])
        else:
            # Non-done files are claim files
            claimed.add(fname)  # full filename as claim key
    return done, claimed


def claim_files(flac_basenames: List[str]) -> List[str]:
    """Create claim files. Returns successfully claimed."""
    claimed = []
    for bn in flac_basenames:
        claim_path = os.path.join(STATE_DIR, f"{bn}.{SESSION_ID}")
        try:
            with open(claim_path, "w") as f:
                f.write(json.dumps({"session": SESSION_ID, "claimed_at": datetime.now().isoformat(), "flac": bn}))
            claimed.append(bn)
        except Exception as e:
            print(f"  ⚠ Failed to claim {bn}: {e}")
    return claimed


def verify_claims(flac_basenames: List[str]) -> List[str]:
    """Re-read state, filter out files claimed by other sessions."""
    verified = []
    for bn in flac_basenames:
        claim_path = os.path.join(STATE_DIR, f"{bn}.{SESSION_ID}")
        if os.path.isfile(claim_path):
            verified.append(bn)
        else:
            others = glob.glob(os.path.join(STATE_DIR, f"{bn}.*"))
            others = [f for f in others if not f.endswith(".done") and SESSION_ID not in f]
            if others:
                print(f"  ⚠ {bn} claimed by another session, skipping")
            else:
                print(f"  ⚠ {bn} claim lost, re-claiming...")
                with open(claim_path, "w") as f:
                    f.write("retry")
                verified.append(bn)
    return verified


def mark_done(flac_basename: str):
    """Write .done and remove claim file."""
    done_path = os.path.join(STATE_DIR, f"{flac_basename}.done")
    claim_path = os.path.join(STATE_DIR, f"{flac_basename}.{SESSION_ID}")
    try:
        with open(done_path, "w") as f:
            f.write(json.dumps({"session": SESSION_ID, "done_at": datetime.now().isoformat()}))
    except:
        pass
    try:
        if os.path.isfile(claim_path):
            os.remove(claim_path)
    except:
        pass


# ============================================================
# PIPELINE
# ============================================================

def monochannel_baseline(flac_path: str, output_base: str, base_name: str) -> str:
    import librosa, soundfile as sf
    from config import FS_TARGET
    out_dir = os.path.join(output_base, "mono_baseline")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{base_name}_mono.wav")
    if os.path.isfile(out_path):
        return out_path
    raw, _ = librosa.load(flac_path, sr=FS_TARGET, mono=True)
    amax = max(abs(raw))
    if amax > 1.0:
        raw = raw / amax
    sf.write(out_path, (raw * 32767).clip(-32768, 32767).astype("int16"), FS_TARGET, subtype="PCM_16")
    return out_path


def process_one_flac(flac_path: str, date_dir: str, flac_basename: str):
    from config import PRODUCTION_IR_SUBSETS, SITE_COORDS
    from beamforming import Beamformer
    from signal_averaging import SignalAverager
    from birdnet_processor import process_directory_pipeline

    base_name = os.path.splitext(flac_basename)[0]
    loc_out = os.path.join(OUTPUT_BASE, date_dir)
    lat = SITE_COORDS.get(LOCATION, {}).get("lat")
    lon = SITE_COORDS.get(LOCATION, {}).get("lon")
    rec_date = datetime.strptime(date_dir, "%Y-%m-%d")

    output_dirs = []

    # Beamforming
    for ir_name in IR_TYPES_LIST:
        ir_type = PRODUCTION_IR_SUBSETS[ir_name]
        bf_dir = os.path.join(loc_out, f"beamforming_{ir_name}")
        output_dirs.append(bf_dir)

        n_expected = len(ir_type.param_values) * len(ir_type.degree_values)
        if ir_type.rep_values:
            n_expected *= len(ir_type.rep_values)
        if ir_type.zenith_speakers:
            n_expected -= len(ir_type.zenith_speakers) * (len(ir_type.degree_values) - 1)

        existing = len([f for f in os.listdir(bf_dir) if f.endswith(".wav") and not f.startswith("._")]) \
                   if os.path.isdir(bf_dir) else 0
        if existing >= n_expected:
            print(f"    ↳ {ir_name}: {existing}/{n_expected} WAVs already done, skipping")
            continue

        try:
            print(f"    Beamforming [{ir_name}]...")
            bf = Beamformer(flac_path=flac_path, output_dir=bf_dir, ir_type_or_name=ir_type)
            bf.run()
        except Exception as e:
            print(f"    ❌ {ir_name}: {e}")

    # Signal Averaging
    sa_dir = os.path.join(loc_out, "signal_averaging")
    sa_out = os.path.join(sa_dir, f"{base_name}_sa.wav")
    output_dirs.append(sa_dir)
    if not os.path.isfile(sa_out):
        try:
            print(f"    Signal Averaging...")
            sa = SignalAverager(flac_path=flac_path, output_dir=sa_dir)
            sa.run()
        except Exception as e:
            print(f"    ❌ SA: {e}")

    # Monochannel
    mono_path = monochannel_baseline(flac_path, loc_out, base_name)
    output_dirs.append(os.path.dirname(mono_path))

    # BirdNET on all
    print(f"    BirdNET analysis ({len(output_dirs)} dirs)...")
    for out_dir in output_dirs:
        if not os.path.isdir(out_dir):
            continue
        if os.path.isfile(os.path.join(out_dir, "results.json")):
            print(f"    ↳ {os.path.basename(out_dir)}: results.json exists, skipping")
            continue
        try:
            process_directory_pipeline(
                directory=out_dir, date=rec_date, identifier_pattern="",
                cleanup=False, dry_run=False, lat=lat, lon=lon,
            )
        except Exception as e:
            print(f"    ❌ BirdNET [{os.path.basename(out_dir)}]: {e}")

    print(f"    ✅ {flac_basename} done")


# ============================================================
# MAIN LOOP
# ============================================================

def worker_loop():
    print("\n" + "=" * 50)
    print(f"🔄 Worker Loop — {SESSION_ID}")
    print("=" * 50)

    total_done = 0
    total_failed = 0

    while True:
        all_flacs = list_all_flac()
        done_set, claimed_set = list_state()

        if not all_flacs:
            time.sleep(60)
            continue

        # Filter available
        available = []
        for date_dir, basename in all_flacs:
            bn = os.path.splitext(basename)[0]
            claim_fname = f"{bn}.{SESSION_ID}"
            if bn not in done_set and claim_fname not in claimed_set:
                # Also check no other session claimed it
                other_claims = [c for c in claimed_set if c.startswith(f"{bn}.")]
                if not other_claims:
                    available.append((date_dir, basename))

        total_all = len(all_flacs)
        n_done = len(done_set)
        n_avail = len(available)

        print(f"\n  ── {datetime.now().strftime('%H:%M:%S')} ──")
        print(f"  Total: {total_all} | Done: {n_done} | Avail: {n_avail} | Done: {total_done} | Failed: {total_failed}")

        if n_avail == 0:
            if n_done >= total_all:
                print(f"\n  🎉 ALL DONE! {n_done}/{total_all}")
                break
            print(f"  ⏳ No available files, waiting {CLAIM_WAIT * 2}s...")
            time.sleep(CLAIM_WAIT * 2)
            continue

        batch = available[:MAX_FILES]
        batch_bn = [os.path.splitext(b)[0] for _, b in batch]

        print(f"  🎯 Claiming {len(batch)}: {[b for _, b in batch]}")
        claimed_ok = claim_files(batch_bn)

        if not claimed_ok:
            time.sleep(CLAIM_WAIT)
            continue

        print(f"  ⏳ Waiting {CLAIM_WAIT}s for GDrive sync...")
        time.sleep(CLAIM_WAIT)

        verified = verify_claims(claimed_ok)
        if not verified:
            continue

        verified_batch = [(d, b) for d, b in batch if os.path.splitext(b)[0] in verified]

        for date_dir, basename in verified_batch:
            flac_path = get_flac_full_path(date_dir, basename)
            if not os.path.isfile(flac_path):
                print(f"  ❌ FLAC not found: {flac_path}")
                total_failed += 1
                mark_done(os.path.splitext(basename)[0])
                continue

            print(f"\n  🎙  Processing: {basename} ({date_dir})")
            t0 = time.time()
            try:
                process_one_flac(flac_path, date_dir, basename)
                mark_done(os.path.splitext(basename)[0])
                total_done += 1
                print(f"  ✅ Done ({time.time() - t0:.0f}s)")
            except Exception as e:
                print(f"  ❌ Failed: {e}")
                total_failed += 1
                mark_done(os.path.splitext(basename)[0])
                import traceback; traceback.print_exc()


if __name__ == "__main__":
    setup_environment()
    worker_loop()
