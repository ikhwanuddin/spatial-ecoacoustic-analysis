#!/usr/bin/env python3
"""
Colab Worker — distributed processing untuk spatial ecoacoustic analysis.

Dijalankan di dalam Google Colab VM via `colab exec`.
Setiap worker meng-claim 5 file FLAC, memproses beamforming + BirdNET + mono,
lalu menandai selesai. Ulangi terus sampai semua file selesai.

State management: file-based di _state/ folder (hindari JSON merge conflict).

Usage (dari Mac Mini, 3 terminal):
    colab exec -s colab-1 --timeout 86400 -f colab_worker.py
"""

import os, sys, json, time, glob, subprocess, hashlib
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
SESSION_ID    = os.environ.get("COLAB_SESSION", "unknown-worker")

# GDrive paths
MONITORING_DIR = f"{GD_BASE}/monitoring_data/{RPIID}"
OUTPUT_BASE    = f"{GD_BASE}/sea-data/{LOCATION}"
STATE_DIR      = f"{OUTPUT_BASE}/_state"
IR_BASE        = f"{GD_BASE}/MAARU-Impulse-Response"
REPO_DIR       = "spatial-ecoacoustic-analysis"

# Env vars for config.py
os.environ["MONITORING_DATA"] = os.path.join(GD_BASE, "monitoring_data")
os.environ["ANALYSIS_OUTPUT"]  = os.path.join(GD_BASE, "sea-data")
os.environ["IR_BASE_PATH"]     = IR_BASE

# ============================================================
# SETUP: GDrive mount, repo clone, install deps
# ============================================================

def run(cmd: list, desc: str = "") -> bool:
    """Run a shell command, print output, return success."""
    if desc:
        print(f"  {desc}...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"  ❌ {desc}: {result.stderr[:200]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"  ❌ {desc}: timeout")
        return False
    except Exception as e:
        print(f"  ❌ {desc}: {e}")
        return False


def setup_environment():
    """One-time setup: mount GDrive, clone repo, install deps."""
    print("=" * 50)
    print(f"🔧 Colab Worker Setup — {SESSION_ID}")
    print("=" * 50)

    # 1. Verify GDrive mount
    if not os.path.isdir(GD_BASE):
        print("❌ GDrive not mounted at /drive/MyDrive/")
        print("   Run: colab drivemount -s <session> /drive")
        sys.exit(1)
    print(f"  ✅ GDrive mounted at {GD_BASE}")

    # 2. Create state directory
    os.makedirs(STATE_DIR, exist_ok=True)

    # 3. Clone repo if needed
    if not os.path.isdir(REPO_DIR):
        print("  Cloning spatial-ecoacoustic-analysis...")
        ok = run(["git", "clone", "https://github.com/ikhwanuddin/spatial-ecoacoustic-analysis.git"],
                 "git clone")
        if not ok:
            sys.exit(1)

    # 4. Install deps
    print("  Installing Python dependencies...")
    ok = run([sys.executable, "-m", "pip", "install", "-q",
              "-r", f"{REPO_DIR}/requirements.txt"], "pip install")
    if not ok:
        print("  ⚠ pip install had issues, trying individual packages...")
        for pkg in ["librosa", "soundfile", "scipy", "numpy", "resampy", "pydub",
                     "birdnetlib", "tensorflow"]:
            run([sys.executable, "-m", "pip", "install", "-q", pkg], f"pip {pkg}")

    # 5. Add repo to path
    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)

    # 6. Verify imports
    try:
        import config
        print(f"  ✅ Pipeline modules loaded (monitoring: {config.MONITORING_DATA})")
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        print("  Make sure repo was cloned correctly")
        sys.exit(1)

    # 7. Precompute IR caches (one-time per session)
    from ircache import build_all_caches
    print("  Building IR steering-vector caches...")
    build_all_caches()


# ============================================================
# STATE MANAGEMENT (file-based, GDrive-safe)
# ============================================================

def list_all_flac() -> List[str]:
    """List all FLAC basenames across all date folders."""
    flacs = []
    if not os.path.isdir(MONITORING_DIR):
        print(f"  ❌ Monitoring dir not found: {MONITORING_DIR}")
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


def list_state() -> Tuple[set, set]:
    """Return (done_flacs, claimed_flacs) from _state/ directory."""
    done = set()
    claimed = set()
    if not os.path.isdir(STATE_DIR):
        return done, claimed
    for fname in os.listdir(STATE_DIR):
        # Done: {basename}.done
        if fname.endswith(".done"):
            done.add(fname[:-5])
        # Claimed: {basename}.{session_id}
        elif fname.startswith("."):
            continue
        else:
            # Try to parse as basename.session_id
            parts = fname.rsplit(".", 1)
            if len(parts) == 2:
                flac_base, sid = parts
                if sid != "done" and not sid.startswith("."):
                    claimed.add(flac_base)
    return done, claimed


def claim_files(flac_basenames: List[str]) -> List[str]:
    """Create claim files for the given FLAC basenames. Returns successfully claimed."""
    claimed = []
    for bn in flac_basenames:
        claim_path = os.path.join(STATE_DIR, f"{bn}.{SESSION_ID}")
        try:
            # Touch the claim file
            with open(claim_path, "w") as f:
                f.write(json.dumps({
                    "session": SESSION_ID,
                    "claimed_at": datetime.now().isoformat(),
                    "flac": bn,
                }))
            claimed.append(bn)
        except Exception as e:
            print(f"  ⚠ Failed to claim {bn}: {e}")
    return claimed


def verify_claims(flac_basenames: List[str]) -> List[str]:
    """Re-read state and filter out files claimed by other sessions."""
    verified = []
    for bn in flac_basenames:
        claim_path = os.path.join(STATE_DIR, f"{bn}.{SESSION_ID}")
        if os.path.isfile(claim_path):
            verified.append(bn)
        else:
            # Check if another session claimed it
            pattern = os.path.join(STATE_DIR, f"{bn}.*")
            others = [f for f in glob.glob(pattern)
                      if not f.endswith(".done") and SESSION_ID not in f]
            if others:
                print(f"  ⚠ {bn} claimed by another session, skipping")
            else:
                print(f"  ⚠ {bn} claim lost (GDrive sync?), re-claiming")
                # Re-claim
                with open(claim_path, "w") as f:
                    f.write("retry")
                verified.append(bn)
    return verified


def mark_done(flac_basename: str):
    """Create .done file and remove claim file."""
    done_path = os.path.join(STATE_DIR, f"{flac_basename}.done")
    claim_path = os.path.join(STATE_DIR, f"{flac_basename}.{SESSION_ID}")
    try:
        with open(done_path, "w") as f:
            f.write(json.dumps({
                "session": SESSION_ID,
                "done_at": datetime.now().isoformat(),
            }))
    except:
        pass
    try:
        if os.path.isfile(claim_path):
            os.remove(claim_path)
    except:
        pass


# ============================================================
# PIPELINE: process one FLAC file
# ============================================================

def monochannel_baseline(flac_path: str, output_base: str, base_name: str) -> str:
    """Extract channel 0 from 6-channel FLAC as baseline mono WAV."""
    import librosa
    import soundfile as sf
    from config import FS_TARGET

    out_dir = os.path.join(output_base, "mono_baseline")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{base_name}_mono.wav")

    if os.path.isfile(out_path):
        print(f"    ↳ mono already exists")
        return out_path

    # Read only channel 0
    raw, _ = librosa.load(flac_path, sr=FS_TARGET, mono=True)
    amax = max(abs(raw))
    if amax > 1.0:
        raw = raw / amax
    sf.write(out_path, (raw * 32767).clip(-32768, 32767).astype("int16"), FS_TARGET, subtype="PCM_16")
    return out_path


def process_one_flac(flac_path: str, date_dir: str, flac_basename: str):
    """Run full pipeline for one FLAC: BF (LabIR/SPIR1/SPIR2) + SA + mono + BirdNET."""
    from config import PRODUCTION_IR_SUBSETS, SITE_COORDS
    from beamforming import Beamformer
    from signal_averaging import SignalAverager
    from birdnet_processor import process_directory_pipeline

    base_name = os.path.splitext(flac_basename)[0]
    loc_out = os.path.join(OUTPUT_BASE, date_dir)
    lat, lon = SITE_COORDS.get(LOCATION, {}).get("lat"), SITE_COORDS.get(LOCATION, {}).get("lon")
    rec_date = datetime.strptime(date_dir, "%Y-%m-%d")

    output_dirs = []

    # ── Beamforming (LabIR + SPIR1 + SPIR2) ──────────────────
    for ir_name in IR_TYPES_LIST:
        ir_type = PRODUCTION_IR_SUBSETS[ir_name]
        bf_dir = os.path.join(loc_out, f"beamforming_{ir_name}")

        output_dirs.append(bf_dir)
        print(f"    Beamforming [{ir_name}]...")

        # Check if already done (count existing WAVs)
        n_expected = len(ir_type.param_values) * len(ir_type.degree_values)
        if ir_type.rep_values:
            n_expected *= len(ir_type.rep_values)
        # Adjust for zenith
        if ir_type.zenith_speakers:
            n_expected -= len(ir_type.zenith_speakers) * (len(ir_type.degree_values) - 1)

        existing = len([f for f in os.listdir(bf_dir) if f.endswith(".wav") and not f.startswith("._")]) \
                   if os.path.isdir(bf_dir) else 0
        if existing >= n_expected:
            print(f"    ↳ {ir_name}: {existing}/{n_expected} WAVs already done, skipping")
            continue

        try:
            bf = Beamformer(flac_path=flac_path, output_dir=bf_dir, ir_type_or_name=ir_type)
            bf.run()
        except Exception as e:
            print(f"    ❌ {ir_name} failed: {e}")
            continue

    # ── Signal Averaging ─────────────────────────────────────
    print(f"    Signal Averaging...")
    sa_dir = os.path.join(loc_out, "signal_averaging")
    sa_out = os.path.join(sa_dir, f"{base_name}_sa.wav")
    output_dirs.append(sa_dir)

    if not os.path.isfile(sa_out):
        try:
            sa = SignalAverager(flac_path=flac_path, output_dir=sa_dir)
            sa.run()
        except Exception as e:
            print(f"    ❌ SA failed: {e}")

    # ── Monochannel Baseline ──────────────────────────────────
    print(f"    Monochannel baseline...")
    mono_path = monochannel_baseline(flac_path, loc_out, base_name)
    mono_dir = os.path.dirname(mono_path)
    output_dirs.append(mono_dir)

    # ── BirdNET on all outputs ────────────────────────────────
    print(f"    BirdNET analysis...")
    for out_dir in output_dirs:
        if not os.path.isdir(out_dir):
            continue
        dir_label = os.path.basename(out_dir)
        has_results = os.path.isfile(os.path.join(out_dir, "results.json"))
        if has_results:
            print(f"    ↳ {dir_label}: results.json exists, skipping")
            continue

        try:
            process_directory_pipeline(
                directory=out_dir,
                date=rec_date,
                identifier_pattern="",
                cleanup=False,
                dry_run=False,
                lat=lat,
                lon=lon,
            )
        except Exception as e:
            print(f"    ❌ BirdNET [{dir_label}] failed: {e}")

    print(f"    ✅ {flac_basename} done")


# ============================================================
# MAIN WORKER LOOP
# ============================================================

def worker_loop():
    """Main loop: claim -> process -> done -> repeat."""
    print("\n" + "=" * 50)
    print(f"🔄 Worker Loop — {SESSION_ID}")
    print("=" * 50)

    total_done = 0
    total_failed = 0

    while True:
        # 1. List all FLACs and current state
        all_flacs = list_all_flac()
        done_set, claimed_set = list_state()

        if not all_flacs:
            print(f"  ❌ No FLAC files found in {MONITORING_DIR}")
            time.sleep(60)
            continue

        # 2. Find available files
        # Keep track of which (date_dir, basename) tuples are available
        available = []
        for date_dir, basename in all_flacs:
            bn = os.path.splitext(basename)[0]  # without .flac
            if bn not in done_set and bn not in claimed_set:
                available.append((date_dir, basename))

        total_all = len(all_flacs)
        n_done = len(done_set)
        n_claimed = len(claimed_set)
        n_avail = len(available)

        print(f"\n  ── {datetime.now().strftime('%H:%M:%S')} ──")
        print(f"  Total: {total_all} | Done: {n_done} | Claimed: {n_claimed} | Available: {n_avail}")

        if n_avail == 0:
            if n_done >= total_all:
                print(f"\n  🎉 ALL DONE! {n_done}/{total_all} files processed.")
                break
            else:
                print(f"  ⏳ No available files, waiting {CLAIM_WAIT * 2}s...")
                time.sleep(CLAIM_WAIT * 2)
                continue

        # 3. Claim up to MAX_FILES
        batch = available[:MAX_FILES]
        batch_basenames = [os.path.splitext(b)[0] for _, b in batch]

        print(f"  🎯 Claiming {len(batch)} files: {[b for _, b in batch]}")
        claimed_ok = claim_files(batch_basenames)
        print(f"  ✅ Claimed {len(claimed_ok)} files")

        # 4. Wait for GDrive sync
        print(f"  ⏳ Waiting {CLAIM_WAIT}s for GDrive sync...")
        time.sleep(CLAIM_WAIT)

        # 5. Verify claims
        verified = verify_claims(claimed_ok)
        if len(verified) < len(claimed_ok):
            print(f"  ⚠ {len(claimed_ok) - len(verified)} claims lost to another session")

        if not verified:
            print(f"  ⚠ No verified claims, retrying...")
            continue

        # 6. Process each verified file
        verified_batch = [(d, b) for d, b in batch
                          if os.path.splitext(b)[0] in verified]

        for date_dir, basename in verified_batch:
            flac_path = get_flac_full_path(date_dir, basename)
            if not os.path.isfile(flac_path):
                print(f"  ❌ FLAC not found: {flac_path}")
                total_failed += 1
                mark_done(os.path.splitext(basename)[0])  # mark done to skip
                continue

            print(f"\n  🎙  Processing: {basename} ({date_dir})")
            t0 = time.time()

            try:
                process_one_flac(flac_path, date_dir, basename)
                mark_done(os.path.splitext(basename)[0])
                total_done += 1
                elapsed = time.time() - t0
                print(f"  ✅ Done ({elapsed:.0f}s) | Progress: {n_done + total_done}/{total_all}")
            except Exception as e:
                print(f"  ❌ Failed: {e}")
                total_failed += 1
                mark_done(os.path.splitext(basename)[0])  # mark done to skip
                import traceback
                traceback.print_exc()

        print(f"  📊 Session stats: {total_done} done, {total_failed} failed")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Colab Worker")
    p.add_argument("--session-id", default=SESSION_ID, help="Unique session identifier")
    p.add_argument("--setup-only", action="store_true", help="Only run setup, then exit")
    args = p.parse_args()

    SESSION_ID = args.session_id

    setup_environment()

    if args.setup_only:
        print("\n✅ Setup complete. Ready for worker loop.")
        sys.exit(0)

    worker_loop()
