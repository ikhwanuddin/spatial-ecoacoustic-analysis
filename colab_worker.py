#!/usr/bin/env python3
"""
Colab Worker — distributed processing per JAM (hour group).

Claim: satu kelompok jam (date_dir + hour), proses semua FLAC di jam itu.
Output beamforming ditaruh di subfolder per-jam:
    beamforming_LabIR/13/  (semua WAV jam 13)

State di _state/:
    {date_dir}_{hour}.done     → hour group selesai
    {date_dir}_{hour}.{sid}    → hour group diklaim

Semua output diflush segera agar terminal tetap interaktif.
"""

import os, sys, json, time, glob, subprocess, uuid
from datetime import datetime, timezone
from typing import List, Set, Tuple, Dict

# ── Config ──────────────────────────────────────────────────

GD_BASE       = "/drive/MyDrive"
LOCATION      = "2A400"
RPIID         = "RPiID-0000000091668b26"
IR_TYPES_LIST = ["LabIR", "SPIR1", "SPIR2"]
CLAIM_WAIT    = 30  # detik — GDrive sync

MONITORING_DIR = f"{GD_BASE}/monitoring_data/{RPIID}"
OUTPUT_BASE    = f"{GD_BASE}/sea-data/{LOCATION}"
STATE_DIR      = f"{OUTPUT_BASE}/_state"
IR_BASE        = f"{GD_BASE}/MAARU-Impulse-Response"
REPO_DIR       = "spatial-ecoacoustic-analysis"
SESSION_FILE   = f"{STATE_DIR}/_session_id.txt"

os.environ["MONITORING_DATA"] = os.path.join(GD_BASE, "monitoring_data")
os.environ["ANALYSIS_OUTPUT"]  = os.path.join(GD_BASE, "sea-data")
os.environ["IR_BASE_PATH"]     = IR_BASE

# Global start time
SESSION_START = datetime.now(timezone.utc)


# ── Helpers ─────────────────────────────────────────────────

def ts() -> str:
    """Current timestamp string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def elapsed_since(start: datetime) -> str:
    """Human-readable elapsed time."""
    secs = (datetime.now(timezone.utc) - start).total_seconds()
    if secs < 60:
        return f"{secs:.0f}s"
    elif secs < 3600:
        return f"{secs/60:.1f}m"
    else:
        h = int(secs // 3600)
        m = int((secs % 3600) // 60)
        return f"{h}h {m}m"


def log(msg: str):
    """Print timestamped message, flush immediately."""
    print(f"[{ts()}] {msg}")
    sys.stdout.flush()


def run_stream(cmd: list, desc: str = "") -> bool:
    if desc:
        log(f"  {desc}...")
    try:
        result = subprocess.run(cmd, timeout=600)
        if result.returncode != 0:
            log(f"  ⚠ {desc} exit code {result.returncode}")
            return False
        return True
    except subprocess.TimeoutExpired:
        log(f"  ❌ {desc}: timeout")
        return False
    except Exception as e:
        log(f"  ❌ {desc}: {e}")
        return False


def get_session_id() -> str:
    os.makedirs(STATE_DIR, exist_ok=True)
    if os.path.isfile(SESSION_FILE):
        with open(SESSION_FILE) as f:
            sid = f.read().strip()
            if sid:
                return sid
    sid = f"worker-{uuid.uuid4().hex[:8]}"
    with open(SESSION_FILE, "w") as f:
        f.write(sid)
    return sid


SESSION_ID = get_session_id()


def parse_hour(basename: str) -> str:
    try:
        return basename.split("-")[0]
    except:
        return "00"


# ── Setup ───────────────────────────────────────────────────

def setup_environment():
    log("=" * 50)
    log(f"🔧 Colab Worker Setup — {SESSION_ID}")
    log("=" * 50)

    if not os.path.isdir(GD_BASE):
        log("❌ GDrive not mounted at /drive/MyDrive/")
        sys.exit(1)
    log(f"  ✅ GDrive at {GD_BASE}")

    os.makedirs(STATE_DIR, exist_ok=True)

    # Clone or pull repo to get latest code
    if os.path.isdir(REPO_DIR):
        log("  Repo exists, pulling latest...")
        if not run_stream(["git", "-C", REPO_DIR, "pull", "origin", "main"],
                          "git pull"):
            log("  git pull failed, recloning...")
            import shutil
            shutil.rmtree(REPO_DIR, ignore_errors=True)
    if not os.path.isdir(REPO_DIR):
        log("  Cloning repo...")
        if not run_stream(["git", "clone",
            "https://github.com/ikhwanuddin/spatial-ecoacoustic-analysis.git"]):
            sys.exit(1)

    # pip install — streaming, no -q
    log("  pip install (streaming)...")
    ok = run_stream([sys.executable, "-m", "pip", "install",
                     "-r", f"{REPO_DIR}/requirements-colab.txt"])
    if not ok:
        log("  ⚠ Fallback: one by one...")
        for pkg in ["librosa", "soundfile", "scipy", "numpy", "resampy",
                     "pydub", "birdnetlib", "tensorflow"]:
            run_stream([sys.executable, "-m", "pip", "install", pkg])

    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)

    try:
        import config
        log(f"  ✅ Modules OK")
    except ImportError as e:
        log(f"  ❌ Import error: {e}")
        sys.exit(1)

    # IR caches
    log("  Building IR caches (2-3 min)...")
    from ircache import build_all_caches
    build_all_caches()
    log("  ✅ IR caches ready")


# ── State Management ────────────────────────────────────────

def list_hour_groups() -> Dict[str, List[Tuple[str, str]]]:
    """List all (date_dir, hour) groups with their FLAC basenames."""
    groups: Dict[str, List[Tuple[str, str]]] = {}
    if not os.path.isdir(MONITORING_DIR):
        return groups
    for date_dir in sorted(os.listdir(MONITORING_DIR)):
        dp = os.path.join(MONITORING_DIR, date_dir)
        if not os.path.isdir(dp) or date_dir == "logs" or date_dir.startswith("."):
            continue
        for f in sorted(os.listdir(dp)):
            if not f.lower().endswith(".flac") or f.startswith("._"):
                continue
            hour = parse_hour(f)
            key = f"{date_dir}_{hour}"
            groups.setdefault(key, []).append((date_dir, f))
    return groups


def list_state() -> Tuple[Set[str], Set[str]]:
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
            claimed.add(fname)
    return done, claimed


def claim_hour_group(key: str) -> bool:
    claim_path = os.path.join(STATE_DIR, f"{key}.{SESSION_ID}")
    try:
        with open(claim_path, "w") as f:
            json.dump({"session": SESSION_ID,
                       "claimed_at": datetime.now(timezone.utc).isoformat(),
                       "group": key}, f)
        return True
    except Exception as e:
        log(f"  ⚠ Failed to claim {key}: {e}")
        return False


def verify_claim(key: str) -> bool:
    claim_path = os.path.join(STATE_DIR, f"{key}.{SESSION_ID}")
    if os.path.isfile(claim_path):
        return True
    others = glob.glob(os.path.join(STATE_DIR, f"{key}.*"))
    others = [f for f in others if not f.endswith(".done") and SESSION_ID not in f]
    if others:
        log(f"  ⚠ {key} diklaim session lain")
    return False


def mark_done(key: str):
    done_path = os.path.join(STATE_DIR, f"{key}.done")
    claim_path = os.path.join(STATE_DIR, f"{key}.{SESSION_ID}")
    try:
        with open(done_path, "w") as f:
            json.dump({"session": SESSION_ID,
                       "done_at": datetime.now(timezone.utc).isoformat()}, f)
    except:
        pass
    try:
        if os.path.isfile(claim_path):
            os.remove(claim_path)
    except:
        pass


# ── Pipeline per hour group ─────────────────────────────────

def process_hour_group(date_dir: str, hour: str, flac_basenames: List[str]) -> float:
    """
    Process all FLACs in one hour group.
    Returns elapsed time in seconds.
    """
    from config import PRODUCTION_IR_SUBSETS, SITE_COORDS
    from beamforming import Beamformer
    from signal_averaging import SignalAverager

    # birdnet_processor auto-applies FP16 monkey-patch at import time
    from birdnet_processor import process_directory_pipeline

    t_start = datetime.now(timezone.utc)
    lat = SITE_COORDS.get(LOCATION, {}).get("lat")
    lon = SITE_COORDS.get(LOCATION, {}).get("lon")
    rec_date = datetime.strptime(date_dir, "%Y-%m-%d")

    date_out = os.path.join(OUTPUT_BASE, date_dir)
    n_flac = len(flac_basenames)

    log(f"    🎙  Hour group: {date_dir} hour {hour} ({n_flac} FLACs)")
    log(f"    ⏱  Started at {t_start.strftime('%H:%M:%S UTC')}")

    # ── Beamforming ──────────────────────────────────────────
    for ir_name in IR_TYPES_LIST:
        ir_type = PRODUCTION_IR_SUBSETS[ir_name]
        bf_hour_dir = os.path.join(date_out, f"beamforming_{ir_name}", hour)
        os.makedirs(bf_hour_dir, exist_ok=True)

        n_expected = len(ir_type.param_values) * len(ir_type.degree_values)
        if ir_type.rep_values:
            n_expected *= len(ir_type.rep_values)
        if ir_type.zenith_speakers:
            n_expected -= len(ir_type.zenith_speakers) * (len(ir_type.degree_values) - 1)

        existing = len([f for f in os.listdir(bf_hour_dir)
                        if f.endswith(".wav") and not f.startswith("._")]) \
                   if os.path.isdir(bf_hour_dir) else 0
        expected_total = n_expected * n_flac
        if existing >= expected_total:
            log(f"    ↳ BF [{ir_name}]: {existing}/{expected_total} WAVs, skipping")
            continue

        log(f"    Beamforming [{ir_name}] — {n_flac} files × {n_expected} dirs → {bf_hour_dir}")
        for basename in flac_basenames:
            flac_path = os.path.join(MONITORING_DIR, date_dir, basename)
            try:
                bf = Beamformer(flac_path=flac_path, output_dir=bf_hour_dir,
                               ir_type_or_name=ir_type)
                bf.run()
            except Exception as e:
                log(f"    ❌ BF [{ir_name}] {basename}: {e}")

    # ── Signal Averaging ─────────────────────────────────────
    sa_hour_dir = os.path.join(date_out, "signal_averaging", hour)
    os.makedirs(sa_hour_dir, exist_ok=True)
    log(f"    Signal Averaging → {sa_hour_dir}")
    for basename in flac_basenames:
        base_name = os.path.splitext(basename)[0]
        sa_out = os.path.join(sa_hour_dir, f"{base_name}_sa.wav")
        if os.path.isfile(sa_out):
            continue
        try:
            sa = SignalAverager(
                flac_path=os.path.join(MONITORING_DIR, date_dir, basename),
                output_dir=sa_hour_dir)
            sa.run()
        except Exception as e:
            log(f"    ❌ SA {basename}: {e}")

    # ── Monochannel Baseline ─────────────────────────────────
    mono_hour_dir = os.path.join(date_out, "mono_baseline", hour)
    os.makedirs(mono_hour_dir, exist_ok=True)
    log(f"    Monochannel baseline → {mono_hour_dir}")
    import librosa as _librosa
    import soundfile as _sf
    from config import FS_TARGET
    for basename in flac_basenames:
        base_name = os.path.splitext(basename)[0]
        out_path = os.path.join(mono_hour_dir, f"{base_name}_mono.wav")
        if os.path.isfile(out_path):
            continue
        try:
            flac_path = os.path.join(MONITORING_DIR, date_dir, basename)
            raw, _ = _librosa.load(flac_path, sr=FS_TARGET, mono=True)
            amax = max(abs(raw))
            if amax > 1.0:
                raw = raw / amax
            _sf.write(out_path,
                      (raw * 32767).clip(-32768, 32767).astype("int16"),
                      FS_TARGET, subtype="PCM_16")
        except Exception as e:
            log(f"    ❌ Mono {basename}: {e}")

    # ── BirdNET ──────────────────────────────────────────────
    all_dirs = [
        os.path.join(date_out, f"beamforming_{ir_name}", hour)
        for ir_name in IR_TYPES_LIST
    ] + [sa_hour_dir, mono_hour_dir]

    log(f"    BirdNET analysis ({len(all_dirs)} dirs)...")
    for out_dir in all_dirs:
        if not os.path.isdir(out_dir):
            continue
        if os.path.isfile(os.path.join(out_dir, "results.json")):
            log(f"    ↳ {os.path.basename(out_dir)}: results.json exists, skipping")
            continue
        try:
            process_directory_pipeline(
                directory=out_dir, date=rec_date, identifier_pattern="",
                cleanup=False, dry_run=False, lat=lat, lon=lon,
            )
        except Exception as e:
            log(f"    ❌ BirdNET [{os.path.basename(out_dir)}]: {e}")

    elapsed = (datetime.now(timezone.utc) - t_start).total_seconds()
    log(f"    ✅ Hour group {date_dir}_{hour} done in {elapsed_since(t_start)}")
    return elapsed


# ── Main Loop ───────────────────────────────────────────────

def worker_loop():
    log("=" * 50)
    log(f"🔄 Worker Loop Started — {SESSION_ID}")
    log(f"   Started at: {SESSION_START.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    log("=" * 50)

    total_done = 0
    total_failed = 0
    total_hours = 0.0  # cumulative processing seconds

    while True:
        groups = list_hour_groups()
        done_set, claimed_set = list_state()

        if not groups:
            log("  ❌ No FLAC files found. Retry 60s...")
            time.sleep(60)
            continue

        # Find available hour groups
        available = []
        for key in sorted(groups.keys()):
            claim_fname = f"{key}.{SESSION_ID}"
            if key not in done_set and claim_fname not in claimed_set:
                other_claims = [c for c in claimed_set if c.startswith(f"{key}.")]
                if not other_claims:
                    n = len(groups[key])
                    available.append((key, n))

        total_grp = len(groups)
        n_done = len(done_set)
        n_avail = len(available)

        log(f"\n{'─' * 50}")
        log(f"  📊 Total groups: {total_grp} | Done: {n_done} | Avail: {n_avail}"
            f" | ✅: {total_done} | ❌: {total_failed}")
        log(f"  ⏱  Session runtime: {elapsed_since(SESSION_START)}"
            f" | Processing time: {total_hours/60:.1f} min")
        log(f"{'─' * 50}")

        if n_avail == 0:
            if n_done >= total_grp:
                total_elapsed = (datetime.now(timezone.utc) - SESSION_START).total_seconds()
                log("")
                log("=" * 50)
                log(f"  🎉 ALL DONE! {n_done}/{total_grp} hour groups")
                log(f"  ⏱  Total session: {elapsed_since(SESSION_START)}")
                log(f"  ⏱  Total processing: {total_hours/60:.1f} min")
                log(f"  ⏱  End: {ts()}")
                log("=" * 50)
                break
            log(f"  ⏳ No available groups, waiting {CLAIM_WAIT * 2}s...")
            time.sleep(CLAIM_WAIT * 2)
            continue

        # Pick one hour group
        key, n_flac = available[0]
        date_dir, hour = key.rsplit("_", 1)
        flac_list = [b for d, b in groups[key]]

        log(f"  🎯 Claiming {key} ({n_flac} FLACs, hour {hour})")

        if not claim_hour_group(key):
            time.sleep(CLAIM_WAIT)
            continue

        log(f"  ⏳ Waiting {CLAIM_WAIT}s for GDrive sync...")
        time.sleep(CLAIM_WAIT)

        if not verify_claim(key):
            log(f"  ⚠ Claim lost, retrying...")
            continue

        # Process
        try:
            elapsed = process_hour_group(date_dir, hour, flac_list)
            mark_done(key)
            total_done += 1
            total_hours += elapsed
        except Exception as e:
            log(f"  ❌ {key} failed: {e}")
            total_failed += 1
            import traceback
            traceback.print_exc()
            sys.stdout.flush()


if __name__ == "__main__":
    setup_environment()
    worker_loop()
