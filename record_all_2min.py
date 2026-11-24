import re
import time
import csv
import serial

PATTERN_FULL = re.compile(
    r"DET:\s*([-+]?\d*\.?\d+)\s*,\s*DIS:\s*([-+]?\d*\.?\d+)\s*,\s*BPM:\s*([-+]?\d*\.?\d+)\s*,\s*BEAT:\s*([-+]?\d*\.?\d+)",
    re.IGNORECASE
)

# إذا أحياناً BPM مش موجودة
PATTERN_NO_BPM = re.compile(
    r"DET:\s*([-+]?\d*\.?\d+)\s*,\s*DIS:\s*([-+]?\d*\.?\d+)\s*,\s*BEAT:\s*([-+]?\d*\.?\d+)",
    re.IGNORECASE
)

def record_all(port: str, out_csv: str = "ecg_all_2min.csv", baud: int = 115200, duration_sec: float = 120.0):
    ser = serial.Serial(port, baud, timeout=1)
    time.sleep(1.2)
    try:
        ser.reset_input_buffer()
    except Exception:
        pass

    rows = []
    t0 = time.time()
    print("[INFO] Recording DET, DIS, BPM, BEAT...")

    try:
        while True:
            t = time.time() - t0
            if t >= duration_sec:
                break

            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                continue

            m = PATTERN_FULL.search(line)
            if m:
                det = float(m.group(1))
                dis = float(m.group(2))
                bpm = float(m.group(3))
                beat = float(m.group(4))
                rows.append([f"{t:.4f}", det, dis, bpm, beat])
                continue

            m2 = PATTERN_NO_BPM.search(line)
            if m2:
                det = float(m2.group(1))
                dis = float(m2.group(2))
                beat = float(m2.group(3))
                rows.append([f"{t:.4f}", det, dis, "", beat])

    finally:
        try:
            ser.close()
        except Exception:
            pass

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t_sec", "DET", "DIS", "BPM", "BEAT"])
        w.writerows(rows)

    print(f"[DONE] Saved {len(rows)} rows to {out_csv}")

if __name__ == "__main__":
    # ✳️ عدلي COM هون
    record_all(port="COM6", out_csv="ecg_all_2min.csv")