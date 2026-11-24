import re
import time
import csv
import serial

# يلتقط DET من أي سطر فيه "DET: ..."
PAT_DET = re.compile(r"DET:\s*([-+]?\d*\.?\d+)", re.IGNORECASE)

def record_det_only(port: str, out_csv: str = "det_2min.csv", baud: int = 115200, duration_sec: float = 120.0):
    ser = serial.Serial(port, baud, timeout=1)
    time.sleep(1.2)
    try:
        ser.reset_input_buffer()
    except Exception:
        pass

    rows = []
    t0 = time.time()
    print("[INFO] Recording DET only...")

    try:
        while True:
            t = time.time() - t0
            if t >= duration_sec:
                break

            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                continue

            m = PAT_DET.search(line)
            if not m:
                continue

            det = float(m.group(1))
            rows.append([f"{t:.4f}", det])

    finally:
        try:
            ser.close()
        except Exception:
            pass

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t_sec", "DET"])
        w.writerows(rows)

    print(f"[DONE] Saved {len(rows)} rows to {out_csv}")

if __name__ == "__main__":
    # ✳️ عدلي COM هون
    record_det_only(port="COM6", out_csv="det_2min.csv")