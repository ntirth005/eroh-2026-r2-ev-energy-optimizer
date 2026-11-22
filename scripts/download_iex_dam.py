"""Download 5 years of IEX Day-Ahead Market HOURLY results from the IEX India website with Playwright (Chromium),
in 30-day chunks (Export -> Excel), then merge them into data/iex_dam_5yr_hourly.csv.

Usage (from the repo root):
    pip install playwright pandas openpyxl xlrd
    playwright install chromium
    python scripts/download_iex_dam.py
A browser window opens (headless=False); chunks are saved to data/raw/iex_chunks/ and already-downloaded chunks are skipped."""
from datetime import datetime, timedelta
from pathlib import Path
import glob
import os
import time
import pandas as pd
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

URL = "https://www.iexindia.com/market-data/day-ahead-market/market-snapshot"
END_DATE = datetime(2026, 11, 22)
START_DATE = END_DATE - timedelta(days=5 * 365)
CHUNK_DAYS = 30          # the site allows at most 31 days per request
HEADLESS = False         # a visible browser by default; use --headless on a server

DOWNLOAD_DIR = str(DATA_DIR / "raw" / "iex_chunks")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def date_chunks(start: datetime, end: datetime, step: int):
    curr = start
    while curr <= end:
        c_end = min(curr + timedelta(days=step - 1), end)
        yield curr, c_end
        curr = c_end + timedelta(days=1)


def set_date_input(locator, date_str: str):
    """Clears and types a date into React/MUI date inputs reliably."""
    locator.click()
    locator.press("Control+a")
    locator.fill(date_str)
    locator.press("Enter")


def download_chunks():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=HEADLESS)
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()

        page.goto(URL, wait_until="networkidle", timeout=90000)
        page.wait_for_selector("table tbody tr", timeout=30000)

        # 1. Interval dropdown: "15-Min-Block" -> "Hourly"
        page.get_by_role("combobox").filter(has_text="15-Min-Block").click()
        page.get_by_role("option", name="Hourly").click()
        time.sleep(1.0)

        # 2. Delivery Period dropdown (default "Yesterday"; options: Yesterday, Today, Tomorrow, Last 8 Days,
        #    Last 31 Days, -Select Range-)  ->  "-Select Range-"  (reveals the From / To date inputs)
        delivery_period = page.get_by_role("combobox").nth(1)
        delivery_period.click()
        page.get_by_role("option", name="Select Range").click()
        page.get_by_placeholder("DD-MM-YYYY").first.wait_for(timeout=10000)
        time.sleep(1.0)

        for c_start, c_end in date_chunks(START_DATE, END_DATE, CHUNK_DAYS):
            prefix = f"iex_dam_{c_start.strftime('%Y%m%d')}_{c_end.strftime('%Y%m%d')}"
            if glob.glob(os.path.join(DOWNLOAD_DIR, f"{prefix}.*")):
                print(f"[SKIP] {prefix}")
                continue

            # Format matches the table's DD-MM-YYYY format (change to DD/MM/YYYY if input expects slashes)
            s_str = c_start.strftime("%d-%m-%Y")
            e_str = c_end.strftime("%d-%m-%Y")
            print(f"[FETCH] {s_str} -> {e_str}")

            try:
                # Locate the two date inputs that appear after selecting Custom/Range
                date_inputs = page.get_by_placeholder("DD-MM-YYYY")
                set_date_input(date_inputs.nth(0), s_str)
                set_date_input(date_inputs.nth(1), e_str)

                # Click "Update Report" and wait for network to settle
                page.get_by_role("button", name="Update Report").click()
                page.wait_for_load_state("networkidle", timeout=30000)
                time.sleep(1.5)

                # Click "Export" and save the file
                with page.expect_download(timeout=30000) as dl_info:
                    page.get_by_role("button", name="Export").click()
                    # If Export opens a dropdown (e.g., Excel / CSV), click Excel
                    excel_btn = page.get_by_text("Excel", exact=False).first
                    if excel_btn.is_visible(timeout=1500):
                        excel_btn.click()

                download = dl_info.value
                ext = os.path.splitext(download.suggested_filename)[1] or ".xlsx"
                save_path = os.path.join(DOWNLOAD_DIR, f"{prefix}{ext}")
                download.save_as(save_path)
                print(f"  -> Saved {os.path.basename(save_path)}")

            except Exception as e:
                print(f"  [ERROR] {s_str} to {e_str}: {e}")

            time.sleep(1.5)

        browser.close()


def merge_chunks(output_csv=str(DATA_DIR / "iex_dam_5yr_hourly.csv")):
    files = sorted(glob.glob(os.path.join(DOWNLOAD_DIR, "iex_dam_*.*")))
    if not files:
        print("No files to merge.")
        return

    dfs = []
    for f in files:
        try:
            raw = pd.read_csv(f, header=None) if f.endswith(".csv") else pd.read_excel(f, header=None)
            header_idx = 0
            for idx, row in raw.head(15).iterrows():
                vals = [str(v).strip().lower() for v in row.values]
                if "date" in vals and "hour" in vals:
                    header_idx = idx
                    break

            df = raw.iloc[header_idx + 1:].copy()
            df.columns = [str(c).strip() for c in raw.iloc[header_idx].values]

            if "Date" in df.columns:
                df["Date"] = df["Date"].ffill()
            if "Hour" in df.columns:
                df = df[pd.to_numeric(df["Hour"], errors="coerce").notna()]

            dfs.append(df)
        except Exception as e:
            print(f"[WARN] Skipped {os.path.basename(f)}: {e}")

    if dfs:
        final_df = pd.concat(dfs, ignore_index=True).drop_duplicates()
        final_df.to_csv(output_csv, index=False)
        print(f"\nMerged {len(files)} files -> {output_csv} ({len(final_df)} rows)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Download IEX DAM hourly data (default: 5 years ending 22-11-2026).")
    ap.add_argument("--start", help="start date YYYY-MM-DD (default: 5 years before the end date)")
    ap.add_argument("--end", help="end date YYYY-MM-DD (default: 2026-11-22)")
    ap.add_argument("--headless", action="store_true", help="run the browser without a window")
    args = ap.parse_args()
    if args.end:
        END_DATE = datetime.strptime(args.end, "%Y-%m-%d")
        START_DATE = END_DATE - timedelta(days=5 * 365)
    if args.start:
        START_DATE = datetime.strptime(args.start, "%Y-%m-%d")
    HEADLESS = args.headless
    download_chunks()
    merge_chunks()
