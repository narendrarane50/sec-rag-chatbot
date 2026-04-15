import time
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from tqdm import tqdm
from bs4 import XMLParsedAsHTMLWarning
import warnings
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

HEADERS = {
    "User-Agent": "Narendra Rane narendrarane50@gmail.com",
    "Accept-Encoding": "gzip, deflate",
}

OUTPUT_DIR = Path("sec_filings")
OUTPUT_DIR.mkdir(exist_ok=True)

COMPANIES = {
    "AAPL":  "0000320193",
    "MSFT":  "0000789019",
    "GOOGL": "0001652044",
    "AMZN":  "0001018724",
    "META":  "0001326801",
    "NVDA":  "0001045810",
    "TSLA":  "0001318605",
    "JPM":   "0000019617",
    "BAC":   "0000070858",
    "GS":    "0000886982",
}

def get_filing_list(cik, form_type="10-K", count=2):
    cik_padded = cik.lstrip("0").zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    data = resp.json()

    filings = []
    recent = data.get("filings", {}).get("recent", {})
    forms      = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    dates      = recent.get("filingDate", [])

    for form, accession, date in zip(forms, accessions, dates):
        if form == form_type:
            filings.append((accession, date))
            if len(filings) >= count:
                break
    return filings

def get_doc_url(cik, accession_number):
    cik_int = str(int(cik))
    acc_clean = accession_number.replace("-", "")
    index_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_clean}/{accession_number}-index.htm"

    resp = requests.get(index_url, headers=HEADERS)
    if resp.status_code != 200:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # EDGAR index page has a table with columns: Seq, Description, Document, Type, Size
    # We want the row where Type == "10-K"
    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) >= 4:
                # Type is usually the 4th column (index 3)
                row_type = cells[3].get_text(strip=True)
                if row_type == "10-K":
                    link = cells[2].find("a")
                    if link and link.get("href"):
                        href = link["href"]
                        # if not href.startswith("http"):
                        #     href = f"https://www.sec.gov{href}"
                        # return href

                        if not href.startswith("http"):
                            href = f"https://www.sec.gov{href}"
                        # Strip the EDGAR inline viewer wrapper: /ix?doc=/Archives/... → /Archives/...
                        if "/ix?doc=" in href:
                            href = "https://www.sec.gov" + href.split("/ix?doc=")[1]
                        return href

    # Fallback: grab the first .htm link that looks like the main doc
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if href.endswith(".htm") and acc_clean in href:
            if not href.startswith("http"):
                href = f"https://www.sec.gov{href}"
            if "/ix?doc=" in href:
                href = "https://www.sec.gov" + href.split("/ix?doc=")[1]
            return href

    return None

def download_filing(ticker, cik, accession, date):
    doc_url = get_doc_url(cik, accession)
    if not doc_url:
        print(f"  [!] Could not find primary doc for {ticker} {date}")
        return None

    print(f"  Fetching: {doc_url}")
    resp = requests.get(doc_url, headers=HEADERS)
    if resp.status_code != 200:
        print(f"  [!] HTTP {resp.status_code}")
        return None

    soup = BeautifulSoup(resp.text, "lxml")
    for tag in soup(["script", "style", "ix:header", "ix:hidden"]):
        tag.decompose()

    lines = [l.strip() for l in soup.get_text(separator="\n").splitlines() if l.strip()]
    clean_text = "\n".join(lines)

    company_dir = OUTPUT_DIR / ticker
    company_dir.mkdir(exist_ok=True)
    out_path = company_dir / f"{ticker}_10K_{date}.txt"
    out_path.write_text(clean_text, encoding="utf-8")
    return out_path

def download_all(companies, filings_per_company=2):
    results = {}
    for ticker, cik in tqdm(companies.items(), desc="Companies"):
        print(f"\n── {ticker} (CIK: {cik})")
        filings = get_filing_list(cik, count=filings_per_company)
        saved = []
        for accession, date in filings:
            print(f"  Downloading 10-K filed {date}...")
            path = download_filing(ticker, cik, accession, date)
            if path:
                size_kb = path.stat().st_size // 1024
                print(f"  Saved → {path}  ({size_kb:,} KB)")
                saved.append(str(path))
            time.sleep(0.5)
        results[ticker] = saved
    return results

if __name__ == "__main__":
    all_files = download_all(COMPANIES, filings_per_company=2)
    print("\n\n── Download summary ──")
    total = sum(len(v) for v in all_files.values())
    print(f"Downloaded {total} filings across {len(COMPANIES)} companies")
    for ticker, paths in all_files.items():
        print(f"  {ticker}: {len(paths)} filing(s)")