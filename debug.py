# debug.py
import requests

HEADERS = {
    "User-Agent": "Narendra Rane narendrarane50@gmail.com",
    "Accept-Encoding": "gzip, deflate",
}

# Step 1: get the actual accession numbers from submissions API
cik_padded = "0000320193"
url = f"https://data.sec.gov/submissions/CIK{cik_padded}.json"
resp = requests.get(url, headers=HEADERS)
data = resp.json()

recent = data["filings"]["recent"]
forms      = recent["form"]
accessions = recent["accessionNumber"]
dates      = recent["filingDate"]

# Print first 5 10-K filings found
count = 0
for form, acc, date in zip(forms, accessions, dates):
    if form == "10-K":
        print(f"Date: {date}  Accession: {acc}")
        count += 1
        if count == 3:
            break

# Step 2: try fetching the index for the first one
acc = accessions[[i for i,f in enumerate(forms) if f=="10-K"][0]]
cik_int = "320193"
acc_clean = acc.replace("-", "")
print(f"\nTrying index for: {acc}")

# Try -index.htm (not index.json)
index_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_clean}/{acc}-index.htm"
print(f"URL: {index_url}")
resp2 = requests.get(index_url, headers=HEADERS)
print(f"Status: {resp2.status_code}")
print(resp2.text[:3000])