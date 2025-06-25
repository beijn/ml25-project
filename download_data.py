#%%
import http.client, urllib.request, gzip, shutil
from typing import Dict
from urllib.error import HTTPError
from pathlib import Path

HTTPHeaders = Dict[str, str]

listings_url = "https://data.insideairbnb.com/china/hk/hong-kong/2025-03-16/data/listings.csv.gz"
reviews_url = "https://data.insideairbnb.com/china/hk/hong-kong/2025-03-16/data/reviews.csv.gz"

def http_get_request(url: str, retry=5) -> bytes:
  headers = {}
  headers["Accept"] = '*/*'

  req = urllib.request.Request(
    url=url,
    headers=headers,
    method="GET",
  )
  try: res = urllib.request.urlopen(req)
  except HTTPError as e:
    if retry: return http_get_request(url, retry=retry-1)
    else: raise e
  if not isinstance(res, http.client.HTTPResponse): raise Exception("http_get_request: request somehow didn't result in an HTTPResponse.", url, res)
  if res.getcode() != 200: raise Exception("http_get_request: request didn't return 200 OK.", url, res, res.getcode())
  body = res.read()
  if not res.closed: res.close()
  return body

def download_file(url, into='./data/', overwrite=False):
  path = Path(into+'/'+Path(url).name)
  if not overwrite and path.exists(): return path
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_bytes(http_get_request(url))
  return path

def extract_file(path:Path, into='./data/'):
  with gzip.open(path, 'rb') as f_in:
    with open(into+'/'+Path(path).name[:-3], 'wb') as f_out:  
      shutil.copyfileobj(f_in, f_out)

def download_data():
  for url in [reviews_url, listings_url]:
    path = download_file(url)
    extract_file(path)
    