import re
import requests
import os
import zipfile


'''
Google Drive no longer serves large files through docs.google.com/uc with a
download_warning cookie. It answers with an HTML confirmation form instead,
which has to be resubmitted to drive.usercontent.google.com.
'''
DOWNLOAD_URL = "https://drive.usercontent.google.com/download"


def download_file_from_google_drive(id, destination):
    session = requests.Session()

    params = {'id': id, 'export': 'download', 'confirm': 't'}
    response = session.get(DOWNLOAD_URL, params=params, stream=True)

    if is_html_response(response):
        response = session.get(DOWNLOAD_URL, params=parse_form_fields(response.text), stream=True)

    save_response_content(response, destination)

def is_html_response(response):
    if 'Content-Type' not in response.headers:
        return False

    return response.headers['Content-Type'].startswith('text/html')

def parse_form_fields(html):
    return dict(re.findall(r'name="([^"]+)"\s+value="([^"]*)"', html))

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

if __name__ == "__main__":
    file_id = '1mVGfN0Xoj--26_NIPgXS48ub-Rr89GIw'
    destination = os.path.dirname(os.path.abspath(__file__)) + '/networks.zip'
    extract_destination = os.path.dirname(os.path.abspath(__file__)) + '/nn'
    print('Starting networks download')
    if os.path.isfile(destination):
        print('Networks were already downloaded')
    else:
        download_file_from_google_drive(file_id, destination)
        print('Networks were succesfully downloaded.')
        with zipfile.ZipFile(destination, 'r') as zip_ref:
            zip_ref.extractall(extract_destination)
