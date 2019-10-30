from __future__ import print_function

import os

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

# full URLS:
# https://ndownloader.figshare.com/files/18165911 # award notices
# https://ndownloader.figshare.com/files/18165908 # test
# https://ndownloader.figshare.com/files/18165905 # train

URLBASE = 'https://ndownloader.figshare.com/files/{}'
URLS = ['18165911', '18165905', '18165908']
DATA = ['award_notices_RAMP.csv', 'company_revenue_TRAIN.csv',
        'company_revenue_TEST.csv']

def main(output_dir='data'):
    filenames = DATA
    full_urls = [URLBASE.format(url) for url in URLS]

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for url, filename in zip(full_urls, filenames):
        output_file = os.path.join(output_dir, filename)

        if os.path.exists(output_file):
            continue

        print("Downloading from {} ...".format(url))
        urlretrieve(url, filename=output_file)
        print("=> File saved as {}".format(output_file))

if __name__ == '__main__':
    test = os.getenv('RAMP_TEST_MODE', 0)

    if test:
        print("Testing mode, not downloading any data.")
    else:
        main()