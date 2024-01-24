# This script took the output of vimeo_scrape.py and eliminated
# the duplicates. It also ensured that there was no overlap with
# the list of monitored videos in videos.txt.

# python3 vimeo_reduce.py > vimeo_unmonitored.txt

import re

def remove_comma_and_digits(s):
    pattern = r',\d+$'
    return re.sub(pattern, '', s)

vimeo_raw_unmonitored = open('vimeo_raw_unmonitored.txt', 'r')
raw_strings = vimeo_raw_unmonitored.readlines()
reduced_set = set(raw_strings)

vimeo_monitored = open('videos.txt', 'r')
monitored_strings = vimeo_monitored.readlines()
monitored_urls = []
for string in monitored_strings:
    monitored_urls.append(remove_comma_and_digits(string.strip()))

for string in reduced_set:
    url = remove_comma_and_digits(string.strip())
    if url not in monitored_urls:
        print(string.strip())
vimeo_raw_unmonitored.close()
vimeo_monitored.close()
