# This script took the output of vimeo_scrape.py and eliminated
# the duplicates. It also ensured that there was no overlap with
# the list of monitored videos in videos.txt

vimeo_raw_unmonitored = open('vimeo_raw_unmonitored.txt', 'r')
raw_strings = vimeo_raw_unmonitored.readlines()
reduced_set = set(raw_strings)

vimeo_monitored = open('videos.txt', 'r')
monitored_strings = vimeo_monitored.readlines()

for string in reduced_set:
    if string not in monitored_strings:
        print(string.strip())
vimeo_raw_unmonitored.close()
vimeo_monitored.close()
