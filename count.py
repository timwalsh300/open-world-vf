import os
import re

vantage_points = ['virginia', 'oregon', 'seoul', 'sydney',
                  'london', 'frankfurt', 'stockholm',
                  'uae', 'brazil', 'africa']

VID_RE = re.compile('_\d+_')

for mode in ['tor', 'https']:
  site_dict = {}
  for i in range(240):
      site_dict[str(i)] = 0
  for region in vantage_points:
    print(region)
    for path, dirnames, filenames in os.walk('/data/timothy.walsh/July2023/monitored_' + mode + '/' + region + '_' + mode):
        if 'capture.pcap' in filenames:
            if len(filenames) < 4:
                print('bad capture', path)
        found = VID_RE.search(path)
        if found != None:
            site = found.group()[1:-1]
            if 240 <= int(site) <= 249:
                site_adjusted = str(int(site) - 240)
            elif 250 <= int(site) <= 259:
                site_adjusted = str(int(site) - 210)
            elif 260 <= int(site) <= 269:
                site_adjusted = str(int(site) - 180)
            elif 270 <= int(site) <= 279:
                site_adjusted = str(int(site) - 150)
            elif 280 <= int(site) <= 289:
                site_adjusted = str(int(site) - 120)
            elif 290 <= int(site) <= 299:
                site_adjusted = str(int(site) - 90)
            else:
                site_adjusted = site
            site_dict[site_adjusted] += 1

  for i in range(240):
    captures = site_dict[str(i)]
    print(str(i), captures, int(captures / 50) * '#')

  count = 0
  for path, dirnames, filenames in os.walk('/data/timothy.walsh/July2023/unmonitored_' + mode):
    if 'capture.pcap' in filenames:
        count += 1
        if len(filenames) < 4:
            print('bad capture', path)
  print(str(count), 'unmonitored captures,', mode)
