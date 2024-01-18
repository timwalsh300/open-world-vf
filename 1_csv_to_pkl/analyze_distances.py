# This script produced the data that we showed in
# Figures 3.8, 3.9, and 3.10 in our analysis of the
# differences between geographic vantage points, in
# terms of the distances between our clients and the
# servers to which they connected

import pandas as pd
import numpy as np
import geoip2.database
from geopy.distance import geodesic
import matplotlib.pyplot as plt

location_coords = {
    'africa': (-33.924870, 18.424055),
    'brazil': (-23.550520, -46.633308),
    'frankfurt': (50.110924, 8.682127),
    'london': (51.507351, -0.127758),
    'oregon': (45.839722, -119.700556),
    'seoul': (37.532600, 127.024612),
    'stockholm': (59.334591, 18.063240),
    'sydney': (-33.865143, 151.209900),
    'uae': (24.466667, 54.366669),
    'virginia': (38.944444, -77.455833)
}

city_db_reader =  geoip2.database.Reader('GeoLite2-City.mmdb')

missed = set()
found = set()
cities = set()
countries = set()

# Function to get coordinates from IP Address.
def get_coordinates(ip):
    try:
        response = city_db_reader.city(ip)
        if response.city.name is None:
            missed.add(ip)
            countries.add(response.country.name)
            return (np.nan, np.nan)
        else:
            found.add(ip)
            cities.add(response.city.name + ', ' + response.country.name)
            return (response.location.latitude, response.location.longitude)
    except:
        return (np.nan, np.nan)

for protocol in ['https', 'tor']:
    for platform in ['youtube', 'facebook', 'vimeo', 'rumble']:
        # Read the CSV file.
        print('Starting', protocol, platform)
        df = pd.read_csv('dschuster8_monitored_' + protocol + '_' + platform + '.csv',
                         header=None)
        df.rename(columns={3840: 'location', 3841: 'heavy_hitter'}, inplace=True)
        df['coordinates'] = df['heavy_hitter'].apply(get_coordinates)
        df['distance'] = df.apply(lambda row: geodesic(location_coords[row['location']], row['coordinates']).kilometers if row['coordinates'] != (np.nan, np.nan) else np.nan, axis=1)
        stats = df.groupby('location')['distance'].agg([('min_distance', 'min'),
                                                        ('percentile_25', lambda x: x.quantile(0.25)),
                                                        ('median_distance', 'median'),
                                                        ('percentile_75', lambda x: x.quantile(0.75)),
                                                        ('max_distance', 'max')])
        #unique_ips_by_location = df.groupby('location')['heavy_hitter'].unique()
        #for location, ips in unique_ips_by_location.items():
        #    print(f"{location}: {list(ips)}")
        print(stats)

        print(str(len(found)), 'IP addresses resolved down to the city level...')
        print(cities)
        print(str(len(missed)), 'IP addresses resolved only down to the country level...')
        #print(missed)
        print(countries)
        print('\n')
        missed.clear()
        found.clear()
        cities.clear()
        countries.clear()

#        fig, ax = plt.subplots()
#        stats.plot(kind='bar', y='mean', yerr='std', ax=ax, capsize=4)
#        plt.ylabel('Distance (km)')
#        plt.title('Mean Distances and StdDev by Location')
#        plt.tight_layout()
#        plt.savefig('analyze_distances_' + protocol + '_' + platform + '_distances.pdf')
