# This script produced Figure 3.1 to show the effects of Generic Receive Offload
# and Generic Segmentation Offload

import dpkt
import matplotlib.pyplot as plt
import numpy as np

def read_pcap(file_name):
    print(file_name)
    sizes = []
    try:
        with open(file_name, 'rb') as f:
            pcap = dpkt.pcap.Reader(f)
            for ts, buf in pcap:
                # Make sure we have enough bytes to access IP header length field
                if len(buf) < 34:  # 14 bytes for Ethernet + 20 for IP header
                    continue

                # Extract the IP packet length
                eth = dpkt.ethernet.Ethernet(buf)
                if isinstance(eth.data, dpkt.ip.IP):
                    total_length = int.from_bytes(buf[16:18], byteorder='big')
                    sizes.append(total_length)
    except Exception as e:
        pass
    return sizes

def cdf(data):
    data_sorted = np.sort(data)
    p = 1. * np.arange(len(data)) / (len(data) - 1)
    return data_sorted, p

# Update these file names with your actual file names
files_https_videos_on = [f'https-videos-on/0_{i}_0/capture.pcap' for i in range(6)]
files_https_videos_off = [f'https-videos-off/0_{i}_0/capture.pcap' for i in range(6)]
files_tor_videos_on = [f'tor-videos-on/0_{i}_0/capture.pcap' for i in range(6)]
files_tor_videos_off = [f'tor-videos-off/0_{i}_0/capture.pcap' for i in range(6)]
files_https_homepages_on = [f'https-homepages-on/0_{i}_0/capture.pcap' for i in range(20)]
files_https_homepages_off = [f'https-homepages-off/0_{i}_0/capture.pcap' for i in range(20)]
files_tor_homepages_on = [f'tor-homepages-on/0_{i}_0/capture.pcap' for i in range(20)]
files_tor_homepages_off = [f'tor-homepages-off/0_{i}_0/capture.pcap' for i in range(20)]

# Aggregate sizes from all files for each scenario
sizes_https_videos_on = [size for file in files_https_videos_on for size in read_pcap(file)]
sizes_https_videos_off = [size for file in files_https_videos_off for size in read_pcap(file)]
sizes_tor_videos_on = [size for file in files_tor_videos_on for size in read_pcap(file)]
sizes_tor_videos_off = [size for file in files_tor_videos_off for size in read_pcap(file)]
sizes_https_homepages_on = [size for file in files_https_homepages_on for size in read_pcap(file)]
sizes_https_homepages_off = [size for file in files_https_homepages_off for size in read_pcap(file)]
sizes_tor_homepages_on = [size for file in files_tor_homepages_on for size in read_pcap(file)]
sizes_tor_homepages_off = [size for file in files_tor_homepages_off for size in read_pcap(file)]

# Calculate CDFs
sizes_sorted_https_videos_on, p_https_videos_on = cdf(sizes_https_videos_on)
sizes_sorted_https_videos_off, p_https_videos_off = cdf(sizes_https_videos_off)
sizes_sorted_tor_videos_on, p_tor_videos_on = cdf(sizes_tor_videos_on)
sizes_sorted_tor_videos_off, p_tor_videos_off = cdf(sizes_tor_videos_off)
sizes_sorted_https_homepages_on, p_https_homepages_on = cdf(sizes_https_homepages_on)
sizes_sorted_https_homepages_off, p_https_homepages_off = cdf(sizes_https_homepages_off)
sizes_sorted_tor_homepages_on, p_tor_homepages_on = cdf(sizes_tor_homepages_on)
sizes_sorted_tor_homepages_off, p_tor_homepages_off = cdf(sizes_tor_homepages_off)

# Plotting
#plt.plot(sizes_sorted_https_homepages_on, p_https_homepages_on, label='HTTPS Homepages GSO/GRO On')
#plt.plot(sizes_sorted_https_homepages_off, p_https_homepages_off, label='HTTPS Homepages GSO/GRO Off')
plt.plot(sizes_sorted_tor_videos_on, p_tor_videos_on, label='Tor Videos GSO/GRO On')
plt.plot(sizes_sorted_tor_videos_off, p_tor_videos_off, label='Tor Videos GSO/GRO Off')
#plt.plot(sizes_sorted_tor_homepages_on, p_tor_homepages_on, label='Tor Homepages GSO/GRO On')
#plt.plot(sizes_sorted_tor_homepages_off, p_tor_homepages_off, label='Tor Homepages GSO/GRO Off')
plt.plot(sizes_sorted_https_videos_on, p_https_videos_on, label='HTTPS Videos GSO/GRO On')
plt.plot(sizes_sorted_https_videos_off, p_https_videos_off, label='HTTPS Videos GSO/GRO Off')

plt.xlabel('Packet Size (bytes)')
plt.ylabel('CDF')
plt.title('CDF of Packet Sizes Showing the Effect of Offloading')
plt.legend()
plt.grid(True)

# Save the plot to a PNG file
plt.savefig('cdf_plot.png', dpi=300)

# Optionally, if you also want to see the plot
# plt.show()
