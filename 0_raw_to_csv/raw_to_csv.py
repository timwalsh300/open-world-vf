# this takes five command line arguments...
# 1: full path to the root directory for the raw dataset
# 2: number of CPUs available
# 3: sirinam_wf, sirinam_vf, rahman, hayden, schuster2, schuster4, schuster8, schuster16, dschuster8, dschuster16
# 4: monitored or unmonitored
# 5: tor or https
#
# output is a set of .csv files to e.g. sirinam_wf/monitored_tor/

import sys
import os
import dpkt
import socket
import re
import multiprocessing

# https://dpkt.readthedocs.io/en/latest/_modules/examples/print_packets.html
def inet_to_str(inet):
    try:
        return socket.inet_ntop(socket.AF_INET, inet)
    except ValueError:
        return socket.inet_ntop(socket.AF_INET6, inet)

# this is the function that each process executes, finding all the files
# in the given subdirectory, parsing each .pcap file according to the
# constants above and the logic below, and writing a line to a .csv file
def parse(subdir):
    files = 0
    output_csv = open(OUTPUT_DIR + subdir[len(INPUT_DIR) + 1:] + '.csv', 'w')
    for path, dirnames, filenames in os.walk(subdir):
        for filename in filenames:
            if filename == 'capture.pcap':
                # do what follows for every pcap in the raw dataset
                print(multiprocessing.current_process().name, 'doing', path, flush = True)
                output = []
                conversations = {}
                time_0 = 0.0
                packets_written = 0
                try:
                    input_pcap = dpkt.pcap.Reader(open(os.path.join(path, filename), 'rb'))
                    for timestamp, buf in input_pcap:
                        if time_0 == 0.0:
                            time_0 = timestamp
                        else:
                            if timestamp - time_0 > MAX_TIME:
                                break
                        try:
                            eth = dpkt.ethernet.Ethernet(buf)
                        except:
                            print('bad frame')
                            continue
                        ip = eth.data
                        # ignore ARP, etc.
                        if not isinstance(ip, dpkt.ip.IP):
                            continue
                        # determine the distant end and direction
                        if inet_to_str(ip.src)[:7] == '172.31.':
                            distant_end = inet_to_str(ip.dst)
                            direction = 1
                        else:
                            distant_end = inet_to_str(ip.src)
                            direction = -1
                        if PROTOCOL == 'tor' and (distant_end + '\n') not in RELAYS:
                            continue
                        if REPRESENTATION == 'sirinam_wf' or REPRESENTATION == 'sirinam_vf':
                            output.append(direction)
                            packets_written += 1
                            if packets_written == POINTS:
                                break
                        if REPRESENTATION == 'rahman':
                            output.append(direction * (timestamp - time_0))
                            packets_written += 1
                            if packets_written == POINTS:
                                break
                        if REPRESENTATION == 'hayden':
                            output.append(direction * ip.len)
                            packets_written += 1
                            if packets_written == POINTS:
                                break
                        if 'schuster' in REPRESENTATION:
                            # initialize the vectors for a new distant end with the
                            # number of POINTS all set to 0, record the
                            # timestamp of the first packet, and initialize the
                            # timestamp of the last packet that we'll update on
                            # each appearance of this distant end
                            if distant_end not in conversations:
                                conversations[distant_end] = {'recv': [0 for i in range(POINTS)],
                                                              'sent': [0 for i in range(POINTS)],
                                                              'first_packet': timestamp,
                                                              'last_packet': 0.0}
                            # determine the appropriate period
                            period = int((timestamp - time_0) / (1 / PPS))
                            transport = ip.data
                            if isinstance(transport, dpkt.tcp.TCP):
                                # ignore TCP RST, ACK noise
                                if ip.len > 52:
                                    # remove 20 + 32 bytes of IP and TCP headers, and
                                    # increment bytes for the appropriate period and direction
                                    if direction == -1:
                                        conversations[distant_end]['recv'][period] += ip.len - 52
                                    if direction == 1:
                                        conversations[distant_end]['sent'][period] += ip.len - 52
                                    conversations[distant_end]['last_packet'] = timestamp
                            if isinstance(transport, dpkt.udp.UDP):
                                if ip.len > 28:
                                    # remove 20 + 8 bytes of IP and UDP headers, and
                                    # increment bytes for the appropriate period and direction
                                    if direction == -1:
                                        conversations[distant_end]['recv'][period] += ip.len - 28
                                    if direction == 1:
                                        conversations[distant_end]['sent'][period] += ip.len - 28
                                    conversations[distant_end]['last_packet'] = timestamp
                except:
                    print('bad pcap:', path)
                    continue

                # find the heavy hitter when using the schuster method so that
                # we can compute some interesting statistics later
                heavy_hitter = 'null'
                if 'schuster' in REPRESENTATION:
                    # iterate over all the conversations to find the one with the
                    # video stream traffic
                    heavy_hitter_product = 0.0
                    for distant_end, flow_info in conversations.items():
                        duration = (flow_info['last_packet'] - flow_info['first_packet'])
                        if (sum(flow_info['recv']) + sum(flow_info['sent'])) * duration > heavy_hitter_product:
                            heavy_hitter_product = (sum(flow_info['recv']) + sum(flow_info['sent'])) * duration
                            heavy_hitter = distant_end
                    # abort if the heaviest flow transferred less than 1 MB, which is
                    # implausible for a video, and probably means it used a Tor connection
                    # over port 22, triggering our stupid tcpdump filter
                    if sum(conversations[heavy_hitter]['recv']) < 1000000:
                        print('<1 MB:', path)
                        continue
                if 'schuster' in REPRESENTATION and PROTOCOL == 'https':
                    for i in range(POINTS):
                        # two channels representing bytes sent and received per period...
                        # values in output alternate sent and received, with every pair
                        # representing one time period
                        if 'dschuster' in REPRESENTATION:
                            output.append(conversations[heavy_hitter]['sent'][i])
                            output.append(conversations[heavy_hitter]['recv'][i])
                        # just combine sent and received into total bytes per period
                        else:
                            output.append(conversations[heavy_hitter]['sent'][i] + conversations[heavy_hitter]['recv'][i])
                        
                if 'schuster' in REPRESENTATION and PROTOCOL == 'tor':
                    # merge the Tor conversations into one output
                    if 'dschuster' in REPRESENTATION:
                        output = [0 for i in range(2 * POINTS)]
                        for distant_end, flow_info in conversations.items():
                            for i in range(POINTS):
                                output[i * 2] += flow_info['sent'][i]
                                output[(i * 2) + 1] += flow_info['recv'][i]
                    else:
                        output = [0 for i in range(POINTS)]
                        for distant_end, flow_info in conversations.items():
                            for i in range(POINTS):
                                output[i] += flow_info['sent'][i]
                                output[i] += flow_info['recv'][i]

                # write the labeled representation of the video stream to
                # the .csv file that we'll read with Pandas
                if 'dschuster' in REPRESENTATION:
                    for i in range(2 * POINTS):
                        output_csv.write(str(output[i]) + ',')
                else:
                    for i in range(POINTS):
                        if i < len(output):
                            output_csv.write(str(output[i]) + ',')
                        else:
                            output_csv.write('0,')
                        
                if MONITORED:
                    region = REGION_RE.search(subdir[len(INPUT_DIR):]).group()[:-1]
                else:
                    region = 'oregon'
                # total bytes represented for Schuster, and total packets for Sirinam, Rahman, or Hayden
                output_size = sum(output) if 'schuster' in REPRESENTATION else len(output)
                visit = VISIT_RE.search(path).group()
                vid = VID_RE.search(path).group()[1:-1]
                if 240 <= int(vid) <= 249:
                    vid_adjusted = str(int(vid) - 240)
                elif 250 <= int(vid) <= 259:
                    vid_adjusted = str(int(vid) - 210)
                elif 260 <= int(vid) <= 269:
                    vid_adjusted = str(int(vid) - 180)
                elif 270 <= int(vid) <= 279:
                    vid_adjusted = str(int(vid) - 150)
                elif 280 <= int(vid) <= 289:
                    vid_adjusted = str(int(vid) - 120)
                elif 290 <= int(vid) <= 299:
                    vid_adjusted = str(int(vid) - 90)
                else:
                    vid_adjusted = vid
                if not MONITORED:
                    vid_adjusted = '240'
                platform_genre = LABELS[int(vid_adjusted)].strip() if MONITORED else 'vimeo,unmonitored'
                output_csv.write(region + ',' + heavy_hitter + ',' + 
                                 platform_genre + ',' + str(output_size) +
                                 ',' + visit + ',' + vid_adjusted + '\n')
                files += 1       
    return (multiprocessing.current_process().name + ' did ' + str(files) + ' files from ' + region)
    
VID_RE = re.compile('_\d+_')
VISIT_RE = re.compile('\d+_\d+/\d+_\d+_\d+')
REGION_RE = re.compile('[a-z]+_')
# this file contains a 'region,genre' line for each video ID in order
LABELS = open('/home/timothy.walsh/VF/0_raw_to_csv/labels.txt', 'r').readlines()
# this file contains all the Tor relay IP addresses for May-August 2023
RELAYS = set(open('/home/timothy.walsh/VF/0_raw_to_csv/relays.txt', 'r').readlines())
REPRESENTATION = sys.argv[3]
MONITORED = True if sys.argv[4] == 'monitored' else False
PROTOCOL = sys.argv[5]
OUTPUT_DIR = ('/home/timothy.walsh/VF/0_raw_to_csv/' + REPRESENTATION + '/' +
              sys.argv[4] + '_' + PROTOCOL + '/')
print('output directory will be', OUTPUT_DIR)
MAX_TIME = 0.0
POINTS = 0
# periods per second for Schuster (e.g. 4 for 1/4-second periods)
PPS = 0
if PROTOCOL == 'tor' and REPRESENTATION == 'sirinam_wf':
    MAX_TIME = 10.0
    # here POINTS is the max number of packets to look at
    POINTS = 5000
if PROTOCOL == 'https' and REPRESENTATION == 'sirinam_wf':
    MAX_TIME = 5.0
    POINTS = 5000
if REPRESENTATION == 'sirinam_vf' or REPRESENTATION == 'rahman' or REPRESENTATION == 'hayden':
    MAX_TIME = 240.0
    POINTS = 25000
if 'schuster' in REPRESENTATION:
    MAX_TIME = 240.0
    # here POINTS is the number of periods or time slices
    if str.isdigit(REPRESENTATION[-2]):
        PPS = int(REPRESENTATION[-2:])
    else:
        PPS = int(REPRESENTATION[-1])
    POINTS = int(MAX_TIME) * PPS
if MAX_TIME == 0.0:
    print('something is wrong with the given PROTOCOL or REPRESENTATION')
INPUT_DIR = sys.argv[1] + '/' + sys.argv[4] + '_' + PROTOCOL
cpus = int(sys.argv[2])
subdirs = [f.path for f in os.scandir(INPUT_DIR) if f.is_dir()]
with multiprocessing.Pool(cpus) as pool:
    iter = pool.imap_unordered(parse, subdirs)
    for i in iter:
        print(i)
