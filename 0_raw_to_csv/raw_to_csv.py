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
import math

# https://dpkt.readthedocs.io/en/latest/_modules/examples/print_packets.html
def inet_to_str(inet):
    try:
        return socket.inet_ntop(socket.AF_INET, inet)
    except ValueError:
        return socket.inet_ntop(socket.AF_INET6, inet)

# this is the function that each process executes, finding all the files
# in the given subdirectory, parsing each .pcap file according to the global
# variables at the bottom of this script, and writing a line to a .csv file
def parse(subdir):
    files = 0
    output_csv = open(OUTPUT_DIR + subdir[len(INPUT_DIR) + 1:] + '.csv', 'w')
    for path, dirnames, filenames in os.walk(subdir):
        for filename in filenames:
            if filename == 'capture.pcap':
                # do what follows for every pcap in the raw dataset
                print(multiprocessing.current_process().name, 'doing', path, flush = True)
                conversations = {}
                time_0 = 0.0
                try:
                    input_pcap = dpkt.pcap.Reader(open(os.path.join(path, filename), 'rb'))
                    # pull each frame from the .pcap file
                    for timestamp, buf in input_pcap:
                        if time_0 == 0.0:
                            time_0 = timestamp
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

                        ######################
                        # handle Tor traffic #
                        ######################
                        if PROTOCOL == 'tor':
                            # from the perspective of a network-level adversary, there
                            # is only one conversation for Tor, so initialize one
                            # only if the dictionary is empty
                            if not conversations:
                                conversations['tor'] = {'recv': [0 for i in range(POINTS)],
                                                        'sent': [0 for i in range(POINTS)],
                                                        'packets': [],
                                                        'first_packet': timestamp,
                                                        'last_packet': 0.0}
                            # if the distant end is not a relay that we found in the Tor
                            # consensus, continue with the next frame in the .pcap
                            if (distant_end + '\n') not in RELAYS:
                                continue
                            # convert artificially large packets, due to GRO and GSO pre-processing
                            # large bursts of traffic, to an estimate of the real number of packets
                            # that would be on the wire, assuming that TCP tries to fill packets up
                            # to the 1,500 byte MTU for Ethernet... e.g. a 3,200 byte "packet" would
                            # be two 1,500 byte packets and one 200 byte packet
                            real_packets = math.ceil(ip.len / 1500)
                            if REPRESENTATION == 'sirinam_wf' or REPRESENTATION == 'sirinam_vf':
                                for i in range(real_packets):
                                    conversations['tor']['packets'].append(direction)
                            if REPRESENTATION == 'rahman':
                                for i in range(real_packets):
                                    conversations['tor']['packets'].append(direction * (timestamp - time_0))
                            # Hayden collected his dataset with GRO and GSO turned on and used the
                            # resulting variation in packet sizes to his advantage
                            if REPRESENTATION == 'hayden':
                                conversations['tor']['packets'].append(direction * ip.len)
                            if 'schuster' in REPRESENTATION:
                                # determine the appropriate period
                                period = int((timestamp - time_0) / (1 / PPS))
                                transport = ip.data
                                if isinstance(transport, dpkt.tcp.TCP):
                                    # ignore TCP RST, ACK noise
                                    if ip.len > 52:
                                        # remove 20 + 32 bytes of IP and TCP headers, and
                                        # increment bytes for the appropriate period and direction
                                        if direction == -1:
                                            conversations['tor']['recv'][period] += ip.len - 52
                                        if direction == 1:
                                            conversations['tor']['sent'][period] += ip.len - 52
                            conversations['tor']['last_packet'] = timestamp

                        #############################
                        # handle HTTPS-only traffic #
                        #############################
                        else: # PROTOCOL == 'https'
                            if distant_end not in conversations:
                                conversations[distant_end] = {'recv': [0 for i in range(POINTS)],
                                                              'sent': [0 for i in range(POINTS)],
                                                              'packets': [],
                                                              'first_packet': timestamp,
                                                              'last_packet': 0.0}
                            # convert artificially large packets, due to GRO and GSO pre-processing
                            # large bursts of traffic, to an estimate of the real number of packets
                            # that would be on the wire, assuming that TCP tries to fill packets up
                            # to the 1,500 byte MTU for Ethernet... e.g. a 3,200 byte "packet" would
                            # be two 1,500 byte packets and one 200 byte packet
                            real_packets = math.ceil(ip.len / 1500)
                            if REPRESENTATION == 'sirinam_wf' or REPRESENTATION == 'sirinam_vf':
                                for i in range(real_packets):
                                    conversations[distant_end]['packets'].append(direction)
                            if REPRESENTATION == 'rahman':
                                for i in range(real_packets):
                                    conversations[distant_end]['packets'].append(direction * (timestamp - time_0))
                            # Hayden collected his dataset with GRO and GSO turned on and used the
                            # resulting variation in packet sizes to his advantage
                            if REPRESENTATION == 'hayden':
                                conversations[distant_end]['packets'].append(direction * ip.len)
                            if 'schuster' in REPRESENTATION:
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

                except Exception as e:
                    print(e)
                    continue

                ########################################################################################
                # now transform the 'conversations' dictionary into an 'output' list and 'output_size' #
                ########################################################################################
                if PROTOCOL == 'tor':
                    heavy_hitter = 'tor'
                    if 'dschuster' in REPRESENTATION:
                        output = [0 for i in range(2 * POINTS)]
                        for i in range(POINTS):
                            output[i * 2] += conversations['tor']['sent'][i]
                            output[(i * 2) + 1] += conversations['tor']['recv'][i]
                        # save the number of bytes for statistical analysis later
                        output_size = sum(output)
                    elif 'schuster' in REPRESENTATION:
                        output = [0 for i in range(POINTS)]
                        for i in range(POINTS):
                            output[i] += conversations['tor']['sent'][i]
                            output[i] += conversations['tor']['recv'][i]
                        # save the number of bytes for statistical analysis later
                        output_size = sum(output)
                    else: # sirinam_wf, sirinam_vf, rahman, or hayden
                        # save the number of packets for statistical analysis later
                        output_size = len(conversations['tor']['packets'])
                        if len(conversations['tor']['packets']) >= POINTS:
                            output = conversations['tor']['packets'][:POINTS]
                        # pad shorter lengths with zeros
                        else:
                            output = conversations['tor']['packets']
                            for i in range(POINTS - len(conversations['tor']['packets'])):
                                output.append(0)

                else: # PROTOCOL == 'https'
                    heavy_hitter = ''
                    heavy_hitter_product = 0.0
                    if 'dschuster' in REPRESENTATION:
                        # find heavy_hitter
                        for distant_end, flow_info in conversations.items():
                            duration = (flow_info['last_packet'] - flow_info['first_packet'])
                            if sum(flow_info['recv']) * duration > heavy_hitter_product:
                                heavy_hitter_product = sum(flow_info['recv']) * duration
                                heavy_hitter = distant_end
                        output = [0 for i in range(2 * POINTS)]
                        for i in range(POINTS):
                            output[i * 2] += conversations[heavy_hitter]['sent'][i]
                            output[(i * 2) + 1] += conversations[heavy_hitter]['recv'][i]
                        # save the number of bytes for statistical analysis later
                        output_size = sum(output)
                    elif 'schuster' in REPRESENTATION:
                        # find heavy_hitter
                        for distant_end, flow_info in conversations.items():
                            duration = (flow_info['last_packet'] - flow_info['first_packet'])
                            if sum(flow_info['recv']) * duration > heavy_hitter_product:
                                heavy_hitter_product = sum(flow_info['recv']) * duration
                                heavy_hitter = distant_end
                        output = [0 for i in range(POINTS)]
                        for i in range(POINTS):
                            output[i] += conversations[heavy_hitter]['sent'][i]
                            output[i] += conversations[heavy_hitter]['recv'][i]
                        # save the number of bytes for statistical analysis later
                        output_size = sum(output)
                    else: # sirinam_wf, sirinam_vf, rahman, or hayden
                        # find heavy_hitter
                        for distant_end, flow_info in conversations.items():
                            duration = (flow_info['last_packet'] - flow_info['first_packet'])
                            if len(flow_info['packets']) * duration > heavy_hitter_product:
                                heavy_hitter_product = len(flow_info['packets']) * duration
                                heavy_hitter = distant_end
                        # save the number of packets for statistical analysis later
                        output_size = len(conversations[heavy_hitter]['packets'])
                        if len(conversations[heavy_hitter]['packets']) >= POINTS:
                            output = conversations[heavy_hitter]['packets'][:POINTS]
                        # pad shorter lengths with zeros
                        else:
                            output = conversations[heavy_hitter]['packets']
                            for i in range(POINTS - len(conversations[heavy_hitter]['packets'])):
                                output.append(0)

                #######################################################
                # write the 'output' list and labels to the .csv file #
                #######################################################
                for i in range(len(output)):
                    output_csv.write(str(output[i]) + ',')
                if MONITORED:
                    region = REGION_RE.search(subdir[len(INPUT_DIR):]).group()[:-1]
                else:
                    region = 'oregon'
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

VID_RE = re.compile(r'_\d+_')
VISIT_RE = re.compile(r'\d+_\d+/\d+_\d+_\d+')
REGION_RE = re.compile(r'[a-z]+_')
# this file contains a 'region,genre' line for each video ID in order
LABELS = open('labels.txt', 'r').readlines()
# this file contains all the Tor relay IP addresses for May-August 2023
RELAYS = set(open('relays.txt', 'r').readlines())
REPRESENTATION = sys.argv[3]
MONITORED = True if sys.argv[4] == 'monitored' else False
PROTOCOL = sys.argv[5]
OUTPUT_DIR = (REPRESENTATION + '/' + sys.argv[4] + '_' + PROTOCOL + '/')
print('output directory will be', OUTPUT_DIR)
MAX_TIME = 0.0
POINTS = 0
# periods per second for Schuster (e.g. 4 for 1/4-second periods)
PPS = 0
if PROTOCOL == 'tor' and REPRESENTATION == 'sirinam_wf':
    MAX_TIME = 10.0
    # here POINTS is the max number of packets to write out
    POINTS = 5000
if PROTOCOL == 'https' and REPRESENTATION == 'sirinam_wf':
    MAX_TIME = 5.0
    POINTS = 5000
if REPRESENTATION == 'sirinam_vf' or REPRESENTATION == 'rahman' or REPRESENTATION == 'hayden':
    MAX_TIME = 240.0
    POINTS = 25000
if 'schuster' in REPRESENTATION:
    MAX_TIME = 240.0
    # here POINTS is the number of periods
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
