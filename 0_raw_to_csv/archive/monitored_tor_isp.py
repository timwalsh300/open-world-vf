import sys
import os
import dpkt
import socket
import re
import multiprocessing

VID_RE = re.compile('_\d+_')
VISIT_RE = re.compile('\d+_\d+/\d+_\d+_\d+')
REGION_RE = re.compile('[a-z]+_tor')
LABELS = open('/home/timothy.walsh/VF/labels.txt', 'r').readlines()
RELAYS = set(open('/home/timothy.walsh/VF/relays.txt', 'r').readlines())
# number of packets to parse
PACKETS = 5000

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
    region = REGION_RE.search(subdir[42:]).group()[:-4]
    output_csv = open('/home/timothy.walsh/VF/df_monitored_tor_isp/' + region + '_df_monitored_tor_isp.csv', 'w')
    for path, dirnames, filenames in os.walk(subdir):
        for filename in filenames:
            if filename == 'capture.pcap':
                print(multiprocessing.current_process().name, 'doing', path, flush = True)
                # do what follows for every pcap in the raw dataset
                input_pcap = dpkt.pcap.Reader(open(os.path.join(path, filename), 'rb'))
                output = []
                time_0 = 0.0
                try:
                    for timestamp, buf in input_pcap:
                        if time_0 == 0:
                            time_0 = timestamp
                        else:
                            if timestamp - time_0 > 10.0:
                                break;
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
                        # only append this direction if it's Tor traffic
                        if (distant_end + '\n') in RELAYS:
                            output.append(direction)
                except:
                    print('bad pcap:', path)
                    continue

                # write the labeled representation of the video stream to
                # the .csv file that we'll import into TensorFlow / PyTorch
                for i in range(PACKETS):
                    if i < len(output):
                        output_csv.write(str(output[i]) + ',')
                    else:
                        output_csv.write('0,')
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
                platform_genre = LABELS[int(vid_adjusted)].strip()
                visit = VISIT_RE.search(path).group()
                output_csv.write(region + ',' + 'all_tor' + ',' + 
                                 platform_genre + ',' + str(len(output)) +
                                 ',' + visit + ',' + vid_adjusted + '\n')
                files += 1       
    return (multiprocessing.current_process().name + ' did ' + str(files) + ' files from ' + region)
    
# this takes two command line arguments...
# first: full path to a directory containing subdirs to process
# second: number of CPUs available, should be <= number of subdirs
#
# in the same directory as this file, there must be a labels.txt
#
# output is a .csv file for each subdir, placed in the same root_path
# directory
if __name__ == '__main__':
    root_path = sys.argv[1]
    cpus = int(sys.argv[2])
    subdirs = [f.path for f in os.scandir(root_path) if f.is_dir()]
    with multiprocessing.Pool(cpus) as pool:
        iter = pool.imap_unordered(parse, subdirs)
        for i in iter:
            print(i)
