import sys
import os
import dpkt
import socket
import re
import multiprocessing

VID_RE = re.compile('_\d+_')
REGION_RE = re.compile('[a-z]+_tor')
LABELS = open('labels.txt', 'r').readlines()
# number of cells to parse
CELLS = 50000

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
    region = REGION_RE.search(subdir).group()[:-4]
    output_csv = open(subdir + '_wang.csv', 'w')
    for path, dirnames, filenames in os.walk(subdir):
        for filename in filenames:
            if filename == 'capture.pcap':
                print(multiprocessing.current_process().name, 'doing', path, flush = True)
                # do what follows for every pcap in the raw dataset
                input_pcap = dpkt.pcap.Reader(open(os.path.join(path, filename), 'rb'))
                conversations = {}
                time_0 = 0.0
                try:
                    for timestamp, buf in input_pcap:
                        try:
                            eth = dpkt.ethernet.Ethernet(buf)
                        except:
                            continue
                        ip = eth.data
                        # ignore ARP, etc.
                        if not isinstance(ip, dpkt.ip.IP):
                            continue
                        # ignore TCP ACK, RST
                        if ip.len <= 52:
                            continue
                        # determine the distant end and direction
                        if inet_to_str(ip.src)[:3] == '172':
                            distant_end = inet_to_str(ip.dst)
                            direction = 1
                        else:
                            distant_end = inet_to_str(ip.src)
                            direction = -1
                        # initialize the vector for a new distant end
                        if distant_end not in conversations:
                            conversations[distant_end] = []
                        # convert packet to cell(s) and append
                        cells = int((ip.len - 52) / 500)
                        for i in range(cells):
                            conversations[distant_end].append(direction)
                except:
                    pass

                # iterate over all the conversations to find the one with the
                # video stream traffic (or Tor connection)
                heavy_hitter = ''
                heavy_hitter_cells = 0
                for distant_end, cells in conversations.items():
                    if len(cells) > heavy_hitter_cells:
                        heavy_hitter_cells = len(cells)
                        heavy_hitter = distant_end

                # write the labeled representation of the video stream to
                # the .csv file that we'll import into TensorFlow / PyTorch
                #
                # also remove likely SENDME cells based on Wang's methodology
                sendme_counter = 0
                for i in range(CELLS):
                    if i < heavy_hitter_cells:
                        symbol = conversations[heavy_hitter][i]
                        # if counter is at 45 and this cell is outgoing,
                        # skip it and decrement the counter by 40
                        if sendme_counter == 45:
                            if symbol == 1:
                                sendme_counter -= 40
                                continue
                        output_csv.write(str(symbol) + ',')
                        # increment counter for outgoing cells
                        if symbol == 1:
                            sendme_counter += 1
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
                output_csv.write(region + ',' + heavy_hitter + ',' + 
                                 platform_genre + ',' + str(heavy_hitter_cells) +
                                 ',' + vid_adjusted + '\n')
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
