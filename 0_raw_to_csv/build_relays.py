import sys
import os

def read(hour, relay_ip_addresses):
    with open(hour, 'r') as f:
        lines = f.readlines()
        ip = ''
        for i in range(0, len(lines)):
            line = lines[i]
            parts = line.strip().split(' ')

            if parts[0] == 'r':
                ip = parts[6]
                if ip not in relay_ip_addresses:
                    relay_ip_addresses.add(ip)
                    
if __name__ == "__main__":
    path_to_consensuses = sys.argv[1]
    relay_ip_addresses = set()
    for root, subFolders, files in os.walk(path_to_consensuses):
        for file in files:
            hour = root + '/' + file
            read(hour, relay_ip_addresses)
    for address in relay_ip_addresses:
        print(address)
