FROM mono:latest

WORKDIR /app

# --- FIX START ---
# 1. Debian Buster ist EOL -> Wir müssen auf die Archiv-Server wechseln, sonst gibt es 404 Fehler.
# 2. Wir nutzen "-o Acquire::Check-Valid-Until=false", da die Archiv-Signaturen veraltet sind.
RUN echo "deb http://archive.debian.org/debian buster main" > /etc/apt/sources.list && \
    echo "deb http://archive.debian.org/debian-security buster/updates main" >> /etc/apt/sources.list && \
    apt-get -o Acquire::Check-Valid-Until=false update && \
    apt-get install -y dnsutils
# --- FIX END ---

COPY dnsExfiltrator.cs .

# Kompilieren (Deine Flags waren korrekt)
RUN mcs -unsafe -r:System.IO.Compression,System.Web.Extensions -out:DNSExfiltrator.exe dnsExfiltrator.cs