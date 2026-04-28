---
chapter: ch-08
course: boson-agent
phase: read
kind: glossary
title: "Networking Glossary — Protocols and Terms"
---

# Networking Glossary

Reference sheet for ch-08. Pair with [[tcp-websocket-background]] for depth on TCP/WebSocket specifically. This page is terminology-only — look things up here when a term appears and you need a 1-paragraph grounding.

---

## 1. The protocol stack (layer-by-layer)

```
Layer 7   Application   HTTP, WebSocket, SSH, FTP, SMTP, DNS, gRPC, MQTT
Layer 4   Transport     TCP, UDP, QUIC
Layer 3   Network       IP (v4, v6), ICMP
Layer 2   Link          Ethernet, WiFi (802.11)
Layer 1   Physical      cables, radio waves
```

(This is the OSI model compressed — the full OSI has 7 layers but most people talk in the 4-layer TCP/IP model which collapses presentation + session + application into one.)

---

## 2. Transport layer (Layer 4)

### TCP (Transmission Control Protocol)
Reliable, ordered, connection-oriented byte stream between two endpoints. Guarantees delivery, order, deduplication, and flow control at the cost of latency (3-way handshake, retransmits, ACKs). Most application protocols you interact with (HTTP, WebSocket, SSH, FTP) run on TCP. **boson-agent's Gateway uses TCP (via WebSocket).**

### UDP (User Datagram Protocol)
Connectionless, fire-and-forget datagrams. No delivery guarantee, no order, no dedup. Low latency because no handshake or retransmits. Used where "fast + some loss tolerable" matters: live video, games, DNS queries, VoIP, real-time telemetry.

### QUIC
Newer transport (2021) built on top of UDP. Provides TCP-like reliability + built-in TLS 1.3 encryption + multiplexed streams without head-of-line blocking. **HTTP/3 runs on QUIC.** Matters for modern web performance, not directly relevant to boson-agent yet.

---

## 3. Network layer (Layer 3)

### IP (Internet Protocol)
Addresses and routes packets. **IPv4** (32-bit addresses, e.g. `192.168.1.1`) is running out; **IPv6** (128-bit, e.g. `2001:db8::1`) is the long-term replacement. IP itself is stateless and unreliable — all reliability comes from TCP or application code above it.

### ICMP (Internet Control Message Protocol)
Used for network diagnostics. `ping` sends an ICMP Echo Request; `traceroute` uses ICMP Time Exceeded. Not a transport protocol — apps don't send data through ICMP.

---

## 4. Application layer (Layer 7) — protocols you'll actually write code against

### HTTP (Hypertext Transfer Protocol)
Request-response protocol over TCP. Client asks (`GET /path`), server replies (`200 OK` + body). Each request is independent — **stateless** at the protocol level; "sessions" and "login state" are faked via cookies or tokens. Versions: HTTP/1.1 (text-based, one request per connection by default), HTTP/2 (binary, multiplexed streams over one TCP connection), HTTP/3 (runs on QUIC).

### HTTPS
HTTP running inside a TLS-encrypted tunnel. Same protocol, encrypted transport. Adds authentication via certificates. Adds a small latency cost for the TLS handshake (reduced to ~1 RTT in TLS 1.3).

### WebSocket
Long-lived, bidirectional message-framing protocol over TCP. Starts as HTTP, upgrades via `Upgrade: websocket` header. See [[tcp-websocket-background]] for details. **Used by boson-agent's Gateway for browser ↔ server streaming.**

### SSH (Secure Shell)
Encrypted, authenticated shell / file transfer protocol over TCP (port 22). Used for remote login, `scp` file transfer, git-over-ssh, port forwarding, tunneling. Separate concept from HTTPS — SSH has its own auth system (keys or passwords) independent of TLS.

### FTP / SFTP
Old file transfer protocols. FTP (port 21) is plaintext and uses two connections (control + data). SFTP is file transfer over an SSH connection — preferred over FTP today. Mostly legacy; object storage (S3) has replaced them for new systems.

### SMTP / IMAP / POP3
Email protocols. SMTP sends mail between servers; IMAP/POP3 fetch mail for clients. Plaintext by default, encrypted via STARTTLS or implicit TLS on alternate ports.

### DNS (Domain Name System)
Translates domain names ("example.com") to IP addresses ("93.184.216.34"). Usually runs on UDP port 53 (queries small enough to fit in one packet). Falls back to TCP for large responses. Hierarchical (root → TLD → authoritative). Caching at every level. When you type a URL, DNS resolution happens first.

### gRPC
RPC framework from Google built on HTTP/2. Uses Protocol Buffers for message encoding. Strongly typed, streaming support, better for service-to-service communication than JSON-over-HTTP. Common in microservice architectures.

### MQTT
Pub/sub message protocol designed for IoT / constrained devices. Tiny header, works over TCP. Common in sensor networks, home automation, industrial telemetry.

### Custom protocols
Nothing stops you from inventing your own application protocol over TCP or UDP. `socket.bind()` on any port, define your own framing, done. boson-agent's wire protocol (the JSON frame schema in `protocol.py`) is effectively a custom protocol layered inside WebSocket frames.

---

## 5. Addressing and identifiers

### IP address
Layer-3 identifier for a machine (or network interface). IPv4 dotted-quad (`10.0.0.1`), IPv6 colon-hex (`fe80::1`). **Private ranges** (`10.0.0.0/8`, `192.168.0.0/16`, `172.16.0.0/12`) are only routable inside LANs; public internet uses the rest.

### Port
Layer-4 identifier for a specific listener on a machine. 16-bit (0-65535). Well-known ports: 22 (SSH), 25 (SMTP), 53 (DNS), 80 (HTTP), 443 (HTTPS), 3306 (MySQL), 5432 (PostgreSQL), 6379 (Redis), 8080 (common HTTP dev), 8000/8001 (common Python dev).

### Socket
Programming abstraction for one endpoint of a connection. A socket is identified by `(IP, port)`. A connected TCP socket pairs two sockets: `(local_ip, local_port, remote_ip, remote_port)`. Python's `socket` module wraps the OS `socket()` syscall.

### URL / URI
Uniform Resource Locator. Example: `wss://api.example.com:8080/ws?token=xyz`.
- `wss` — scheme (WebSocket Secure)
- `api.example.com` — host (DNS-resolved to IP)
- `8080` — port
- `/ws` — path
- `?token=xyz` — query string

### URI vs URL
URI is the generic term (Uniform Resource Identifier). URL is a subset of URI that includes location. In practice, used interchangeably. URN is the rare other subset (`urn:isbn:0451450523`).

---

## 6. Infrastructure terms

### NAT (Network Address Translation)
Router remaps private IPs to a single public IP. Why your home devices all appear to come from one public IP even though each has its own LAN IP (`192.168.1.x`). Breaks server-initiated connections from outside — incoming connections need port forwarding or a public-facing proxy.

### Firewall
Packet filter that allows/blocks traffic by port, IP, or protocol. Can be host-based (iptables, Windows firewall) or network-based (hardware box between LAN and internet). "Opening port 443" = firewall rule letting packets to port 443 through.

### Proxy
Server that forwards client requests to another server. **Forward proxy** — sits between client and internet (corporate filtering, anonymization). **Reverse proxy** — sits between internet and your servers (nginx, Caddy, HAProxy). Reverse proxies are how you terminate TLS, load-balance, and add access control for backend services.

### Load balancer
Distributes incoming connections across multiple backend servers. L4 (TCP-level, by `(src_ip, src_port)` hash) or L7 (HTTP-aware, can route by URL path or header). Typically also a reverse proxy.

### CDN (Content Delivery Network)
Geographically distributed reverse proxies that cache static content near users. Cloudflare, Akamai, Fastly. Reduces latency and origin-server load.

### TLS / SSL
Transport Layer Security (modern) / Secure Sockets Layer (old name, deprecated protocol versions). Encrypts and authenticates data between two endpoints. TLS 1.3 is current. Provides:
- **Encryption** — third parties can't read traffic
- **Integrity** — tampering detected
- **Authentication** — server (and optionally client) proves identity via certificates

HTTPS = HTTP over TLS. WSS = WebSocket over TLS.

---

## 7. Quick reference: "what port is X"

| Service | Port | Transport |
|---|---|---|
| DNS | 53 | UDP (TCP fallback) |
| SSH | 22 | TCP |
| HTTP | 80 | TCP |
| HTTPS | 443 | TCP |
| FTP control | 21 | TCP |
| SMTP | 25 / 587 | TCP |
| IMAP / IMAPS | 143 / 993 | TCP |
| POP3 / POP3S | 110 / 995 | TCP |
| MySQL | 3306 | TCP |
| PostgreSQL | 5432 | TCP |
| Redis | 6379 | TCP |
| MongoDB | 27017 | TCP |

---

## 8. How this connects to ch-08

When reading `packages/gateway/gateway/server/websocket.py` and related files, you'll see:
- `async def start_server(host, port)` — TCP listen socket on `(host, port)`
- `async for message in websocket` — reads WebSocket frames over the established TCP connection
- `websocket.send(...)` — writes WebSocket frames
- Session binding via `session_id` — boson-agent's app-layer identity, above WebSocket, above TCP

Nothing in boson-agent reaches below layer 4. All the TCP and IP and Ethernet happens in the OS kernel; Python just sees "a stream arrives, I write a stream back". But knowing the layers lets you debug: "connection refused" = kernel said no port is listening, "connection reset" = remote TCP sent RST, "connection closed" = orderly FIN exchange.

---

**Cross-reference:** [[tcp-websocket-background]] for how TCP handshake and WebSocket upgrade work in detail.
