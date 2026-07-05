import socket
import ipaddress
from urllib.parse import urlparse
from typing import List, Optional

def is_safe_url(url: str, allowed_domains: Optional[List[str]] = None) -> bool:
    """
    Checks if a URL is safe from SSRF.
    Enforces scheme constraints, DNS resolution, and blocks private/loopback/link-local ranges.
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        
        hostname = parsed.hostname
        if not hostname:
            return False
        
        # Verify domain if allowlist is provided
        if allowed_domains and hostname not in allowed_domains:
            return False
        
        # Resolve hostname to all associated IP addresses (IPv4 & IPv6)
        addr_info = socket.getaddrinfo(hostname, None)
        for family, _, _, _, sockaddr in addr_info:
            ip_str = sockaddr[0]
            ip = ipaddress.ip_address(ip_str)
            # Block private, loopback, link-local, multicast, unspecified ranges
            if (ip.is_private or 
                ip.is_loopback or 
                ip.is_link_local or 
                ip.is_multicast or 
                ip.is_unspecified):
                return False
                
        return True
    except Exception:
        return False
