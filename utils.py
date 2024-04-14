import hashlib

def create_hash(string):
    return hashlib.md5(string.encode("utf-8")).hexdigest()