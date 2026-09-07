"""Shared runtime helper, preserved from its historical experiment owner."""
import json

def read_field(path, key):
    """Read one compact generated JSON field without loading huge query tails."""
    marker = ('"' + key + '":').encode()
    tail = b''
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            data = tail + chunk
            at = data.find(marker)
            if at >= 0:
                value = data[at + len(marker):]
                while len(value) <= 64 * 1024 * 1024:
                    try:
                        return json.JSONDecoder().raw_decode(value.decode())[0]
                    except json.JSONDecodeError:
                        chunk = stream.read(1024 * 1024)
                        if not chunk:
                            raise
                        value += chunk
                raise ValueError('diagnostic field exceeds bounded read budget')
            tail = data[-len(marker):]
    raise KeyError(key)
