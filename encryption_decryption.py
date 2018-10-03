# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 12:14:04 2017

@author: tulincakmak
"""



key='1234'
st='Tulin Cakmak'

import base64
def encode(key, clear):
    enc = []
    for i in range(len(clear)):
        key_c = key[i % len(key)]
        enc_c = chr((ord(clear[i]) + ord(key_c)) % 256)
        enc.append(enc_c)
    return base64.urlsafe_b64encode("".join(enc).encode()).decode()

def decode(key, enc):
    dec = []
    enc = base64.urlsafe_b64decode(enc).decode()
    for i in range(len(enc)):
        key_c = key[i % len(key)]
        dec_c = chr((256 + ord(enc[i]) - ord(key_c)) % 256)
        dec.append(dec_c)
    return "".join(dec)

print(encode(key,st))
print(decode(key,'woXCp8Kfwp3Cn1J2wpXCnMKfwpTCnw=='))