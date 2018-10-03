# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 10:57:28 2017

@author: tulincakmak
"""
string='05437835182'
str=string.encode()

from cryptography.fernet import Fernet
key = Fernet.generate_key() #this is your "password"
cipher_suite = Fernet(key)
encoded_text = cipher_suite.encrypt(str)
decoded_text = cipher_suite.decrypt(encoded_text)

print(encoded_text)
print(decoded_text)
print(key)

zBc5TLwJms8vzJYo8Dp2YGgnpusYnUAzIdM-kTyLTps=

from Crypto.Cipher import AES
obj = AES.new('This is a key123', AES.MODE_CFB, 'This is an IV456')
message = "The answer is no"
ciphertext = obj.encrypt(message)
ciphertext='\xd6\x83\x8dd!VT\x92\xaa`A\x05\xe0\x9b\x8b\xf1'
obj2 = AES.new('This is a key123', AES.MODE_CFB, 'This is an IV456')
obj2.decrypt(ciphertext)


from Crypto.Cipher import AES
# Encryption
encryption_suite = AES.new('This is a key123', AES.MODE_CBC, 'This is an IV456')
cipher_text = encryption_suite.encrypt("5437835182")

# Decryption
decryption_suite = AES.new('This is a key123', AES.MODE_CBC, 'This is an IV456')
plain_text = decryption_suite.decrypt(cipher_text)


import base64
import hashlib
from Crypto import Random
from Crypto.Cipher import AES

class AESCipher(object):

    def __init__(self, key): 
        self.bs = 32
        self.key = hashlib.sha256(key.encode()).digest()

    def encrypt(self, raw):
        raw = self._pad(raw)
        iv = Random.new().read(AES.block_size)
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return base64.b64encode(iv + cipher.encrypt(raw))

    def decrypt(self, enc):
        enc = base64.b64decode(enc)
        iv = enc[:AES.block_size]
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return self._unpad(cipher.decrypt(enc[AES.block_size:])).decode('utf-8')

    def _pad(self, s):
        return s + (self.bs - len(s) % self.bs) * chr(self.bs - len(s) % self.bs)

    @staticmethod
    def _unpad(s):
        return s[:-ord(s[len(s)-1:])]
    
msg=input('Hello World')
pwd=input('Password:01234567899876543210')

print(AESCipher.encrypt(msg))


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
