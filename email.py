# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 18:29:20 2018

@author: tulincakmak
"""

import email, imaplib, os
import uuid

from cerebro_configuration import database


class EmailReader:
    def read(self,from_email_address):

        detach_dir = r'D:\ıvırzıvır'  # directory where to save attachments (default: current)  (C:\Users\tulincakmak\Desktop)
        user = "tulin.cakmak@kontra.net"
        pwd = "tlnckmk10025007"

        # connecting to the gmail imap server
        m = imaplib.IMAP4_SSL("imap.gmail.com")
        m.login(user, pwd)
        m.list()

        m.select("INBOX")
        try:
            items = m.search(None, 'UNSEEN', '(FROM "+from_email_address+")')
            print(items)
        except Exception as ex:
            print(ex)

        items = items[0].split()  # getting the mails id

        if len(items) > 0:
            for emailid in items:
                resp, data = m.fetch(emailid, "(RFC822)")
                email_body = data[0][1]  # getting the mail content
                mail = email.message_from_bytes(email_body)
                mailFrom = str(mail["From"])
                start = mailFrom.find('<')
                end = mailFrom.find('>')
                mailFrom = mailFrom[start + 1:end]

                #if mailFrom ==from_email_address:
                    #continue

                if mail.get_content_maintype() != 'multipart':
                        continue

                for part in mail.walk():
                    if part.get_content_maintype() == 'multipart':
                        continue

                    if part.get('Content-Disposition') is None:
                        continue

                    filename = part.get('Content-Disposition').replace("attachment; filename=", "").replace('"', "")
                    filename = str(uuid.uuid4().int) + "_" + filename
                    att_path = os.path.join(detach_dir, filename)

                    if not os.path.isfile(att_path):
                        fp = open(att_path, 'wb')
                        fp.write(part.get_payload(decode=True))
                        fp.close()
                    file_path=str(detach_dir)+ '\\' +str(filename)
        return file_path,filename


