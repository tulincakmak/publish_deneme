# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 18:30:25 2018

@author: tulincakmak
"""

    def readFile(filename, path):
        import pandas as pd
        start=filename.find('.')
        extension=filename[start+1:len(filename)]
        if extension=='xlsx':
            try:
                data=pd.read_excel(path)
                print(data)
            except Exception as ex:
                print(ex)
        if extension=='csv':
            data = pd.read_csv(path)
            print(data)

        return data


    def run(self):
        while True:
            tobechecked = self._db_layer.get_customer_email()
            print(tobechecked)
            for item in tobechecked:
                project_element_id,result=self._db_layer.get_email_to_be_checked(item["email"])
                if result==1:
                    file_paths,filename = self._mail_listener.read(item["email"])
                    data_2=EmailThread.readFile(filename, file_paths)
                    print(data_2)