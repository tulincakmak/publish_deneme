# -*- coding: utf-8 -*-
"""
Created on Thu May 31 17:34:07 2018

@author: tulincakmak
"""

from queue import Queue

import pandas
from sqlalchemy import create_engine, MetaData, select, and_ , Table
from sqlalchemy.orm import sessionmaker


class DataResultType:
    raw = "list"
    frame = "frame"
    single = "single"
    void = "void"

    mappers = {
        raw: lambda cursor: [dict(item) for item in cursor.fetchall()],
        void: lambda cursor: None,
        single: lambda cursor: list(cursor.fetchone())[0],
        frame: lambda cursor: pandas.DataFrame(cursor.fetchall(), columns=cursor.keys())
    }


class DatabaseManager:

    def __init__(self, database_name, user, pass_, host):
        """
        Client DB Manager for sql. Manages all sql transactions
        :param database_name:
        """

        self.__db_api_type_codes = {
            1: "string",
            2: "int",
            3: "int",
            4: "datetime",
            5: "float"
        }
        self.__db = database_name
        self.__connection_string = "mssql+pymssql://" + user + ":" + pass_ + "@" + host + "/" + database_name
        self.__engine = create_engine(self.__connection_string, isolation_level="READ COMMITTED", echo=False,
                                      pool_size=50, max_overflow=0)

        """Create main thread connection"""
        self.__connection = self.__engine.connect()

        """Create a list to keep track child thread connections"""
        self.__open_connections = list()
        self.__open_connections.append(self.__connection)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """
            Close all open connections
        """
        for connection in self.__open_connections:
            if connection:
                try:
                    connection.close()
                except:
                    pass

    def execute_literal(self, statement, result_type="frame"):
        """
            Execute statement
        :param statement: str: literal statement
        :param result_type: DataResultType:
        """

        return self.cursor_to_result(self.__connection.execute(statement), result_type)

    def cursor_to_result(self, cursor, result_type="frame"):
        return DataResultType.mappers[result_type](cursor)

    def bulk_insert_data_frame(self, data_frame, table_name, table=None):
        """
        Inserts a data frame or list of dicts to targeted table
        
        :rtype: None
        """
        table = Table(table_name, MetaData(), autoload=True, autoload_with=self.__engine) if table is None else table
        return self.execute(table.insert(), data_frame)
    
    def get_session(self):
        maker = sessionmaker(bind=self.__engine)
        session = maker(autocommit=True)
        self.__open_connections.append(session)
        return session    
        
    def execute(self, command, data):
        values = data.to_dict(orient="records") if isinstance(data, pandas.DataFrame) else data
        session = self.get_session()
        session.begin(subtransactions=True)
        session.execute(command, values)
        session.commit()

if __name__ == "__main__":
    with DatabaseManager("TempData","tulinC", "tlnckmk", "78.40.231.196") as db:
        data = db.execute_literal("SELECT * from ##dataroomSpec2")
        
      