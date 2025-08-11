import mariadb
import sqlite3
from sqlite3 import Error
from datetime import datetime
import config

def create_connection():
    """ create a database connection to a SQLite or MariaDB database """
    conn = None
    if config.DB_TYPE == "sqlite":
        try:
            conn = sqlite3.connect(config.DB_NAME)
            return conn
        except Error as e:
            print(e)
    elif config.DB_TYPE == "mariadb":
        try:
            conn = mariadb.connect(url=config.FEEDBACK_DB_CONN_URL)
            return conn
        except mariadb.Error as e:
            print(e)
    return conn

def create_table(conn):
    """ create a table from the create_table_sql statement """
    try:
        if config.DB_TYPE == "sqlite":
            sql_create_feedback_table = """ CREATE TABLE IF NOT EXISTS feedback (
                                            id integer PRIMARY KEY,
                                            session_id text NOT NULL,
                                            user_ip text,
                                            user_questions text,
                                            bot_answers text,
                                            rating integer NOT NULL,
                                            creation_time text NOT NULL
                                        ); """
            sql_create_chat_history_table = """ CREATE TABLE IF NOT EXISTS chat_history (
                                                id integer PRIMARY KEY,
                                                session_id text NOT NULL,
                                                user_message text NOT NULL,
                                                bot_message text NOT NULL,
                                                timestamp text NOT NULL
                                            ); """
        elif config.DB_TYPE == "mariadb":
            sql_create_feedback_table = """ CREATE TABLE IF NOT EXISTS feedback (
                                            id int NOT NULL AUTO_INCREMENT,
                                            session_id varchar(255) NOT NULL,
                                            user_ip varchar(255),
                                            user_questions text,
                                            bot_answers text,
                                            rating int NOT NULL,
                                            creation_time timestamp NOT NULL,
                                            PRIMARY KEY (id)
                                        ); """
            sql_create_chat_history_table = """ CREATE TABLE IF NOT EXISTS chat_history (
                                                id int NOT NULL AUTO_INCREMENT,
                                                session_id varchar(255) NOT NULL,
                                                user_message text NOT NULL,
                                                bot_message text NOT NULL,
                                                timestamp timestamp NOT NULL,
                                                PRIMARY KEY (id)
                                            ); """
        c = conn.cursor()
        c.execute(sql_create_feedback_table)
        c.execute(sql_create_chat_history_table)
        conn.commit()
    except (Error, mariadb.Error) as e:
        print(e)

def add_chat_history(conn, chat_history_data):
    """ Log a new chat history entry """
    if config.DB_TYPE == "sqlite":
        sql = ''' INSERT INTO chat_history(session_id,user_message,bot_message,timestamp)
                  VALUES(?,?,?,?)
              '''
    elif config.DB_TYPE == "mariadb":
        sql = ''' INSERT INTO chat_history(session_id,user_message,bot_message,timestamp)
                  VALUES(%s,%s,%s,%s)
              '''
    cur = conn.cursor()
    cur.execute(sql, chat_history_data)
    conn.commit()
    return cur.lastrowid

def get_chat_history(conn, session_id):
    """ Query chat history by session_id """
    cur = conn.cursor()
    if config.DB_TYPE == "sqlite":
        cur.execute("SELECT user_message, bot_message FROM chat_history WHERE session_id = ? ORDER BY timestamp ASC", (session_id,))
    elif config.DB_TYPE == "mariadb":
        cur.execute("SELECT user_message, bot_message FROM chat_history WHERE session_id = %s ORDER BY timestamp ASC", (session_id,))
    rows = cur.fetchall()
    return rows

def add_feedback(conn, feedback_data):
    """ Log a new feedback entry """
    if config.DB_TYPE == "sqlite":
        sql = ''' INSERT INTO feedback(session_id,user_ip,user_questions,bot_answers,rating,creation_time)
                  VALUES(?,?,?,?,?,?)
              '''
    elif config.DB_TYPE == "mariadb":
        sql = ''' INSERT INTO feedback(session_id,user_ip,user_questions,bot_answers,rating,creation_time)
                  VALUES(%s,%s,%s,%s,%s,%s)
              '''
    cur = conn.cursor()
    cur.execute(sql, feedback_data)
    conn.commit()
    return cur.lastrowid

def initialize_database():
    conn = create_connection()
    if conn is not None:
        create_table(conn)
        conn.close()
    else:
        print("Error! cannot create the database connection.")

if __name__ == '__main__':
    initialize_database()
