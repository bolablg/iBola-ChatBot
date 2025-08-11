import psycopg2
import sqlite3
from sqlite3 import Error
from datetime import datetime
import config

def create_connection():
    """ create a database connection to a SQLite or PostgreSQL database """
    conn = None
    if config.DB_TYPE == "sqlite":
        try:
            conn = sqlite3.connect(config.DB_NAME or "feedback.db")
            return conn
        except Error as e:
            print(e)
    elif config.DB_TYPE == "postgresql":
        try:
            conn = psycopg2.connect(
                dbname=config.DB_NAME,
                user=config.DB_USER,
                password=config.DB_PASSWORD,
                host=config.DB_HOST,
                port=config.DB_PORT
            )
            return conn
        except psycopg2.Error as e:
            print(e)
    return conn

def create_table(conn):
    """ create a table from the create_table_sql statement """
    try:
        if config.DB_TYPE == "sqlite":
            sql_create_feedback_table = """ CREATE TABLE IF NOT EXISTS feedback (
                                            id integer PRIMARY KEY,
                                            session_id text NOT NULL,
                                            question text NOT NULL,
                                            answer text NOT NULL,
                                            rating integer NOT NULL,
                                            timestamp text NOT NULL
                                        ); """
        elif config.DB_TYPE == "postgresql":
            sql_create_feedback_table = """ CREATE TABLE IF NOT EXISTS feedback (
                                            id serial PRIMARY KEY,
                                            session_id text NOT NULL,
                                            question text NOT NULL,
                                            answer text NOT NULL,
                                            rating integer NOT NULL,
                                            timestamp timestamp NOT NULL
                                        ); """
        c = conn.cursor()
        c.execute(sql_create_feedback_table)
        conn.commit()
    except (Error, psycopg2.Error) as e:
        print(e)

def add_feedback(conn, feedback_data):
    """ Log a new feedback entry """
    if config.DB_TYPE == "sqlite":
        sql = ''' INSERT INTO feedback(session_id,question,answer,rating,timestamp)
                  VALUES(?,?,?,?,?)
              '''
    elif config.DB_TYPE == "postgresql":
        sql = ''' INSERT INTO feedback(session_id,question,answer,rating,timestamp)
                  VALUES(%s,%s,%s,%s,%s)
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