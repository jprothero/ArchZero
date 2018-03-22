import sqlite3
from os import mkdir
from os.path import exists, join
from pathlib import Path
import numpy as np
import model_assembly as ma
import pickle

def create_data_table():
    conn = sqlite3.connect("data.db", detect_types=sqlite3.PARSE_DECLTYPES)
    
    c = conn.cursor()
    c.execute("""DROP TABLE IF EXISTS data""")
    c.execute("""CREATE TABLE data (depth INT, loss FLOAT, board_string TEXT, probas_string, result)""")
    
    conn.commit()
    conn.close()
    
def create_tables():
    create_data_table()
        
def select(statement):
    conn = sqlite3.connect("data.db", detect_types=sqlite3.PARSE_DECLTYPES)
    c = conn.cursor()
    c.execute(statement)
    results = c.fetchall()
    conn.commit()
    conn.close()
    return results

def insert(statement, insert):
    conn = sqlite3.connect("data.db", detect_types=sqlite3.PARSE_DECLTYPES)
    c = conn.cursor()
    #print("Inserting:", insert)
    c.execute(statement, insert)
    results = c.lastrowid
    conn.commit()
    conn.close()
    return results

def update(statement):
    conn = sqlite3.connect("data.db", detect_types=sqlite3.PARSE_DECLTYPES)
    c = conn.cursor()
    #print("Updating:", statement)
    c.execute(statement)
    conn.commit()
    conn.close()

#def create_folders():
    #motifs_path = "motifs"
    #if not exists(motifs_path):
    #    mkdir(motifs_path)