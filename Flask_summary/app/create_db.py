import sqlite3

connection = sqlite3.connect("development.db")
cur = connection.cursor()
with open('schema.sql') as fp:
    cur.executescript(fp.read())  # or con.executescript
print("db created successfully.")
