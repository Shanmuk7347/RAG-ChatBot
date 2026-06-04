import sqlite3
import json


def get_con_cur():
    con = sqlite3.connect("database.db")
    con.execute("PRAGMA foreign_keys = ON")
    cur = con.cursor()

    return con, cur

""" ======================= DATABASE SETUP ========================= """

def create_tables():
    con, cur = get_con_cur()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS chats(
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL)"""
    )

    cur.execute("""
    CREATE TABLE IF NOT EXISTS messages(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    sources TEXT,
    FOREIGN KEY(chat_id) REFERENCES chats(id)
    ON DELETE CASCADE)"""
    )

    cur.execute("""
    CREATE TABLE IF NOT EXISTS documents(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    FOREIGN KEY(chat_id) REFERENCES chats(id)
    ON DELETE CASCADE)"""
    )
    con.commit()
    con.close()

""" ======================= CHATS ========================="""

def save_chat(chat_id:str, name:str):
    con, cur = get_con_cur()
    cur.execute("INSERT INTO chats(id, name) VALUES (?, ?)", (chat_id, name))
    con.commit()
    con.close()

def load_chats():
    con, cur = get_con_cur()
    cur.execute("SELECT id, name FROM chats")
    rows = cur.fetchall()
    con.close()

    return [{
        "id": row[0],
        "name": row[1]} 
        for row in rows]

def delete_chat(chat_id:str):
    con, cur = get_con_cur()
    cur.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
    con.commit()
    con.close()

""" ======================= MESSAGES ========================="""

def save_message(chat_id:str, role:str, content:str, sources):
    con, cur = get_con_cur()
    cur.execute("INSERT INTO messages(chat_id, role, content, sources) VALUES (?, ?, ?, ?)", (chat_id, role, content, sources))
    con.commit()
    con.close()

def load_messages(chat_id:str):
    con, cur = get_con_cur()
    cur.execute("SELECT role, content, sources FROM messages WHERE chat_id = ? ORDER BY id ", (chat_id,))  
    rows = cur.fetchall()
    con.close()  
    return [
        {"role": row[0],
         "content": row[1],
         "sources": json.loads(row[2]) if row[2] else None}
        for row in rows
    ]

""" ======================= DOCUMENTS ========================="""

def save_documents(chat_id:str, filename:str):
    con, cur = get_con_cur()
    cur.execute("INSERT INTO documents(chat_id, filename) VALUES (?, ?)", (chat_id, filename))
    con.commit()
    con.close()

def load_documents(chat_id:str):
    con, cur = get_con_cur()
    cur.execute("SELECT filename FROM documents WHERE chat_id = ?", (chat_id,))
    rows = cur.fetchall()
    con.close()
    return [row[0] for row in rows]


