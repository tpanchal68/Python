drop table if exists users;
create table users (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	firstname VARCHAR NOT NULL,
	lastname VARCHAR NOT NULL,
	email TEXT NOT NULL,
	password TEXT NULL,
	created_on DATETIME DEFAULT CURRENT_TIMESTAMP
);