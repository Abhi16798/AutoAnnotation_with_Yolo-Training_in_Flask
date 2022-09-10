import pymysql
from datetime import timezone, datetime

def get_db_cursor():
	db = pymysql.connect(host='localhost', user='root', password='', db='videolabel')
	cur = db.cursor()
	return db, cur

def get_one_record(sql_command, params):
	db, cur = get_db_cursor()
	num = cur.execute(sql_command, params)
	record = cur.fetchone()
	cur.close()
	db.close()
	return num, record

def get_records(sql_command, params):
	db, cur = get_db_cursor()
	num = cur.execute(sql_command, params)
	rv = cur.fetchall()
	cur.close()
	db.close()
	return rv

def get_record_num(sql_command, params):
	db, cur = get_db_cursor()
	num = cur.execute(sql_command, params)
	cur.close()
	db.close()
	return num

def get_one_data_where(sql_command, params):
	db, cur = get_db_cursor()
	num = cur.execute(sql_command, params)
	row_headers = [x[0] for x in cur.description]  # this will extract row headers
	record = cur.fetchone()
	cur.close()
	db.close()
	if record is None :
		return 0, {}
	return num, dict(zip(row_headers, record))

def get_one_data(sql_command, id):
	db, cur = get_db_cursor()
	num = cur.execute(sql_command, id)
	row_headers = [x[0] for x in cur.description]  # this will extract row headers
	record = cur.fetchone()
	json_data = []
	if record is not None :
		json_data.append(dict(zip(row_headers, record)))
	cur.close()
	db.close()
	return num, json_data

def update_record(sql_command, params):
	db, cur = get_db_cursor()
	cur.execute(sql_command, params)
	db.commit()
	cur.close()
	db.close()

def insert_record(sql_command, params):
	db, cur = get_db_cursor()
	cur.execute(sql_command, params)
	id = cur.lastrowid
	db.commit()
	cur.close()
	db.close()
	return id

def get_full_data(sql_command, row_headers=None):
	db, cur = get_db_cursor()
	num = cur.execute(sql_command)
	if row_headers is None: row_headers = [x[0] for x in cur.description]  # this will extract row headers
	rv = cur.fetchall()
	json_data = []
	for row in rv:
		result = []
		for item in row:
			if type(item) is datetime: item = item.astimezone(timezone.utc)
			elif item == '0000-00-00 00:00:00': item = ''
			result.append(item)
		json_data.append(dict(zip(row_headers, result)))
	cur.close()
	db.close()
	return json_data
