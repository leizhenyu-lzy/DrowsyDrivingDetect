# 1950083 自动化 刘智宇
import pymysql

if __name__ == "__main__":
    password = input("DataBase Password:")
    db = pymysql.connect(host="localhost", port=3306, user="root", password=password, database="atguigudb")
    cursor = db.cursor()
    cursor.execute("select * from employees;")
    datas = cursor.fetchall()
    for data in datas:
        print(data)
    db.close()
