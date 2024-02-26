import yaml

MLFLOW_TRACKING_URI = "http://192.168.0.70:5001"

#for log the run model_info file
class RunInfo:
    def __init__(self, exp_name, run_name, run_id, run_date, yolo_model_type) -> None:
        self.run_info_dict = {
            "exp_name": exp_name,
            "run_name": run_name,
            "run_id": run_id,
            "yolo_model_type": yolo_model_type,
            "run_date": run_date
        }
        return

    def save_yaml(self, save_dir):
        with open(save_dir, "w", encoding="utf8") as f:
            yaml.dump(self.run_info_dict, f, allow_unicode=True)
        return
    
import os
from dotenv import load_dotenv
import psycopg2

load_dotenv()

class Database:
    def __init__(self, host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT"),\
        database=os.getenv("DB_NAME"),\
        user=os.getenv("DB_ID"), password=os.getenv("DB_PW")) -> None:  
        if database is None or user is None or password is None:
            if os.environ.get("DB_NAME") is not None:
                host = os.environ.get("DB_HOST")
                port = os.environ.get("DB_PORT")
                database = os.environ.get("DB_NAME")
                user = os.environ.get("DB_ID")
                password = os.environ.get("DB_PW")
            else:
                host="192.168.0.70"
                port="5434"
                database="mlops_database"
                user="nextk"
                password="nextk"
            print(f'init database 2:{host} {port} {database} {user} {password}')     
        print(f'init database:{host} {port} {database} {user} {password}')     
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
    
    def connect(self):
        self.conn = psycopg2.connect(
            host=self.host,
            port=self.port,
            database=self.database,
            user=self.user,
            password=self.password )

    def close(self):
        self.conn.close()
        
    def update(self, sql, vars):
        try:
            cur = self.conn.cursor()
            cur.execute(sql, vars)
            self.conn.commit()
            cur.close()
            return id
        except Exception as e:
            print(e)
            raise(e)
        
        
    def commit(self, sql, vars):
        try:
            cur = self.conn.cursor()
            cur.execute(sql, vars)
            id = tuple(cur.fetchone())[0]
            self.conn.commit()
            cur.close()
            return id
        except Exception as e:
            print(e)
            # raise(e)
            return 0

class AppSql:
    training_status_log = "INSERT INTO training_status (modelName, bucketName, totalEpoch, trainStartTime, created , modified) VALUES (%s, %s, %s, %s, %s, %s) RETURNING id;"
    training_status_update = "UPDATE training_status SET currentEpoch=%s, modified=%s WHERE id = %s;"    
    training_status_update_end = "UPDATE training_status SET trainEndTime=%s, modified=%s WHERE id = %s;"
