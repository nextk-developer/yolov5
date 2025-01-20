import os
import torch
import struct
import mlflow
import psycopg2
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from dotenv import dotenv_values

CONFIG = dotenv_values(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.env"))

def convert_model(pt_path):
    wts_path = os.path.splitext(pt_path)[0] + ".wts"
    model = torch.load(pt_path, map_location="cpu")
    model = model['ema' if model.get('ema') else 'model'].float()

    anchor_grid = model.model[-1].anchors * model.model[-1].stride[..., None, None]
    delattr(model.model[-1], 'anchor_grid')  # model.model[-1] is detect layer

    # The parameters are saved in the OrderDict through the "register_buffer" method, and then saved to the weight.
    model.model[-1].register_buffer("anchor_grid", anchor_grid)
    model.model[-1].register_buffer("strides", model.model[-1].stride)

    model.eval()

    with open(wts_path, 'w') as f:
        f.write('{}\n'.format(len(model.state_dict().keys())))
        for k, v in model.state_dict().items():
            vr = v.reshape(-1).cpu().numpy()
            f.write('{} {} '.format(k, len(vr)))
            for vv in vr:
                f.write(' ')
                f.write(struct.pack('>f', float(vv)).hex())
            f.write('\n') 
    return

def encrypt_model(pt_path):
    wts_path = os.path.splitext(pt_path)[0] + ".wts"
    enc_file_path = os.path.splitext(wts_path)[0] + "_enc.wts"
    
    _key = bytes([0xFF, 0x01, 0xEE, 0xD7, 0xC4, 0xB5, 0x02, 0x07, 0x08, 0x00, 0x08, 0x00, 0x00, 0x00, 0xAE, 0xFF])
    _iv = bytes([0x1F, 0x5A, 0x9F, 0x0B, 0x3F, 0xFC, 0xFF, 0xCD, 0xFF, 0x00, 0x45, 0x78, 0x26, 0x74, 0x69, 0xFF])

    with open(wts_path, "r", newline="\r\n") as f:
        buffer = f.read().encode("utf-8")

    aes = AES.new(_key, AES.MODE_CBC, iv=_iv)

    block_size = 16
    padded_buffer = pad(buffer, block_size)
    encrypted_text = aes.encrypt(padded_buffer)

    with open(enc_file_path, "wb") as f:
        f.write(encrypted_text)

    return

def init_mlflow(project_name, run_name):
    os.environ["MLFLOW_TRACKING_URI"] = f"http://{CONFIG['MLOPS_SERVER_IP']}:{CONFIG['MLFLOW_WEB_PORT']}"
    os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"

    os.environ["MLFLOW_EXPERIMENT_NAME"] = project_name
    os.environ["MLFLOW_RUN"] = run_name
    
    os.environ["MLFLOW_TRACKING_USERNAME"] = "admin"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "password"

    os.environ["AWS_ACCESS_KEY_ID"] = CONFIG["MLFLOW_STORAGE_USER"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = CONFIG["MLFLOW_STORAGE_PASSWORD"]
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://{CONFIG['MLOPS_SERVER_IP']}:{CONFIG['MLFLOW_STORAGE_API_PORT']}"
    print("successfully initialized mlflow")
    return


def log_artifact(local_path: str, artifact_path: str):
    mlflow.log_artifact(local_path=local_path, artifact_path=artifact_path)
    return


class Database:
    def __init__(self):
        self.db = psycopg2.connect(
            host=CONFIG["MLOPS_SERVER_IP"],
            port=CONFIG["POSTGRESQL_DB_PORT"],
            dbname=CONFIG["POSTGRESQL_DB_NAME"],
            user=CONFIG["POSTGRESQL_DB_USER"],
            password=CONFIG["POSTGRESQL_DB_PASSWORD"],
        )
        self.cursor = self.db.cursor()
        return

    def __del__(self):
        self.db.close()
        self.cursor.close()
        return

    def execute(self, query):
        self.cursor.execute(query)
        return

    def fetchall(self):
        return self.cursor.fetchall()

    def fetchone(self):
        return self.cursor.fetchone()

    def commit(self):
        self.db.commit()
        return
