import os
import torch
import struct
from pathlib import Path
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad

key = bytes([0xff, 0x01, 0xee, 0xd7, 0xc4, 0xb5, 0x02, 0x07, 0x08, 0x00, 0x08, 0x00, 0x00, 0x00, 0xae, 0xff])
iv = bytes([0x1f, 0x5a, 0x9f, 0x0b, 0x3f, 0xfc, 0xff, 0xcd, 0xff, 0x00, 0x45, 0x78, 0x26, 0x74, 0x69, 0xff])

class ModelExporter:
    def __init__(self, pt_path) -> None:
        self.pt_path = pt_path
        self.wts_path = os.path.splitext(pt_path)[0] + ".wts"
        self.wts_enc_path = os.path.splitext(pt_path)[0] + "_enc.wts"
        return

    def convert(self):
        model = torch.load(self.pt_path, map_location="cpu")
        model = model['ema' if model.get('ema') else 'model'].float()

        anchor_grid = model.model[-1].anchors * model.model[-1].stride[..., None, None]
        delattr(model.model[-1], 'anchor_grid')  # model.model[-1] is detect layer

        # The parameters are saved in the OrderDict through the "register_buffer" method, and then saved to the weight.
        model.model[-1].register_buffer("anchor_grid", anchor_grid)
        model.model[-1].register_buffer("strides", model.model[-1].stride)

        model.eval()

        with open(Path(self.wts_path), 'w') as f:
            f.write('{}\n'.format(len(model.state_dict().keys())))
            for k, v in model.state_dict().items():
                vr = v.reshape(-1).cpu().numpy()
                f.write('{} {} '.format(k, len(vr)))
                for vv in vr:
                    f.write(' ')
                    f.write(struct.pack('>f', float(vv)).hex())
                f.write('\n') 
        return
    
    def encrypt(self):
        with open(self.wts_path, 'r', newline="\r\n") as f:
            buffer = f.read().encode("utf-8")

        aes = AES.new(key, AES.MODE_CBC, iv=iv)

        block_size = 16
        padded_buffer = pad(buffer, block_size)
        encrypted_text = aes.encrypt(padded_buffer)

        with open(self.wts_enc_path, 'wb') as f:
            f.write(encrypted_text)        
        return