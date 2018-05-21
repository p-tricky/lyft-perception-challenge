import torch
from torchvision import transforms
from models import LinkNet34
import time
import sys, skvideo.io, json, base64
import numpy as np
from PIL import Image
from io import BytesIO, StringIO

img_transform = transforms.Compose([
    lambda x: x[:544],
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255])
])

# Define encoder function
def encode(array):
    pil_img = Image.fromarray(array)
    buff = BytesIO()
    pil_img.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")

file = sys.argv[-1]

if file == 'demo.py':
  print ("Error loading video")
  quit
    
video = skvideo.io.vread(file)

answer_key = {}

# Frame numbering starts at 1
frame = 1

model = LinkNet34(num_classes=3).cuda()
model.load_state_dict(torch.load('./best.pt'))
model.eval()

start = time.time()
for rgb_frame in video:
    
    input_img = torch.unsqueeze(img_transform(rgb_frame).cuda(), dim=0)
    out = model(input_img)

    out = np.argmax(out.cpu().detach().numpy()[0], axis=0)
    
    # Look for red cars :)
    binary_car_result = np.where(out == 2,1,0).astype('uint8')
    
    # Look for road :)
    binary_road_result = np.where(out == 1,1,0).astype('uint8')

    answer_key[frame] = [encode(binary_car_result), encode(binary_road_result)]
    
    # Increment frame
    frame+=1
    
end = time.time()    
# Print output in proper json format
print (json.dumps(answer_key))
