import torch 
import torch.nn.functional as F
import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt

from Fractal_CNN import FractalCNN

def test_image(model_path, image_path):
    print(f"Loading Model: {model_path}")
    
    model = FractalCNN(num_fractal_levels=3, hidden_channels=32, num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    model.eval()
    
    print(f"Image Testing: {image_path}")
    
    img = Image.open(image_path)
    
    if img.mode != 'L':
        img = img.convert('L')
        
    img = img.resize((256,256))
    img_a = np.array(img)/ 255.0
    
    
    img_tensor = torch.FloatTensor(img_a)
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.unsqueeze(0)
    
    
    with torch.no_grad():
        output = model(img_tensor)
        prob = F.softmax(output,dim=1)
        #Testing with softmax for now
        
        pred_class = torch.argmax(prob,dim=1).item()
        
        
    if pred_class == 0:
        res = "Real"
    else:
        res = "Fake"
            
    return res 


if __name__ == "__main__":
    MODEL_FILE = r"D:\Projects\Generalizable AI-Generated Image Detection Based on Fractal Self-Similarity in the Spectrum\FractalCNN-ImageDetection\fractal_cnn_finetuned_0.6722.pth"
    
    TEST_IMAGE = r"D:\Projects\Generalizable AI-Generated Image Detection Based on Fractal Self-Similarity in the Spectrum\FractalCNN-ImageDetection\test-image4.png"
    
    try:
        result = test_image(MODEL_FILE,TEST_IMAGE)
        print(f"This image appears to be: {result}")
        
    except FileNotFoundError as e:
        print(f'File not found {e}')
        