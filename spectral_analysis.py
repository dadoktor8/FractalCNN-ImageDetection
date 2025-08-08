import numpy as np 
import cv2 
import torch 
import torch.nn.functional as F 
from scipy import ndimage
import matplotlib.pyplot as plt 

class SpectralAnalyzer:
    def __init__(self, kernel_size = 7):
        self.kernel_size = kernel_size 
        
    def extract_noise_residual(self,image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
        blurred = cv2.medianBlur(image.astype(np.uint8), self.kernel_size)
        noise_residual = image.astype(np.float32) - blurred.astype(np.float32)
        
        return noise_residual
    
    def compute_spectrum(self,image):
        fft = np.fft.fft2(image)
        fft_shifted = np.fft.fftshift(fft)
        
        magnitude_spectrum = np.abs(fft_shifted)
        magnitude_spectrum = magnitude_spectrum / np.max(magnitude_spectrum)
        
        return magnitude_spectrum
    
    def split_into_fractal_branches(self,spectrum): 
        h,w = spectrum.shape
        mid_h,mid_w = h//2,w//2 #Why do we split into 4 branches?!
        
        branch_00 = spectrum[:mid_h, :mid_w]
        branch_01 = spectrum[:mid_h, mid_w:]
        branch_10 = spectrum[mid_h:, :mid_w]
        branch_11 = spectrum[mid_h:, mid_w:]
        
        return [branch_00, branch_01, branch_10, branch_11]
    
    def compute_self_similarity(self,branches):
        similarity = branches[0]
        for branch in branches[1:]:
            if branch.shape != similarity.shape: 
                branch = cv2.resize(branch,similarity.shape[::-1])
            similarity = similarity * branch
        
        return similarity
    
    def visualize_spectrum(self,image, title="Spectrum"):
        noise_residual = self.extract_noise_residual(image)
        spectrum = self.compute_spectrum(noise_residual)
        
        plt.figure(figsize=(12,4))
        
        plt.subplot(1,3,1)
        plt.imshow(image, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1,3,2)
        plt.imshow(noise_residual, cmap='gray')
        plt.title('Noise Residual')
        plt.axis('off')
        
        plt.subplot(1,3,3)
        plt.imshow(np.log(spectrum + 1e-8), cmap='hot')
        plt.title(f'{title} - Log Spectrum')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        
def test_spectral_analysis():
    test_image = np.random.rand(224,224) * 255
    analyzer = SpectralAnalyzer()
    
    noise_residual = analyzer.extract_noise_residual(test_image)
    print(f"Noise residual Shape: {noise_residual.shape}")
    
    spectrum = analyzer.compute_spectrum(noise_residual)
    print(f"Spectrum Shape: {spectrum.shape}")
    
    branches = analyzer.split_into_fractal_branches(spectrum)
    print(f"Number of branches: {len(branches)}")
    print(f"Branch Shapes: {[b.shape for b in branches]}")
    
    similarity = analyzer.compute_self_similarity(branches)
    print(f"Self-similarity shape: {similarity.shape}")
    
    analyzer.visualize_spectrum(test_image,"TEST IMAGE")
    
    return analyzer

if __name__ == "__main__":
    analyzer = test_spectral_analysis()