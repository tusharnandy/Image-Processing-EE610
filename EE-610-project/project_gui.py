# -*- coding: utf-8 -*-
"""Project_GUI.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1XRg1CRyB5q2XY0A6kXQVCJGUDVmRovNS
"""

absolute = r"C:\Users\Tushar\Desktop\EE-610-project\weights"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
import tkinter as tk
from PIL import ImageTk, Image
import PIL.Image as pil_image
import tkinter.font as font
import cv2
import warnings
from functools import partial
warnings.filterwarnings('ignore')
from tkinter import filedialog
import torch
from models import RDN

"""
Global Variables in the code:
image: current image or the variation which is displayed
image_array: stack of images and variations created after browsing
action_array: stack of strings having names of operations till now
"""

def select_img():     
    """
    Function to get an image from PC
    """
    global image, image_array, action_array
    path = filedialog.askopenfilename(title='open', filetypes = (("PNG Files", "*.png*"), ("JPG Files", "*.jpg*")))
    #gets the path of the file from pc
    if path != ():
        image = cv2.imread(path) #image is a numpy ndarray now
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #convert to RGB
        PIL_image = Image.fromarray(image.astype('uint8'), 'RGB') #converted to PIL
        Tk_image = ImageTk.PhotoImage(PIL_image) #converted to ImageTk for compatibility with GUI framework
        panel.configure(image = Tk_image, bd = 10) #place image in the panel
        panel.image = Tk_image #place image in the panel
        action_array = ["Browse"] #save info of last action in stack for display
        last["text"] = "Browse" #will be displayed
        image_array = [] #stack for storing images and variations
        image_array.append(image) #store image in stack

def undo(): 
    global image ,image_array, action_array
    if len(image_array) > 1: #execute only if some action has been done
        image_array.pop() #returns (pops) last element of the stack
        image = image_array[-1] #new last element after popping
        action_array.pop() #pops last element of action stack
        last['text'] = action_array[-1] #new last action
        PIL_image = Image.fromarray(image.astype('uint8'), 'RGB') #converted to PIL
        Tk_image = ImageTk.PhotoImage(PIL_image) #converted to ImageTk
        panel.configure(image = Tk_image)  #configure previous (now current) image
        panel.image = Tk_image        #image placed on the panel 

def undo_all():
    global image ,image_array, action_array
    if len(image_array) > 1:  #execute only if some action has been done
        image = image_array[0] #original image
        image_array = [image] #singleton list of original image
        action_array = [action_array[0]] #singleton list of 'Browse' action 
        last['text'] = action_array[0] #display 'Browse'
        PIL_image = Image.fromarray(image.astype('uint8'), 'RGB') #converted to PIL
        Tk_image = ImageTk.PhotoImage(PIL_image) #converted to ImageTk
        panel.configure(image = Tk_image) #configure original image
        panel.image = Tk_image #original image placed on panel

#https://stackoverflow.com/questions/57033158/how-to-save-images-with-the-save-button-on-the-tkinter-in-python
def save_as():
    """
    Function to save the modified image in any location in the device
    """
    global image
    path = filedialog.asksaveasfile(defaultextension=".png") #gets the path to location
    if path is None:  #if not chosen any path
        return
    PIL_image = Image.fromarray(image.astype('uint8'), 'RGB') #convert to PIL image
    PIL_image.save(path.name) #save the image

def enter_thresh_value():
    """
    Creates a window and takes user input for the threshold value
    """
    frame_t = tk.Tk() #new window
    frame_t.title("Enter Threshold Value") #title
    frame_t.geometry('300x80') #dimensions
    global thresh_entry  #variable to hold input ksize
    thresh_entry = tk.Text(frame_t, width = 10, height = 2, relief = 'solid', font = ('Comic Sans', 20)) #place to write input, for user
    thresh_entry.place(relx = 0, rely = 0) #positioning relative to borders
    thresh_btn = tk.Button(frame_t, text = 'Apply', command = partial(thresholding_gradient,frame_t),
                        height = 1, font = ('Arial', 12, "bold"), relief = 'raised') #user will press this button after typing
    thresh_btn.place(relx = 0.65, rely = 0.6) #positioning relative to borders
    frame_t.mainloop() # Window loops and waits for events

def color_balanced(org_img, r = 1, g = 1, b = 1, c = 1):
    image = org_img.copy()
    image[:,:,0] = image[:,:,0] * r 
    image[:,:,1] = image[:,:,1] * g
    image[:,:,2] = image[:,:,2] * b
    return image

from scipy import signal
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

sobelx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0 ,1]], np.float32)
sobely = np.array([[-1,-2,-1],
                   [0, 0 ,0],
                   [1, 2, 1]], np.float32)
gaussian = np.array([[1, 2, 1],
                    [2, 4 ,2],
                    [1, 2, 1]], np.float32) / 16

#Hysterisis Thresholding
def hist_thresh(pixel, th = 50, tl = 20):
    if pixel >= th:
        pixel = 255
    elif pixel >= tl:
        pixel = 20
    else:
        pixel = 0
    return pixel
v_hist_thresh = np.vectorize(hist_thresh)

#binary thresholding
def binary_thresh(pixel, t = 127):
    if pixel >= t:
        pixel = 255
    else:
        pixel = 0
    return pixel

v_binary_thresh = np.vectorize(binary_thresh)

#Functionalising the whole code
def canny_edge_detector(th = 50, tl = 25, color = True):
    global image, image_array, action_array
    arr=  np.array(image).copy()
    if color:
        org_red = arr[:,:,0]
        org_green = arr[:,:,1]
        org_blue = arr[:,:,2]

        G_arr = np.zeros(org_red.shape)
        theta_arr = np.zeros(org_red.shape) 
        dRdx = np.zeros(org_red.shape)
        dRdy = np.zeros(org_red.shape)
        dGdx = np.zeros(org_red.shape)
        dGdy = np.zeros(org_red.shape)
        dBdx = np.zeros(org_red.shape)
        dBdy = np.zeros(org_red.shape)
        red = org_red.copy()
        green = org_green.copy()
        blue = org_blue.copy()
        
        #smoothing with Gaussian kernel
        red = signal.convolve2d(org_red, gaussian)
        green = signal.convolve2d(org_green, gaussian)
        blue = signal.convolve2d(org_blue, gaussian)

        #calculating gradient using convolution with Sobel operators      
        dRdx = signal.convolve2d(red ,sobelx)
        dRdy = signal.convolve2d(red , sobely)
        dGdx = signal.convolve2d(green, sobelx)
        dGdy = signal.convolve2d(green, sobely)
        dBdx = signal.convolve2d(blue, sobelx)
        dBdy = signal.convolve2d(blue, sobely)
        gxx = abs(dRdx)**2 + abs(dGdx)**2 + abs(dBdx)**2
        gyy = abs(dRdy)**2 + abs(dGdy)**2 + abs(dBdy)**2
        gxy = dRdx*dRdy + dGdx*dGdy + dBdx*dBdy 
        theta_arr = 0.5*np.arctan2(2*gxy, gxx - gyy)
        G_arr = np.sqrt(0.5*(gxx + gyy + (gxx-gyy)*np.cos(2*theta_arr) + 2*gxy*np.sin(2*theta_arr)))
    
    else: #if grayscale image or user wants use of intensity channel
        if len(arr.shape) == 3:  
            org_red = arr[:,:,0]
            org_green = arr[:,:,1]
            org_blue = arr[:,:,2]
            org_arr = sum([org_red,org_green,org_blue])/3  #intensity
        
        dFdx = np.zeros(org_arr.shape)
        dFdy = np.zeros(org_arr.shape)
        new_arr = org_arr.copy()
        #smoothing with Gaussian kernel
        new_arr = signal.convolve2d(org_arr, gaussian)
        
        #calculating gradient using convolution with Sobel operators      
        dFdx = signal.convolve2d(new_arr, sobelx)
        dFdy = signal.convolve2d(new_arr, sobely)
        G_arr = np.sqrt(dFdx **2 + dFdy ** 2)
        theta_arr = np.arctan2(dFdy, dFdx)
        
    G_arr = G_arr * 255 / G_arr.max()  #scale grad values to range [0,255]
    Gn = np.zeros(G_arr.shape, dtype=np.int32)
    angle = theta_arr * 180. / np.pi  #convert to degrees
    
    for i in range(1,arr.shape[0]-1):
        for j in range(1,arr.shape[1]-1):

            q = 255 #will hold grad value of first neighbour
            r = 255 #will hold grad value of second neighbour
            theta = angle[i,j]

            #Four types of edge directions: Horizontal, Vertical, +45 deg, -45 deg
            #Digital Image Processing by Gonzalez and Woods, page 731
            if (-180 <= theta < -157.5) or (-22.5 <= theta < 22.5) or (157.5 <= theta <= 180):
                q = G_arr[i, j+1]
                r = G_arr[i, j-1]
            elif (22.5 <= theta < 67.5) or (-157.5 <= theta <= -112.5):
                q = G_arr[i+1, j-1]
                r = G_arr[i-1, j+1]
            elif (67.5 <= theta < 112.5) or (-112.5 <= theta < -67.5):
                q = G_arr[i+1, j]
                r = G_arr[i-1, j]
            elif (112.5 <= angle[i,j] < 157.5) or (-67.5 <= theta < -22.5):
                q = G_arr[i-1, j-1]
                r = G_arr[i+1, j+1]

            #if value of grad is more than value of grad at both neighbours 
            # ... along direction, then set Z = grad at [i,j]
            #otherwise set Z = 0 at [i,j]
            if (G_arr[i,j] >= q) and (G_arr[i,j] >= r):
                Gn[i,j] = G_arr[i,j]
            else:
                Gn[i,j] = 0
                
    strong_weak = v_hist_thresh(Gn, th, tl)
    final_canny = strong_weak.copy()
    for i in range(1, strong_weak.shape[0]-1):
        for j in range(1, strong_weak.shape[1]-1):
            if (strong_weak[i,j] != 0) and (strong_weak[i,j] != 255): #if weak pixel
                final_canny[i,j] = float(255 in strong_weak[i-1:i+2, j-1:j+2]) * 255
    
    image = final_canny
    rgb_PIL = Image.fromarray(image.astype('uint8')) #convert to PIL
    rgb_PIL = ImageTk.PhotoImage(rgb_PIL) #convert to PhotoImage
    panel.configure(image = rgb_PIL) #configure new image
    panel.image = rgb_PIL #place on the panel
    image_array.append(image) #store onto the stack
    last["text"] = "Canny Edge Detection" #Display action
    action_array.append("Canny Edge Detection") #store action onto stack


def thresholding_gradient(frame_t, color = True):
    t = int(thresh_entry.get(1.0, "end-1c"))
    global image, image_array, action_array
    arr=  np.array(image).copy()
    if color:
        org_red = arr[:,:,0]
        org_green = arr[:,:,1]
        org_blue = arr[:,:,2]

        G_arr = np.zeros(org_red.shape)
        theta_arr = np.zeros(org_red.shape) 
        dRdx = np.zeros(org_red.shape)
        dRdy = np.zeros(org_red.shape)
        dGdx = np.zeros(org_red.shape)
        dGdy = np.zeros(org_red.shape)
        dBdx = np.zeros(org_red.shape)
        dBdy = np.zeros(org_red.shape)
        red = org_red.copy()
        green = org_green.copy()
        blue = org_blue.copy()
        
        #smoothing with Gaussian kernel
        red = signal.convolve2d(org_red, gaussian)
        green = signal.convolve2d(org_green, gaussian)
        blue = signal.convolve2d(org_blue, gaussian)

        #calculating gradient using convolution with Sobel operators      
        dRdx = signal.convolve2d(red ,sobelx)
        dRdy = signal.convolve2d(red , sobely)
        dGdx = signal.convolve2d(green, sobelx)
        dGdy = signal.convolve2d(green, sobely)
        dBdx = signal.convolve2d(blue, sobelx)
        dBdy = signal.convolve2d(blue, sobely)
        gxx = abs(dRdx)**2 + abs(dGdx)**2 + abs(dBdx)**2
        gyy = abs(dRdy)**2 + abs(dGdy)**2 + abs(dBdy)**2
        gxy = dRdx*dRdy + dGdx*dGdy + dBdx*dBdy 
        theta_arr = 0.5*np.arctan2(2*gxy, gxx - gyy)
        G_arr = np.sqrt(0.5*(gxx + gyy + (gxx-gyy)*np.cos(2*theta_arr) + 2*gxy*np.sin(2*theta_arr)))
    
    else: #if grayscale image or user wants use of intensity channel
        if len(arr.shape) == 3:  
            org_red = arr[:,:,0]
            org_green = arr[:,:,1]
            org_blue = arr[:,:,2]
            org_arr = sum([org_red,org_green,org_blue])/3  #intensity
        
        dFdx = np.zeros(org_arr.shape)
        dFdy = np.zeros(org_arr.shape)
        new_arr = org_arr.copy()
        #smoothing with Gaussian kernel
        new_arr = signal.convolve2d(org_arr, gaussian)
        
        #calculating gradient using convolution with Sobel operators      
        dFdx = signal.convolve2d(new_arr, sobelx)
        dFdy = signal.convolve2d(new_arr, sobely)
        G_arr = np.sqrt(dFdx **2 + dFdy ** 2)
        theta_arr = np.arctan2(dFdy, dFdx)
        
    G_arr = G_arr * 255 / G_arr.max()  #scale grad values to range [0,255]
    binary_grad = v_binary_thresh(G_arr, t)
    image = binary_grad
    rgb_PIL = Image.fromarray(image.astype('uint8')) #convert to PIL
    rgb_PIL = ImageTk.PhotoImage(rgb_PIL) #convert to PhotoImage
    panel.configure(image = rgb_PIL) #configure new image
    panel.image = rgb_PIL #place on the panel
    image_array.append(image) #store onto the stack
    last["text"] = "Thresholding Gradient" #Display action
    action_array.append("Thresholding Gradient") #store action onto stack

def dilate(struct_dim = (5,5)):
    global image, image_array, action_array
    arr = np.array(image).copy()
    if arr.max() == 255:
        arr = arr/255
    dilated = arr.copy()
    h = (struct_dim[0]-1)//2
    b = (struct_dim[1]-1)//2
    for x in range(1, arr.shape[0]-1):
        for y in range(1, arr.shape[1]-1):
            if 1 in arr[x-h:x+h+1, y-b:y-b+1]:
                dilated[x,y] = 1.0
            else:
                dilated[x,y] = 0.0
#     return dilated
    image = dilated * 255
    rgb_PIL = Image.fromarray(image.astype('uint8')) #convert to PIL
    rgb_PIL = ImageTk.PhotoImage(rgb_PIL) #convert to PhotoImage
    panel.configure(image = rgb_PIL) #configure new image
    panel.image = rgb_PIL #place on the panel
    image_array.append(image) #store onto the stack
    last["text"] = "Dilation" #Display action
    action_array.append("Dilation") #store action onto stack

def erode(struct_dim = (5,5)):
    global image, image_array, action_array
    arr = np.array(image).copy()
    if arr.max() == 255:
        arr = arr/255
    eroded = arr.copy()
    h = (struct_dim[0]-1)//2
    b = (struct_dim[1]-1)//2
    for x in range(1, arr.shape[0]-1):
        for y in range(1, arr.shape[1]-1):
            if 0 in arr[x-h:x+h+1, y-b:y-b+1]:
                eroded[x,y] = 0.0
            else:
                eroded[x,y] = 1.0
#     return eroded
    image = eroded * 255
    rgb_PIL = Image.fromarray(image.astype('uint8')) #convert to PIL
    rgb_PIL = ImageTk.PhotoImage(rgb_PIL) #convert to PhotoImage
    panel.configure(image = rgb_PIL) #configure new image
    panel.image = rgb_PIL #place on the panel
    image_array.append(image) #store onto the stack
    last["text"] = "Erosion" #Display action
    action_array.append("Erosion") #store action onto stack

def opening(org_image, struct_dim = (5,5)):
    eroded = erode(org_image, struct_dim)
    opened = dilate(eroded, struct_dim)
    return opened

def closing(org_image, struct_dim = (5,5)):
    dilated = dilate(org_image, struct_dim)
    closed = erode(eroded, struct_dim)
    return closed

"""## $\text{Super-resolution}$

The objective of super-resolution is to scale the image to a larger size without degrading the quality of the image. 

For this task, we utilize a neural network architecure called "Residual Dense Network", designed by Zhang et al (paper in refernces).
"""

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def denormalize(img): # converts the pixels back to standard scale [0..255]
    img = img.mul(255.0).clamp(0.0, 255.0)
    return img

def scalex2():
    global image, image_array, action_array

    model2 = RDN(n=2,    # scaling factor
                num_channels=3,     # 
                G_0=64,    # G_0 : refer report
                G=64,     # G
                D=16,      # D
                C=8).to(device)    # C and send to available device

    state_dict = model2.state_dict()
    state_keys = [key for key in state_dict.keys()]
    for (i, (n, p)) in enumerate(torch.load(fr"{absolute}\rdn_x2.pth", map_location=lambda storage, loc: storage).items()):
        # print(f"n = {n}, key = {state_keys[i]}")
        state_dict[state_keys[i]].copy_(p)

    model2.eval()   # convert model to testing mode

    image = Image.fromarray(image)
    image_width = (image.width // 2) * 2        # converts image dimensions to nearest multiple of scaling_factor
    image_height = (image.height // 2) * 2      # converts image dimensions to nearest multiple of scaling_factor
    lr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    lr = np.expand_dims(np.array(lr).astype(np.float32).transpose([2, 0, 1]), 0) / 255.0
    lr = torch.from_numpy(lr).to(device)

    with torch.no_grad():       # interpolate the new image from original
        preds = model2(lr).squeeze(0)
    image = denormalize(preds).permute(1, 2, 0).byte().cpu().numpy()

    rgb_PIL = Image.fromarray(image.astype('uint8')) #convert to PIL
    rgb_PIL = ImageTk.PhotoImage(rgb_PIL) #convert to PhotoImage
    panel.configure(image = rgb_PIL) #configure new image
    panel.image = rgb_PIL #place on the panel
    image_array.append(image) #store onto the stack
    last["text"] = "Scale x2" #Display action
    action_array.append("Scale x2") #store action onto stack

def scalex3():
    global image, image_array, action_array

    model3 = RDN(n=3,    # scaling factor
                num_channels=3,     # 
                G_0=64,    # G_0 : refer report
                G=64,     # G
                D=16,      # D
                C=8).to(device)    # C and send to available device

    state_dict = model3.state_dict()
    state_keys = [key for key in state_dict.keys()]
    for (i, (n, p)) in enumerate(torch.load(rf"{absolute}\rdn_x3.pth", map_location=lambda storage, loc: storage).items()):
        # print(f"n = {n}, key = {state_keys[i]}")
        state_dict[state_keys[i]].copy_(p)

    model3.eval()   # convert model to testing mode

    image = Image.fromarray(image)
    image_width = (image.width // 2) * 2        # converts image dimensions to nearest multiple of scaling_factor
    image_height = (image.height // 2) * 2      # converts image dimensions to nearest multiple of scaling_factor

    lr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)

    lr = np.expand_dims(np.array(lr).astype(np.float32).transpose([2, 0, 1]), 0) / 255.0
    lr = torch.from_numpy(lr).to(device)

    with torch.no_grad():       # interpolate the new image from original
        preds = model3(lr).squeeze(0)
    
    image = denormalize(preds).permute(1, 2, 0).byte().cpu().numpy()

    rgb_PIL = Image.fromarray(image.astype('uint8')) #convert to PIL
    rgb_PIL = ImageTk.PhotoImage(rgb_PIL) #convert to PhotoImage
    panel.configure(image = rgb_PIL) #configure new image
    panel.image = rgb_PIL #place on the panel
    image_array.append(image) #store onto the stack
    last["text"] = "Scale x3" #Display action
    action_array.append("Scale x3") #store action onto stack

def scalex4():
    global image, image_array, action_array

    model4 = RDN(n=4,    # scaling factor
                num_channels=3,     # 
                G_0=64,    # G_0 : refer report
                G=64,     # G
                D=16,      # D
                C=8).to(device)    # C and send to available device

    state_dict = model4.state_dict()
    state_keys = [key for key in state_dict.keys()]
    for (i, (n, p)) in enumerate(torch.load(fr"{absolute}\rdn_x4.pth", map_location=lambda storage, loc: storage).items()):
        # print(f"n = {n}, key = {state_keys[i]}")
        state_dict[state_keys[i]].copy_(p)

    model4.eval()   # convert model to testing mode

    image = Image.fromarray(image)
    image_width = (image.width // 2) * 2        # converts image dimensions to nearest multiple of scaling_factor
    image_height = (image.height // 2) * 2      # converts image dimensions to nearest multiple of scaling_factor

    lr = image.resize((image_width, image_height), resample=pil_image.BICUBIC)

    lr = np.expand_dims(np.array(lr).astype(np.float32).transpose([2, 0, 1]), 0) / 255.0
    lr = torch.from_numpy(lr).to(device)

    with torch.no_grad():       # interpolate the new image from original
        preds = model4(lr).squeeze(0)
    
    image = denormalize(preds).permute(1, 2, 0).byte().cpu().numpy()

    rgb_PIL = Image.fromarray(image.astype('uint8')) #convert to PIL
    rgb_PIL = ImageTk.PhotoImage(rgb_PIL) #convert to PhotoImage
    panel.configure(image = rgb_PIL) #configure new image
    panel.image = rgb_PIL #place on the panel
    image_array.append(image) #store onto the stack
    last["text"] = "Scale x4" #Display action
    action_array.append("Scale x4") #store action onto stack

# https://www.geeksforgeeks.org/file-explorer-in-python-using-tkinter/
# https://stackoverflow.com/questions/10133856/how-to-add-an-image-in-tkinter

h = 1000
w = 1000

root = tk.Tk()  #Main GUI Window
root.title('Image Enhancer') #its title
root.geometry("{}x{}".format(w,h))  #its dimensions

panel = tk.Label(root, background = 'white', relief = 'raised') #panel to display the image. Parent: root
panel.pack(side = 'left', anchor = tk.NW) #anchored to top right

my_font = font.Font(family='Helvetica', size=18, weight='bold')
buttonFrame = tk.Frame(root, width = 100) #creating a frame to hold all the buttons in one place. Parent: root
buttonFrame.pack(side = 'top', anchor = tk.NE) #achored to top left corner


cannyB = tk.Button(buttonFrame, text = "Canny Edge Detection", command = canny_edge_detector, 
                    height = 1, width = 30, font = my_font, bg = 'orange')
threshB = tk.Button(buttonFrame, text = "Thresholding Gradient", command = enter_thresh_value, 
                    height = 1, width = 30, font = my_font, bg = 'orange')
erosionB = tk.Button(buttonFrame, text = "Erosion", command = erode, 
                    height = 1, width = 30, font = my_font, bg = 'cyan')
dilationB = tk.Button(buttonFrame, text = "Dilation", command = dilate, 
                    height = 1, width = 30, font = my_font, bg = 'cyan')
scalex2B = tk.Button(buttonFrame, text = "Scale x2", command = scalex2, 
                    height = 1, width = 30, font = my_font, bg = 'green')
scalex3B = tk.Button(buttonFrame, text = "Scale x3", command = scalex3, 
                    height = 1, width = 30, font = my_font, bg = 'green')
scalex4B = tk.Button(buttonFrame, text = "Scale x4", command = scalex4, 
                    height = 1, width = 30, font = my_font, bg = 'green')

# #defining buttons for every operation with commands having respective function calls. Parent: buttonFrame
# #when button is pressed the function under command will be called
browseB = tk.Button(buttonFrame, text = "Browse", command = select_img, 
                    height = 1, width = 20, font = my_font, bg = 'lightgreen')
saveB = tk.Button(buttonFrame, text = "Save As", command = save_as, 
                  height = 1, width = 20, font = my_font, bg = 'lightgreen')
undoB = tk.Button(buttonFrame, text = "Undo", command = undo, 
                  height = 1, width = 20, font = my_font, bg = 'lightpink')
undo_allB = tk.Button(buttonFrame, text = "Undo All", command = undo_all, 
                      height = 1, width = 20, font = my_font, bg = 'lightpink')

last_label = tk.Label(buttonFrame, text = "Last Action:", height = 2, width = 20, font = my_font, bg = 'yellow')
last = tk.Label(buttonFrame, text = "No Action", height = 2, width = 20, font = my_font,  bg = 'yellow', fg = 'red')
# # label to display last action

# Some additional instructions displayed through a label and text. Parent: buttonFrame
# info_text = '''Click the Browse button to select an image from your system, 
# apply the necessary method(s) and then save the image using the Save As button.
# '''
# info = tk.Label(buttonFrame, text = info_text, font = ('Helvetica', 15), height = 15, padx = 5, wraplength = 200)
# info.pack(side = 'bottom', fill = 'both') #at the bottom of buttonFrame

# #placing all the buttons in the buttonFrame frame
browseB.pack()
saveB.pack()
undoB.pack()
undo_allB.pack()
# eq_histB.pack()
# gamma_correctB.pack()
# log_xformB.pack()
# blur_button.pack()
# sharpen_button.pack()
# threshold_button.pack()
last_label.pack()
last.pack()
cannyB.pack()
threshB.pack()
erosionB.pack()
dilationB.pack()
scalex2B.pack()
scalex3B.pack()
scalex4B.pack()


root.mainloop() # Window loops and waits for events