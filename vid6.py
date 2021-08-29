import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib import image
import numpy as np
from functions import *


def import_callback():
    global image_og
    global image_curr
    global image_prev
    global display_image

    filename = filedialog.askopenfilename()
    image_og = image.imread(filename)

    if filename.split(".")[-1] =="png":
        image_og = image_og[:, :, :3]
        image_og = (image_og*255).astype("uint8")

    if len(image_og.shape) == 2:
        img_temp = np.zeros((image_og.shape[0], image_og.shape[1], 3))
        img_temp[:,:,0], img_temp[:,:,1], img_temp[:,:,2] = image_og.copy(), image_og.copy(), image_og.copy()
        image_og = img_temp.astype("uint8")

    image_curr = image_og
    image_prev = image_og

    display_image = Image.fromarray(image_curr)
    displayImage(display_image)

def displayImage(display_image):
    ImagetoDisplay = display_image.resize((900,600), Image.ANTIALIAS)
    ImagetoDisplay = ImageTk.PhotoImage(display_image)
    showWindow.config(image=ImagetoDisplay)
    showWindow.photo_ref = ImagetoDisplay
    showWindow.pack()

def equalize_callback():
    global image_og
    global image_curr
    global image_prev
    global display_image

    image_prev = image_curr
    image_curr = hist_equalize(image_curr)
    display_image = Image.fromarray(image_curr)
    displayImage(display_image)

def log_callback():
    global image_og
    global image_curr
    global image_prev
    global display_image

    image_prev = image_curr
    image_curr = log_transform(image_curr)
    display_image = Image.fromarray(image_curr)
    displayImage(display_image)

def undo_callback():
    global image_curr
    global image_prev

    image_curr = image_prev
    display_image = Image.fromarray(image_curr)
    displayImage(display_image)

def undoAll_callback():
    global image_og
    global image_curr
    global image_prev

    image_curr = image_prev = image_og
    display_image = Image.fromarray(image_curr)
    displayImage(display_image)

def save_callback():
    savefile = filedialog.asksaveasfile(defaultextension=".jpg")
    display_image.save(savefile)

def close_callback():
    window.destroy()

def brightness_callback():
    global beta_input
    global image_og
    global image_curr
    global image_prev
    global display_image

    beta = float(beta_input.get())
    beta_input.set("")
    image_prev = image_curr
    image_curr = brightness(image_curr, beta)
    display_image = Image.fromarray(image_curr)
    displayImage(display_image)

def contrast_callback():
    global alpha_input
    global image_og
    global image_curr
    global image_prev
    global display_image

    alpha = float(alpha_input.get())
    alpha_input.set("")
    image_prev = image_curr
    image_curr = contrast(image_curr, alpha)
    display_image = Image.fromarray(image_curr)
    displayImage(display_image)

def gamma_callback():
    global gamma_input
    global image_og
    global image_curr
    global image_prev
    global display_image

    gamma = float(gamma_input.get())
    gamma_input.set("")
    image_prev = image_curr
    image_curr = gamma_corrected(image_curr, gamma)
    display_image = Image.fromarray(image_curr)
    displayImage(display_image)

def blur_callback(pos):
    global sigma
    sigma = float(pos)

def apply_blur():
    global image_og
    global image_curr
    global image_prev
    global display_image
    global sigma

    image_prev = image_curr
    image_curr = gaussian_blur(image_curr, sigma=sigma)
    display_image = Image.fromarray(image_curr)
    displayImage(display_image)

def sharpen_callback(pos):
    global correction
    correction = float(pos)

def apply_sharpen():
    global image_og
    global image_curr
    global image_prev
    global display_image
    global correction

    image_prev = image_curr
    image_curr = sharpen_lap(image_curr, correction)
    display_image = Image.fromarray(image_curr)
    displayImage(display_image)

window = tk.Tk() # creating a GUI window

# making the window full-screen
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()
window.geometry(f"{screen_width}x{screen_height}")

Frame1 = tk.Frame(window, height=20, width=200)
Frame1.pack(anchor=tk.N)

Frame2 = tk.Frame(window, height=20, width=screen_width)
Frame2.pack(anchor=tk.NW)

Frame3 = tk.Frame(window, height=20)
Frame3.pack(anchor=tk.NW)


importButton = tk.Button(Frame1, text="import", padx=10, pady=5, command=import_callback)
importButton.grid(row=0, column=0)

saveButton = tk.Button(Frame1, text="Save", padx=10, pady=5, command=save_callback)
saveButton.grid(row=0, column=1)

closeButton = tk.Button(Frame1, text="Close", padx=10, pady=5, command=close_callback)
closeButton.grid(row=0, column=2 )

equalizeButton = tk.Button(Frame1, text="Equalize", padx=10, pady=5, command=equalize_callback)
equalizeButton.grid(row=1, column=0 )

logButton = tk.Button(Frame1, text="Log Transform", padx=10, pady=5, command=log_callback)
logButton.grid(row=1, column=1)

undoButton = tk.Button(Frame1, text="Undo", padx=10, pady=5, command=undo_callback)
undoButton.grid(row=1, column=2)

undoAllButton = tk.Button(Frame1, text="Undo All", padx=10, pady=5, command=undoAll_callback)
undoAllButton.grid(row=2, column=1)

# brightnessSlider = tk.Scale(Frame2, label="Select Brightness",
#                             from_=-10, to=10, orient=tk.HORIZONTAL,
#                             length=screen_width/2, resolution=0.1,
#                             command=brightness_callback)
# brightnessSlider.pack(anchor=tk.N)
#
# brightnessButton = tk.Button(Frame2, text="Apply Brightness", command=apply_brightness)
# brightnessButton.pack()

blurSlider = tk.Scale(Frame2, label="Select Sigma",
                            from_=0.1, to=5, orient=tk.HORIZONTAL,
                            length=screen_width/2, resolution=0.1,
                            command=blur_callback)
blurSlider.pack(anchor=tk.N)

blurButton = tk.Button(Frame2, text="Blur", command=apply_blur)
blurButton.pack()

sharpenSlider = tk.Scale(Frame2, label="Selecting Sharpness",
                            from_=0, to=10, resolution=0.1,
                            orient=tk.HORIZONTAL, length=screen_width/2,
                            command=sharpen_callback)
sharpenSlider.pack(anchor=tk.N)

sharpenButton = tk.Button(Frame2, text="Sharpen", command=apply_sharpen)
sharpenButton.pack()

gamma_input = tk.StringVar()

gamma_entry = tk.Label(Frame3, text="Enter Gamma")
gamma_entry.grid(row=0, column=0)

gammaEntry = tk.Entry(Frame3, bd=7, textvariable=gamma_input)
gammaEntry.grid(row=1, column=0)

correctButton = tk.Button(Frame3, text="Gamma Correct", padx=10, pady=5, command=gamma_callback)
correctButton.grid(row=2, column=0)

#----------------------------------
beta_input = tk.StringVar()

beta_entry = tk.Label(Frame3, text="Enter Brightness (-ve or +ve)")
beta_entry.grid(row=0, column=1)

betaEntry = tk.Entry(Frame3, bd=7, textvariable=beta_input)
betaEntry.grid(row=1, column=1)

brightnessButton = tk.Button(Frame3, text="Apply Brightness", padx=10, pady=5, command=brightness_callback)
brightnessButton.grid(row=2, column=1)

#----------------------------------
alpha_input = tk.StringVar()

alpha_entry = tk.Label(Frame3, text="Enter Contrast (+ve)")
alpha_entry.grid(row=0, column=2)

alphaEntry = tk.Entry(Frame3, bd=7, textvariable=alpha_input)
alphaEntry.grid(row=1, column=2)

brightnessButton = tk.Button(Frame3, text="Apply Contrast", padx=10, pady=5, command=contrast_callback)
brightnessButton.grid(row=2, column=2)



showWindow = tk.Label(window)
showWindow.pack()

tk.mainloop() # running the window
