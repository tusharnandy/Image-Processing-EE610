from tkinter import *
from PIL import ImageTk, Image
from matplotlib import image
from functions import hist_equalize
# import matplot.pyplot as plt

root = Tk()
root.title("Image Editor")
root.iconbitmap()

img_og = image.imread("images/swiss-alps.jpg")
img_current = img_og
img_prev = img_og

my_img = ImageTk.PhotoImage(Image.fromarray(img_current))
my_label = Label(image=my_img)
my_label.grid(row=0, column=0, columnspan=3)

def do_something():
    global my_label
    global button_equalize
    global img_prev
    global img_current

    # img_prev = img_current
    # img_current = hist_equalize(img_current)

    my_label.grid_forget()
    my_img = ImageTk.PhotoImage(Image.fromarray(img_prev))
    my_label = Label(image=my_img)

    button_equalize = Button(root, text="equalize", command = lambda: do_something())

    my_label.grid(row=0, column=0, columnspan=3)
    button_equalize.grid(row=1, column=0)


button_equalize = Button(root, text="equalize", command = lambda: do_something())
button_equalize.grid(row=1, column=0)

root.mainloop()
