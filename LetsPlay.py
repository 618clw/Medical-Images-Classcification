import tkinter as tk
import PIL
from PIL import Image
import imageio
# from PIL import ImageTK
window = tk.Tk()

def create_gif(image_list, gif_name):
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
        # Save them as frames into a gif
    imageio.mimsave(gif_name, frames, 'GIF', duration=0.1)

    return

def gif_heatmap():
    image_list = ['heatmap.png']
    gif_name = 'heatmap.gif'
    create_gif(image_list, gif_name)

def gif_iniImag(init):
    image_list = [str(init)+'.png']
    gif_name = 'init.gif'
    create_gif(image_list, gif_name)

def paosini():

    # Classfying

    CResult = 1
    if CResult == 1:
        confirmLabel.config(text="Atelectasis")
    elif CResult == 2:
        confirmLabel.config(text="Cardiomegaly")
    elif CResult == 3:
        confirmLabel.config(text="Effusion")
    elif CResult == 4:
        confirmLabel.config(text="Infiltration")
    elif CResult == 5:
        confirmLabel.config(text="Mass")
    elif CResult == 6:
        confirmLabel.config(text="Nodule")
    elif CResult == 7:
        confirmLabel.config(text="Pneumonia")
    elif CResult == 8:
        confirmLabel.config(text="Pneumothorax")
    elif CResult == 9:
        confirmLabel.config(text="Consolidation")
    elif CResult == 10:
        confirmLabel.config(text="Edema")
    elif CResult == 11:
        confirmLabel.config(text="Emphysema")
    elif CResult == 12:
        confirmLabel.config(text="Fibrosis")
    elif CResult == 13:
        confirmLabel.config(text="Pleural Thickening")
    elif CResult == 14:
        confirmLabel.config(text="Hernia")

def Showing_iniImag():
    global ImagName
    ImagName = ImagEntry.get()
    print('name:')
    print(ImagName)
    gif_iniImag(ImagName)     # convert png to gif

    img_png_1 = tk.PhotoImage(file='init.gif')
    label_img1 = tk.Label(image=img_png_1)
    label_img1.image = img_png_1
    label_img1.pack(side='left')

def Showing_heatmap():
    gif_heatmap()     # convert png to gif
    img_png_2 = tk.PhotoImage(file='heatmap.gif')
    label_img2 = tk.Label(image=img_png_2)
    label_img2.image = img_png_2
    label_img2.pack(side='right')


ImagLabel = tk.Label(window, text="ImagName: ")
ImagEntry = tk.Entry(window)
ImagName = ImagEntry.get()

button1 = tk.Button(window, text="Show iniPicture", command=Showing_iniImag)
# img_open = Image.open(ImagName)
# img_png = ImageTK.PhotoImage(img_open)
# label_img = tk.Label(root, image=img_png)

button2 = tk.Button(window, text="Run", command=paosini)
button3 = tk.Button(window, text="Show Heatmap", command=Showing_heatmap)
confirmLabel = tk.Label(window)

ImagLabel.pack()
ImagEntry.pack()
button1.pack()
# label_img.pack()
button2.pack()
button3.pack()
confirmLabel.pack()

window.mainloop()










