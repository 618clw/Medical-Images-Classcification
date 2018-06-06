import tkinter as tk
import PIL
from PIL import Image
#from PIL import ImageTK
window = tk.Tk()

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

def Showing_Imag():
    global ImagName
    ImagName = ImagEntry.get()
    print('name:')
    print(ImagName)
    img_open = Image.open("D:/Program Files/Python/chexnet1-master/"+str(ImagName)+".png")
    img_open.show()
    # img_png = PIL.ImageTK.PhotoImage(img_open)
    # label_img = tk.Label(image=img_open)
    # label_img.pack()

ImagLabel = tk.Label(window, text="ImagName: ")
ImagEntry = tk.Entry(window)
ImagName = ImagEntry.get()

button1 = tk.Button(window, text="Show Picture", command=Showing_Imag)
# img_open = Image.open(ImagName)
# img_png = ImageTK.PhotoImage(img_open)
# label_img = tk.Label(root, image=img_png)

button2 = tk.Button(window, text="Run", command=paosini)
confirmLabel = tk.Label(window)

ImagLabel.pack()
ImagEntry.pack()
button1.pack()
# label_img.pack()
button2.pack()
confirmLabel.pack()

window.mainloop()










