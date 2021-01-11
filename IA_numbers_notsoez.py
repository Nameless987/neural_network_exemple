from tkinter import *
import numpy as np
from math import *
from random import *
import time
import random
import keyboard


tk = Tk()
cnv=Canvas(tk, width=800, height=400, bg="grey")
cnv.pack(padx=0, pady=0)

#-------------------------------------------------------------------------------------------------------------------------------------------
#variables initiales

c = []
ind = []

for i in range(0, 7):
    c.append("red")
    ind.append(1)

nb_guessed = 0
weights_i_h = []
weights_h_o = []
listeBlanche = []

s0 = cnv.create_rectangle(225, 145, 235, 195, fill = c[0])
s1 = cnv.create_rectangle(225, 205, 235, 255, fill = c[1])
s2 = cnv.create_rectangle(175, 255, 225, 265, fill = c[2])
s3 = cnv.create_rectangle(165, 205, 175, 255, fill = c[3])
s4 = cnv.create_rectangle(165, 145, 175, 195, fill = c[4])
s5 = cnv.create_rectangle(175, 135, 225, 145, fill = c[5])
s6 = cnv.create_rectangle(175, 195, 225, 205, fill = c[6])

def init():
    cnv.create_rectangle(150, 280, 250, 380, fill="gold")
    cnv.create_line(400, 0, 400, 400, fill="black")
    cnv.create_oval(575, 175, 625, 225, fill="white")
    cnv.create_oval(575, 100, 625, 150, fill="white")
    cnv.create_oval(575, 250, 625, 300, fill="white")
    cnv.create_oval(575, 325, 625, 375, fill="white")
    cnv.create_oval(575, 25, 625, 75, fill="white")

    #___________________________________________________________________________________________________________________________________________

    cnv.create_oval(425, 175, 475, 225, fill=c[0])
    cnv.create_oval(425, 125, 475, 175, fill=c[1])
    cnv.create_oval(425, 225, 475, 275, fill=c[2])
    cnv.create_oval(425, 275, 475, 325, fill=c[3])
    cnv.create_oval(425, 75, 475, 125, fill=c[4])
    cnv.create_oval(425, 25, 475, 75, fill=c[5])
    cnv.create_oval(425, 325, 475, 375, fill=c[6])

    #___________________________________________________________________________________________________________________________________________

    cnv.create_oval(725, 175, 775, 225, fill="white")

for i in range(0, 7):
    for j in range(0, 5):
        cnv.create_line(600, 75*j+50, 450, 50*i+50, width=1, fill=c[i])
        weights_i_h.append(1)
for i in range(0, 5):
    cnv.create_line(600, 75*i+50, 750, 200, width=1, fill="black")
    weights_h_o.append(1)

def lines():
    for i in range(0, 7):
        for j in range(0, 5):
            cnv.create_line(600, 75*j+50, 450, 50*i+50, width=int(10*nn.sigmoid(nn.w1[i][j])), fill="black")

    for i in range(0, 5):
        cnv.create_line(600, 75*i+50, 750, 200, width=int(10*nn.sigmoid(nn.w2[i])), fill="black")

texte = 0

#-------------------------------------------------------------------------------------------------------------------------------------------
#class

entrees = np.array(([1, 1, 1, 1, 1, 1, -1], [1, 1, -1, -1, -1, -1, -1], [1, -1, 1, 1, -1, 1, 1], [1, 1, 1, -1, -1, 1, 1], [1, 1, -1, -1, 1, -1, 1], [-1, 1, 1, -1, 1, 1, 1], [-1, 1, 1, 1, 1, 1, 1], [1, 1, -1, -1, -1, 1, -1], [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, -1, 1, 1, 1]), dtype=float)
sorties = np.array(([0], [0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7], [0.8], [0.9]), dtype=float)
entrees = entrees/np.amax(entrees, axis=0)
vals = np.split(entrees, [10])[0]

class neural_network(object):
    def __init__(self):
        self.inputSize = 7
        self.hiddenSize = 5
        self.outputSize = 1
        self.w1 = np.random.randn(self.inputSize, self.hiddenSize)
        self.w2 = np.random.randn(self.hiddenSize, self.outputSize)
  
    def forward(self, vals):
        self.r = np.dot(vals, self.w1)
        self.r_s = self.sigmoid(self.r)
        self.r2 = np.dot(self.r_s, self.w2)
        result = self.sigmoid(self.r2)
        return result

    def backward(self, vals, sorties, result):
        self.out_error = sorties-result
        self.out_delta = self.out_error*self.d_sigmoid(result)

        self.r_s_error = self.out_delta.dot(self.w2.T)
        self.r_s_delta = self.r_s_error*self.d_sigmoid(self.r_s)

        self.w1 += vals.T.dot(self.r_s_delta)
        self.w2 += self.r_s.T.dot(self.out_delta)

    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
  
    def d_sigmoid(self, x):
        return 1/((1+np.exp((5*(x-0.5))**2)))

    def train(self, vals, sorties):
        result = self.forward(vals)
        self.backward(vals, sorties, result)
    
    def predict(self):
        return np.matrix.round(self.forward(predictions), 1)*10

nn = neural_network()

#-------------------------------------------------------------------------------------------------------------------------------------------
#draw

def draw():
    global s0, s1, s2, s3, s4, s5, s6, nb_guessed, texte
    for i in range(0, 7):
        if(ind[i] == 1):
            c[i] = "red"
        else:
            c[i] = "grey"
    cnv.delete(s0)
    cnv.delete(s1)
    cnv.delete(s2)
    cnv.delete(s3)
    cnv.delete(s4)
    cnv.delete(s5)
    cnv.delete(s6)
    s0 = cnv.create_rectangle(225, 145, 235, 195, fill = c[0])
    s1 = cnv.create_rectangle(225, 205, 235, 255, fill = c[1])
    s2 = cnv.create_rectangle(175, 255, 225, 265, fill = c[2])
    s3 = cnv.create_rectangle(165, 205, 175, 255, fill = c[3])
    s4 = cnv.create_rectangle(165, 145, 175, 195, fill = c[4])
    s5 = cnv.create_rectangle(175, 135, 225, 145, fill = c[5])
    s6 = cnv.create_rectangle(175, 195, 225, 205, fill = c[6])
    cnv.delete(texte)
    texte = cnv.create_text(200, 330, font=('Arial',75,'bold italic'), text = nb_guessed, fill = "red")

#-------------------------------------------------------------------------------------------------------------------------------------------
#key

def key(pos):
    global nb_guessed
    if(pos.x <= 225 and pos.x >= 175):
        if(pos.y <= 205 and pos.y >= 195):
            ind[6] *= -1
        elif(pos.y <= 145 and pos.y >= 135):
            ind[5] *= -1
        elif(pos.y <= 265 and pos.y >= 255):
            ind[2] *= -1
    elif(pos.x <= 235 and pos.x >= 225):
        if(pos.y <= 195 and pos.y >= 145):
            ind[0] *= -1
        elif(pos.y <= 255 and pos.y >= 205):
            ind[1] *= -1
    elif(pos.x <= 175 and pos.x >= 165):
        if(pos.y <= 195 and pos.y >= 145):
            ind[4] *= -1
        elif(pos.y <= 255 and pos.y >= 205):
            ind[3] *= -1
    else:
        pass

#-------------------------------------------------------------------------------------------------------------------------------------------
#main

for i in range(0, 100000):
    pass

def main():
    global nb_guessed, predictions
    cnv.delete(ALL)
    init()
    lines()
    cnv.bind("<Button-1>", key)
    nn.train(vals, sorties)
    predictions = np.array(([ind[0], ind[1], ind[2], ind[3], ind[4], ind[5], ind[6]]), dtype=float)
    nb_guessed = int(nn.predict())
    draw()
    print(nn.w1[0][0])
    tk.after(15, main)

#-------------------------------------------------------------------------------------------------------------------------------------------

main()
tk.mainloop()