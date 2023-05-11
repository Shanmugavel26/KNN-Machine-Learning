#Import libraries
import tkinter as tk
from tkinter import*
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Load the dataset
dataset = pd.read_csv("C:/Users\HP\Desktop\ML\Iris_new.csv")
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train the model
classifier = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
classifier.fit(X_train, y_train)


#GUI
root=tk.Tk()
root.title("KNN CLASSIFIER")
root.geometry("1000x1000")
icon=root.iconbitmap("C:/Users\HP\Downloads\mlic.ico")
photo=PhotoImage(file="C:/Users\HP\Downloads\MLimage.png")
label1=Label(root,image=photo,height=550,width=1120,bg="#3c6382").place(x=130,y=100)
frame1=Frame(root,height=500,width=560,bg="white").place(x=175,y=130)


border_frame1=Frame(frame1,height=561,width=7,bg="#01a3a4").place(x=123,y=93)
border_frame2=Frame(root,height=7,width=1128,bg="#01a3a4").place(x=130,y=93)
border_frame3=Frame(root,height=561,width=7,bg="#01a3a4").place(x=1254,y=93)
border_frame4=Frame(root,height=7,width=1128,bg="#01a3a4").place(x=130,y=647)

title_label=Label(root, text="PREDICT  SPECIES",font=("script MJ bold",16,"bold"),bg="#f39c12")
title_label.place(x=370,y=150)

label3 =Label(root, text="SEPAL LENGTH:",font=("georgia",10,"bold"),bg="#718093")
label3.place(x=230,y=220)
entry1 =Entry(frame1,width=30)
entry1.place(x=410,y=220)

label4 =Label(frame1, text="SEPAL WIDTH:",font=("georgia",10,"bold"),bg="#718093")
label4.place(x=230,y=270)
entry2 = tk.Entry(frame1,width=30)
entry2.place(x=410,y=270)

label5 =Label(frame1, text="PETAL LENGTH:",font=("georgia",10,"bold"),bg="#718093")
label5.place(x=230,y=320)
entry3 = tk.Entry(frame1)
entry3.place(x=410,y=320)

label6 =Label(frame1, text="PETAL WIDTH:",font=("georgia",10,"bold"),bg="#718093")
label6.place(x=230,y=370)
entry4 = tk.Entry(frame1)
entry4.place(x=410,y=370)

# Define the prediction function
def predict():
    sepal_length = float(entry1.get())
    sepal_width = float(entry2.get())
     
    test_data = sc.transform([[sepal_length, sepal_width]])
    prediction = classifier.predict(test_data)
    result_label.config(font=("georgia",20,"bold"),bg="#2ed573",text="Predicted Species: " + prediction[0])
    result_label.place(x=250,y=450)

def predict():
    petal_length=float(entry3.get())
    petal_width=float(entry3.get())
    test_data = sc.transform([[petal_length,petal_width]])
    prediction = classifier.predict(test_data)
    result_label.config(font=("georgia",20,"bold"),bg="#2ed573",text="Predicted Species: " + prediction[0])
    result_label.place(x=250,y=450)
    
def clear():
    widgets =(entry1,entry2,entry3,entry4)
    for i in widgets:
        i.delete(0,END)

# Define the prediction button
button =Button(root, text="PREDICT",font=("georgia",14,"bold"),bg="#0097e6",command=predict)
button.place(x=560,y=540)

button =Button(root, text="CLEAR",font=("georgia",14,"bold"),bg="#0097e6",command=clear)
button.place(x=270,y=540)

# Define the result label
result_label =Label(frame1, text="")
result_label.pack()

root.mainloop()
