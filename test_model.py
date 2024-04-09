import os
import cv2
import numpy as np
import tkinter as tk
import tensorflow.compat.v1 as tf

from tkinter import filedialog
from tkinter import messagebox

def submit_username():
	global username_entry, username
	username = username_entry.get()
	root.destroy()

# Create the main window
root = tk.Tk()
root.title("Username Input")

# Create a label and entry widget for username input
username_label = tk.Label(root, text="Enter Username:")
username_label.pack()
username_entry = tk.Entry(root)
username_entry.pack()

# Create a submit button
submit_button = tk.Button(root, text="Submit", command=submit_username)
submit_button.pack()

# Run the Tkinter event loop
root.mainloop()

input_image_size=200
input_image_size=200

print("**************** Loading The Siamese Model ****************\n")

tf.disable_v2_behavior()
tf.reset_default_graph()
sess = tf.Session()
tf.disable_eager_execution()
saver = tf.compat.v1.train.import_meta_graph('./model_params/model.meta')
saver.restore(sess,'./model_params/model')
graph = tf.compat.v1.get_default_graph()
left_input = graph.get_tensor_by_name("left_input:0")
right_input = graph.get_tensor_by_name("right_input:0")
output = graph.get_tensor_by_name("output:0")
is_training = graph.get_tensor_by_name("is_training:0")
prob = graph.get_tensor_by_name("prob:0")

print("**************** Succesfully Loaded The Model ****************\n")

root = tk.Tk()
messagebox.showinfo(
    "Information",
    "The program will prompt you to select two signature/handwriting images that are to be compared and will display if they match or not"
)
root.destroy()

root = tk.Tk() # it will create top level or application window
# answer1 = "./Test_network_data/forged_signatures/arpan/Signature.jpeg"
answer1 = filedialog.askopenfilename(
    parent=root,
    initialdir=os.getcwd(),
    title="Please select a genuine signature image file"
)
answer2 = filedialog.askopenfilename(
    parent=root,
    initialdir=os.getcwd(),
    title="Please select a signature image that is to be tested"
)
root.destroy()

try:
	sig1_img = cv2.imread(answer1, 0)
	resized1 = np.array(cv2.resize(sig1_img, (input_image_size, input_image_size), interpolation=cv2.INTER_CUBIC)).reshape(-1,40000)
	resized1 = resized1 / 255
	sig2_img = cv2.imread(answer2, 0)
	resized2 = np.array(cv2.resize(sig2_img, (input_image_size, input_image_size), interpolation=cv2.INTER_CUBIC)).reshape(-1,40000)
	resized2 = resized2 / 255
	out = sess.run(output, feed_dict={left_input: resized1, right_input: resized2, is_training: False})
	out = sess.run(tf.compat.v1.nn.softmax(out))
	label = ''
	out = np.argmax(out)

	if out == 1:
		label = False
	else:
		label = True

	if label == True:
		root = tk.Tk()
		messagebox.showinfo("Result", f"The signature of {username} matched")
		root.destroy()
	else:
		root = tk.Tk()
		messagebox.showwarning("Result", f"The signature of {username} didn't match")
		root.destroy()
except:
    print("Something went wrong, please try again")
