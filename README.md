# Handwriting_Authentication_SNN
 
I have explained the complete project and related theory in my article https://mahesh-chat.medium.com/signature-verification-system-using-siamese-neural-network-e0b3d0200596
Signatures continue to play an important role in financial, commercial and legal transaction for account openings, withdrawals and transaction payments. In contractual matters, signatures also play a vital role.On the other hand, the threats and monetary losses continue to rise dramatically.

Problem statement :-
A student at Emily Carr University of Art and Design lost $600 from his bank account through fraudulent cheques. In this case, the most surprising factor was that the money got deducted through half a dozen cheques worth $100 each, and none of them had matching signatures. Of course, we should blame the bank staff for authorizing payments without validating the signatures accurately. But if you are a bank employee who has to deal with hundreds of cheque clearances per day, it could be a tedious task to perform due diligence with greater precision.
https://www.cbc.ca/news/canada/british-columbia/td-bank-refuses-to-refund-art-student-600-in-fraudulent-cheques-1.5278144


Solution :-
Deep learning can be used to make a signature verification system using CNN(convolutional neural network) . But the problem with the CNN it requires lots of data for training . In our case we require lots of signature sample of each account holder which is practically hard to achieve.
For this we have one shot learning which can be implemented using siamese neural network. It requires only one image of signature .This signature will be stored in the database. Sample image will be compared with all the images in the database . if none of the image is matching then manualchecking can be done. in case of matching , account holder is identified .





Dataset :-
Dataset has two folders :- one folder contains the forged signatures while another folder contains the real signatures .
In real signature folder , there are 5 signatures from the same person.while forge signature folder contains 5 signatures from different persons.
The dataset is mainly used for the testing purpose as we are using pre trained model .




Files present in the project :-
Main folder :- it contains the main.py file and the the actual simese network code .
Pre-trained_model :- It contains two 4 files
a) model.meta :- It contains meta graph(computational graph) which stores the all the computations of the graph.
b)model .index and model.data :- also called as checkpoint files. these files contain the values of the parameters after training the model.
c)checkpoint :- it simply keeps a record of latest checkpoint files saved.
Dataset:- contains signature images for testing .





.Results:-
Some results i got i am attaching here.as i have checked already , Model is showing very good results. Model is able to recognise the signatures from the same person or we can say genuine signatures and forged signatures . I have used Tkinter library for GUI application as it is fast and easy to use. For more detailed working and results please visit my github profile and run on it on your local machine.




![Screenshot 2021-05-15 014711](https://user-images.githubusercontent.com/46081668/119404887-18e04c80-bcfe-11eb-96e7-d03dcb165574.png)
![Screenshot 2021-05-15 014922](https://user-images.githubusercontent.com/46081668/119404902-1da50080-bcfe-11eb-84a1-171b91e7fd17.png)
![Screenshot 2021-05-15 015009](https://user-images.githubusercontent.com/46081668/119404909-20075a80-bcfe-11eb-9fb6-d2c1b1c43059.png)
