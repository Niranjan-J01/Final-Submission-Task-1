import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch as ts 
import torch.nn as nn
import pandas as pd 
import numpy as np 

device = ts.device("cuda" if ts.cuda.is_available() else "cpu")

# transforming the images to tensor and normalizing it 
from torchvision import transforms

trans = transforms.Compose([
    transforms.ToTensor(), # image to tensor to scale(0,1)
    transforms.Normalize((0.5,),(0.5,)) # set of values for normalizing( -1 , 1)
])

# importing dataset 
from torchvision.datasets import FashionMNIST

data_train = FashionMNIST( root = "./data_Fashsion", train=True ,download=True , transform=trans)
data_test = FashionMNIST(root = "./data_Fashsion" , train=False , download=True , transform=trans )

# data splittinng for train and test and LOADING]
from torch.utils.data import DataLoader , random_split

train_size = int(0.8*len(data_train))
val_size = len(data_train) - train_size

train_dataset , val_dataset = random_split( data_train , [train_size , val_size])

# loading

x_train = DataLoader(train_dataset , batch_size=64 , shuffle=True )
x_val = DataLoader(val_dataset , batch_size=64  , shuffle=False )
x_test = DataLoader(data_test , batch_size=512 , shuffle=False )

# model 

class fashion(nn.Module):

    def __init__(self):
        super(fashion , self).__init__()

        # flattening
        self.flatten = nn.Flatten()
 
        # hidden layer
        self.hiddenlayer1 = nn.Linear(784 , 16)
        
        #cleft branches
        self.Leftlayer1=nn.Linear(16 ,8)
        self.Leftlayer2=nn.Linear(8 ,8)
        
        # right branches
        self.rightlayer1=nn.Linear(16,12)
        self.rightlayer2=nn.Linear (12, 8)

        #activation
        self.relu = nn.ReLU()

        #output
        self.output = nn.Linear(16 , 10)   

    def forward(self , x):

        # flattening
        x = self.flatten(x) # (28*28 , 784)

        # hiddenlayer1
        h1 = self.relu(self.hiddenlayer1(x)) # (784 , 16)

        # LEFT LAYERS 
        Lh1 = self.relu(self.Leftlayer1(h1)) # (16 ,8)
        Lh2 = self.relu(self.Leftlayer2(Lh1)) # (8 ,8)
        skip= Lh1 + Lh2 # (8 ,8)

        # Right layers
        Rh1 = self.relu(self.rightlayer1(h1)) # (16 , 12) 
        Rh2 = self.relu(self.rightlayer2(Rh1)) #(12 ,8)

        # concantenate
        con = ts.cat([skip , Rh2] , dim=1) # (8*8 , 16) ; [batch , 16]

        out = self.output(con)        

        return out
    
model = fashion().to(device)

creatrian = nn.CrossEntropyLoss()
optimmizer = ts.optim.Adam(model.parameters())

train_loss = []
vall_loss =[]

# for saving&loadinng the model
import pickle
best_model = float("inf")

for epoch in range(41):

    # trainning

    model.train()
    running_loss = 0.00
    
    for image , label in x_train:

        image , label = image.to(device) , label.to(device)

        optimmizer.zero_grad()
        
        pred = model(image)
        loss = creatrian(pred , label)
        loss.backward()
        optimmizer.step()
        running_loss += loss.item()

    epoch_train_loss = running_loss/len(x_train)
    train_loss.append(epoch_train_loss)

    # val

    model.eval()
    with ts.no_grad():
        running_val_loss = 0.00
        for image , label in x_val:

            image , label = image.to(device) , label.to(device)

            pred=model(image)
            loss = creatrian(pred , label)
            running_val_loss += loss.item()
        
        epoch_val_loss = running_val_loss/len(x_val)
        vall_loss.append(epoch_val_loss)
        if epoch_val_loss < best_model :
            best_model = epoch_val_loss
            with open("best_model.pkl" , "wb") as f:
                pickle.dump(model.state_dict() , f) # savinng the best weights

    print(f"S.NO:{epoch+1} epoch_trainloss  = {epoch_train_loss} $ epoch_val_loss = {epoch_val_loss}")        

# graph 
import matplotlib.pyplot as plt 

loss_data = pd.DataFrame({
    "Trainning Loss" : train_loss,
    "Validation Loss" : vall_loss
})

plt.plot(loss_data["Trainning Loss"] , label = "Training loss")
plt.plot(loss_data["Validation Loss"] , label = "Validation Loss")

plt.xlabel("Epochs")
plt.ylabel("Losses")

plt.legend()
plt.show()

# loading the model

with open("best_model.pkl" , "rb") as f:
    state = pickle.load(f)
model.load_state_dict(state)

#  creating test file
prediction = []
correct = 0
total = 0
with ts.no_grad():
    for image , label in x_test:
        image  , label = image.to(device) , label.to(device)
        outputs = model(image)
        _ , preds = ts.max(outputs , 1)
        correct += (preds==label).sum().item()
        total += label.size(0)
        prediction.extend(preds.cpu().numpy())
print(f"accuraacy = {correct*100/total} % ")

submission  = pd.DataFrame({
    "image ID " : list(range(1 , len(prediction)+1)) , 
    "Labels" : prediction
})        
submission.to_csv("submission.csv" , index=False)



