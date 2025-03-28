# Stock-Price-Prediction


## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset


## Design Steps

### Step 1:
Write your own steps

### Step 2:

### Step 3:



## Program
#### Name:SREENIVEDITAA SARAVANAN
#### Register Number:212223230213
Include your code here
```Python 
# Define RNN Model
class RNNModel(nn.Module):
  def __init__(self, input_size=1,hidden_size=64, num_layers=2, output_size=1):
    super(RNNModel, self).__init__()
    self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_size, output_size)
  def forward(self, x):
    out, _ = self.rnn(x)
    out = self.fc(out[:, -1, :])
    return out
model = RNNModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# Train the Model

criterion = nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

epochs=20
model.train()
train_losses=[]
for epoch in range(epochs):
  epoch_loss=0
  for x_batch,y_batch in train_loader:
    x_batch,y_batch=x_batch.to(device),y_batch.to(device)
    optimizer.zero_grad()
    outputs=model(x_batch)
    loss=criterion(outputs,y_batch)
    loss.backward()
    optimizer.step()
    epoch_loss+=loss.item()
  train_losses.append(epoch_loss/len(train_loader))
  print(f"Epoch [{epoch+1}/{epochs}] Loss: {train_losses[-1]:.4f}")
```

## Output

### True Stock Price, Predicted Stock Price vs time

![{A9BA6717-E67C-4630-B609-D3B8016F6AB7}](https://github.com/user-attachments/assets/a625238b-534f-4de6-94cf-04de17d40d81)


### Predictions 

![{066A6846-0684-498F-84FC-18B2FA3639B0}](https://github.com/user-attachments/assets/5b8fd922-2979-4dfc-9155-3ad5f4ae67bf)


## Result
Thus , a Recurrent Neural Network model for stock price prediction has successfully been devoloped.

