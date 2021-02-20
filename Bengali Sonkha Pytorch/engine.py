import matplotlib.pyplot as plt 
import torch  





def train(train_data, model, optimizer, device, criterion, epoch):
    train_losses = []
    train_loss = 0

    model.train()
    for data, target in train_data:
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)

    # calculate-average-losses
    train_loss = train_loss/len(train_data.sampler)
    train_losses.append(train_loss)

    # print-training/validation-statistics 
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch+1, train_loss))
    return train_losses



def valid(valid_data, model, criterion, device, epoch, early_stopping):
    valid_losses = []
    valid_loss = 0

    model.eval()
    for data, target in valid_data: 
        data = data.to(device)
        target = target.to(device)

        output = model(data)
        loss = criterion(output, target)

        valid_loss += loss.item() * data.size(0)
        
    # applying early stopping    
    early_stopping(valid_loss, model)
        
     if early_stopping.early_stop:
            print("Early stopping")
            break

    # calculate-average-losses
    valid_loss = valid_loss/len(valid_data.sampler)
    valid_losses.append(valid_loss)

    # print-training/validation-statistics 
    print('Epoch: {} \tValidation Loss: {:.6f}'.format(epoch+1, valid_loss))
    return valid_losses
    
