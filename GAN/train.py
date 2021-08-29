import time
from torch.autograd import Variable
from torch import nn
import torch
import numpy as np
from doseloss import DoseLoss

# models, optims should be dicts with keys 'D' and 'G' for the
# objects correspond to the generator and discriminator
def train_GAN(dataloaders, discriminator, generator, loss_fn, optim_D, optim_G,
              scheduler_D, scheduler_G, patch=35, pixelwise_loss=True, lambda_pixelwise = 1.0, 
              n_epochs=50, verbose=True, print_each=1):
    
    device = torch.device('cuda') if torch.cuda.is_available else 'cpu'
    num_patches = patch**3
    
    begin = time.time()
    acc_dict_true = {'train': [], 'val': []}
    acc_dict_fake = {'train': [], 'val': []}
    loss_dict_D = {'train': [], 'val': []}
    loss_dict_G = {'train': [], 'val': []}
    mae_dict = {'train': [], 'val': []}
    best_params = {}
    best_loss = 100. 
    best_epoch = 0
    
    
    if pixelwise_loss:
        doseloss = DoseLoss().to(device)
    
    for i in range(n_epochs):
        epoch_start = time.time()
        for p in ['train','val']:
            if p == 'train':
                discriminator.train()
                generator.train()
            else:
                discriminator.eval()
                generator.eval()
                
            running_loss_D = 0.0
            running_loss_G = 0.0
            running_dose_loss = 0.0
            running_correct_true = 0
            running_correct_fake = 0
            batch_size = dataloaders[p].batch_size
            len_data = dataloaders[p].dataset.__len__()
            
            labels_true = Variable(torch.Tensor(
                np.ones((batch_size, 1, patch, patch, patch))), requires_grad=False).to(device)
            labels_fake = Variable(torch.Tensor(
                np.zeros((batch_size, 1, patch, patch, patch))), requires_grad=False).to(device)
            
            for batch in dataloaders[p]:
                true_A, true_B, mask = batch
                true_A = true_A.to(device)
                true_B = true_B.to(device)
                mask = mask.to(device)
                
                with torch.set_grad_enabled(p == 'train'):
                    
                    fake_B = generator(true_A)

                    # Train the discriminator
                    # Discriminator output is 19x19x19
                    # Should I just flatten discriminator output to calc loss,
                    # or does that mess with gradients?
                    
                    optim_D.zero_grad()
                    output_true = discriminator(true_B)
                    loss_true = loss_fn(output_true, labels_true)

                    # .detach() generator output to avoid accumulating gradients
                    output_fake = discriminator(fake_B.detach())
                    loss_fake = loss_fn(output_fake, labels_fake)

                    loss_D = 0.5 * (loss_true + loss_fake)
                    
                    if p == 'train':
                        loss_D.backward()
                        optim_D.step()
                        
                        
                    # Train GAN
                    # It should be fine that we generated the samples above right?
                    # Since we zero_grad here? Check this
                    optim_G.zero_grad()
                    output_fake = discriminator(fake_B)
                    # We train the generator to make data that looks true
                    # Hence we use labels_true and preds_fake: we want preds on fake data
                    # to match labels_true
                    loss_G = loss_fn(output_fake, labels_true)
                    pixelwise_loss_G = doseloss(fake_B, true_B, mask)
                    total_loss_G = loss_G + lambda_pixelwise*pixelwise_loss_G
                        
                    if p == 'train':
                        total_loss_G.backward()
                        optim_G.step()
                        
                
                # Logits > 0 get mapped to 1, < 0 to 0
                running_loss_D += loss_D.item()*batch_size
                running_loss_G += loss_G.item()*batch_size
                running_dose_loss += pixelwise_loss_G.item()*batch_size
                running_correct_true += torch.sum((output_true > 0) == labels_true).item()
                running_correct_fake += torch.sum((output_fake > 0) == labels_fake).item()
                
        
            acc_dict_true[p].append(running_correct_true/(len_data*num_patches))
            acc_dict_fake[p].append(running_correct_fake/(len_data*num_patches))
            loss_dict_D[p].append(running_loss_D/len_data)
            loss_dict_G[p].append(running_loss_G/len_data)
            mae_dict[p].append(running_dose_loss/len_data)
            
            
            
            # We want the disciminator to be unable to tell what the truth is
            # Or should I use the dose score? That's what we want after all
        if mae_dict['val'][i] < best_loss:
            best_loss = mae_dict['val'][i]
            best_params_D = discriminator.state_dict()
            best_params_G = generator.state_dict()
            best_epoch = i

        
        scheduler_D.step()
        scheduler_G.step()
        
        epoch_end = time.time()
        if (verbose and i%print_each == 0) or i+1 == num_epochs:
            print(f"Epoch {i+1:2d}/{n_epochs}\n"
                  f"Train loss: D {loss_dict_D['train'][i]:.4f}, "
                  f"G {loss_dict_G['train'][i]:.4f}\n"
                  f"Val loss  : D {loss_dict_D['val'][i]:.4f}, "
                  f"G {loss_dict_G['val'][i]:.4f}\n"
                  f"Train acc: Real {acc_dict_true['train'][i]:.4f}, "
                  f"Fake {acc_dict_fake['train'][i]:.4f}\n"
                  f"Val acc  : Real {acc_dict_true['val'][i]:.4f}, "
                  f"Fake {acc_dict_fake['val'][i]:.4f}\n"
                  f"Dose score: Train {mae_dict['train'][i]:.4f}, Val {mae_dict['val'][i]:.4f}\n"
                  f"Epoch time = {epoch_end-epoch_start:.0f}s\n"
                  f"Elapsed time = {epoch_end-begin:.0f}s\n"+
                  ("-"*20)+"\n")
        
    print(f"Best epoch: {best_epoch+1:2d}/{n_epochs}\n"
          f"Train loss: D {loss_dict_D['train'][best_epoch]:.4f}, "
          f"G {loss_dict_G['train'][best_epoch]:.4f}\n"
          f"Val loss  : D {loss_dict_D['val'][best_epoch]:.4f}, "
          f"G {loss_dict_G['val'][best_epoch]:.4f}\n"
          f"Train acc: Real {acc_dict_true['train'][best_epoch]:.4f}, "
          f"Fake {acc_dict_fake['train'][best_epoch]:.4f}\n"
          f"Val acc  : Real {acc_dict_true['val'][best_epoch]:.4f}, "
          f"Fake {acc_dict_fake['val'][best_epoch]:.4f}\n"
          f"Dose score: Train {mae_dict['train'][best_epoch]:.4f}, "
          f"Val {mae_dict['val'][best_epoch]:.4f}\n"
          f"Epoch time = {epoch_end-epoch_start:.0f}s\n"
          f"Elapsed time = {epoch_end-begin:.0f}s\n"+
          ("-"*20)+"\n")
    
    discriminator.load_state_dict(best_params_D)
    generator.load_state_dict(best_params_G)
    return discriminator, generator, acc_dict_true, acc_dict_fake, loss_dict_D, loss_dict_D, mae_dict, best_epoch