'''
Train a transformer-based model for emotion classification
For suppression of the largest singular value of projection matrices spectral normalization is carried out
'''

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from transformers import AdamW
from data_prep import get_train, get_val, get_test
import sys
import os
import argparse
from tools import AverageMeter, accuracy_topk, get_default_device
from models import ElectraSequenceClassifier, BertSequenceClassifier, RobertaSequenceClassifier


def apply_spectral_norm(model, num_layers=12, num_heads=12, hidden_size=768):
    '''
    Spectral Normalisation of Projection Matices
    '''
    projection_matrices = ['query', 'key', 'value']

    old_params = {}
    for name, params in model.named_parameters():
        old_params[name] = params.clone()

    for layer in range(num_layers):
        for proj in projection_matrices:
            param_name = f'electra.encoder.layer.{layer}.attention.self.{proj}.weight'
            mat = old_params[param_name]
            
            normed_head_mats = []
            for head in range(num_heads):
                chunk_size = hidden_size/num_heads
                start = int(head*chunk_size)
                end = int((head+1)*chunk_size)
                mat_head = mat[:,start:end]
                # normed_mat_head = nn.utils.parametrizations.spectral_norm(mat_head) - for newer pytorch
                normed_mat_head = nn.utils.spectral_norm(mat_head)
                normed_head_mats.append(normed_mat_head)
            new_mat = torch.cat(normed_head_mats)
            old_params[param_name] = new_mat
            
    for name, params in model.named_parameters():
        params.data.copy_(old_params[name])


def train(train_loader, model, criterion, optimizer, epoch, device, print_freq=25):
    '''
    Run one train epoch
    '''
    losses = AverageMeter()
    accs = AverageMeter()

    # switch to train mode
    model.train()

    for i, (id, mask, target) in enumerate(train_loader):

        id = id.to(device)
        mask = mask.to(device)
        target = target.to(device)

        # Forward pass
        logits = model(id, mask)
        loss = criterion(logits, target)

        # Backward pass and update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Spectral norm regularization of projection matrices
        apply_spectral_norm(model)

        # measure accuracy and record loss
        acc = accuracy_topk(logits.data, target)
        accs.update(acc.item(), id.size(0))
        losses.update(loss.item(), id.size(0))

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Accuracy {prec.val:.3f} ({prec.avg:.3f})'.format(
                      epoch, i, len(train_loader),
                      loss=losses, prec=accs))

def eval(val_loader, model, criterion, device):
    '''
    Run evaluation
    '''
    losses = AverageMeter()
    accs = AverageMeter()

    # switch to eval mode
    model.eval()

    with torch.no_grad():
        for i, (id, mask, target) in enumerate(val_loader):

            id = id.to(device)
            mask = mask.to(device)
            target = target.to(device)

            # Forward pass
            logits = model(id, mask)
            loss = criterion(logits, target)

            # measure accuracy and record loss
            acc = accuracy_topk(logits.data, target)
            accs.update(acc.item(), id.size(0))
            losses.update(loss.item(), id.size(0))

    print('Test\t Loss ({loss.avg:.4f})\t'
            'Accuracy ({prec.avg:.3f})\n'.format(
              loss=losses, prec=accs))


if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('OUT', type=str, help='Specify output th file')
    commandLineParser.add_argument('ARCH', type=str, help='electra, bert, roberta')
    commandLineParser.add_argument('--B', type=int, default=16, help="Specify batch size")
    commandLineParser.add_argument('--epochs', type=int, default=2, help="Specify epochs")
    commandLineParser.add_argument('--lr', type=float, default=0.000001, help="Specify learning rate")
    commandLineParser.add_argument('--sch', type=int, default=10, help="Specify scheduler rate")
    commandLineParser.add_argument('--seed', type=int, default=1, help="Specify seed")

    args = commandLineParser.parse_args()
    out_file = args.OUT
    arch = args.ARCH
    batch_size = args.B
    epochs = args.epochs
    lr = args.lr
    sch = args.sch
    seed = args.seed

    torch.manual_seed(seed)

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train_spectral_norm.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Get the device
    device = get_default_device()

    # Load the data as tensors
    input_ids_train, mask_train, labels_train = get_train(arch)
    input_ids_val, mask_val, labels_val = get_val(arch)
    input_ids_test, mask_test, labels_test = get_test(arch)

    # Use dataloader to handle batches
    train_ds = TensorDataset(input_ids_train, mask_train, labels_train)
    val_ds = TensorDataset(input_ids_val, mask_val, labels_val)
    test_ds = TensorDataset(input_ids_test, mask_test, labels_test)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)
    test_dl = DataLoader(test_ds, batch_size=batch_size)

    # Initialise classifier
    if arch == 'electra':
        model = ElectraSequenceClassifier()
    elif arch == 'bert':
        model = BertSequenceClassifier()
    elif arch == 'roberta':
        model = RobertaSequenceClassifier()
    else:
        raise Exception("Something has gone wrong with architecture definition.")
    model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=1e-8)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[sch])

    # Criterion
    criterion = nn.CrossEntropyLoss().to(device)

    # Train
    for epoch in range(epochs):

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_dl, model, criterion, optimizer, epoch, device)
        scheduler.step()

        # evaluate on validation set
        eval(val_dl, model, criterion, device)
    
    # evaluate on test set
    print("Test set\n")
    eval(test_dl, model, criterion, device)

    # Save the trained model
    state = model.state_dict()
    torch.save(state, out_file)