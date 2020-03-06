#!/usr/bin/env python3
#
# Partly derived from:
#   https://github.com/locuslab/optnet/blob/master/sudoku/train.py 

import argparse

import os
import shutil
import csv

import numpy as np
import numpy.random as npr
#import setproctitle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm

import satnet
import sys
from logic.logic import get_sudoku_matrix

torch.set_printoptions(linewidth=sys.maxsize)

class SudokuSolver(nn.Module):
    def __init__(self, boardSz, aux, m, S=None):
        super(SudokuSolver, self).__init__()
        n = boardSz**6
        self.sat = satnet.SATNet(n, m, aux, max_iter=100, eps=1e-6)

    def forward(self, y_in, mask):
        out = self.sat(y_in, mask)
        return out


class CSVLogger(object):
    def __init__(self, fname):
        self.f = open(fname, 'w')
        self.logger = csv.writer(self.f)

    def log(self, fields):
        self.logger.writerow(fields)
        self.f.flush()


def print_header(msg):
    print('===>', msg)

def find_unperm(perm):
    unperm = torch.zeros_like(perm)
    for i in range(perm.size(0)):
        unperm[perm[i]] = i
    return unperm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='sudoku24to30_medium')
    parser.add_argument('--boardSz', type=int, default=3)
    parser.add_argument('--batchSz', type=int, default=200)
    parser.add_argument('--testBatchSz', type=int, default=200)
    parser.add_argument('--aux', type=int, default=100)
    parser.add_argument('--m', type=int, default=200)
    parser.add_argument('--nEpoch', type=int, default=2000)
    parser.add_argument('--testPct', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--save', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--perm', action='store_true')

    args = parser.parse_args()

    # For debugging: fix the random seed
    npr.seed(1)
    torch.manual_seed(7)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        print('Using', torch.cuda.get_device_name(0))
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.init()

    save = '{}{}.-aux{}-m{}-lr{}-bsz{}'.format(
            args.data_dir, '.perm' if args.perm else '', args.aux, args.m, args.lr, args.batchSz)
    if args.save:
        save = '{}-{}'.format(args.save, save)
    save = os.path.join('logs', save)
    # if os.path.isdir(save):
    #     shutil.rmtree(save)
    os.makedirs(save, exist_ok=True)

    print_header('Loading data')

    with open(os.path.join(args.data_dir, 'features.pt'), 'rb') as f:
        X_in = torch.load(f)
    with open(os.path.join(args.data_dir, 'labels.pt'), 'rb') as f:
        Y_in = torch.load(f)
    with open(os.path.join(args.data_dir, 'perm.pt'), 'rb') as f:
        perm = torch.load(f)

    N = X_in.size(0)
    nTrain = int(N*(1.-args.testPct))
    nTest = N-nTrain
    assert(nTrain % args.batchSz == 0)
    assert(nTest % args.testBatchSz == 0)

    print_header('Forming inputs')
    X, Y, is_input = process_inputs(X_in, Y_in)
    data = X
    if args.cuda:
        data, is_input, Y = data.cuda(), is_input.cuda(), Y.cuda()

    unperm = None
    if args.perm and not args.mnist:
        print('Applying permutation')
        data[:,:], Y[:,:], is_input[:,:] = data[:,perm], Y[:,perm], is_input[:,perm]
        unperm = find_unperm(perm)

    train_set = TensorDataset(data[:nTrain], is_input[:nTrain], Y[:nTrain])
    test_set =  TensorDataset(data[nTrain:], is_input[nTrain:], Y[nTrain:])

    print_header('Building model')
    model = SudokuSolver(args.boardSz, args.aux, args.m)

    if args.cuda:
        model = model.cuda()


    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    args.model = f"logs/{args.data_dir}.-aux100-m200-lr0.0002-bsz200/it507.pth"
    if args.model:
        print(f"{args.model} loaded")
        model.load_state_dict(torch.load(args.model))

    train_logger = CSVLogger(os.path.join(save, 'train.csv'))
    test_logger = CSVLogger(os.path.join(save, 'test.csv'))
    fields = ['epoch', 'loss', 'err']
    train_logger.log(fields)
    test_logger.log(fields)

    # test(args.boardSz, 0, model, optimizer, test_logger, test_set, args.testBatchSz, unperm)
    # exit(0)
    for epoch in range(1, args.nEpoch+1):
        train(args.boardSz, epoch, model, optimizer, train_logger, train_set, args.batchSz, unperm)
        test(args.boardSz, epoch, model, optimizer, test_logger, test_set, args.testBatchSz, unperm)
        torch.save(model.state_dict(), os.path.join(save, 'it'+str(epoch)+'.pth'))

def process_inputs(X, Y):
    is_input = X.sum(dim=3, keepdim=True).expand_as(X).int().sign()
    # to_soduku(X[0], Y[0], is_input[0])
    # exit(0)

    X      = X.view(X.size(0), -1)
    Y      = Y.view(Y.size(0), -1)
    is_input = is_input.view(is_input.size(0), -1)


    return X, Y, is_input

def to_soduku(X, Y, is_input):
    assert isinstance(X, torch.Tensor)
    assert isinstance(Y, torch.Tensor)
    assert X.size() == (9,9,9)
    assert Y.size() == (9,9,9)
    soduku_X = X.argmax(dim=2) + 1
    soduku_Y = Y.argmax(dim=2) + 1
    is_input = is_input.permute(2,0,1)[0]
    print(soduku_X * is_input)
    print(soduku_Y)
    print(is_input)
    print(torch.sum(is_input))
    return

@torch.no_grad()
def recursive_inference(preds, mask, model):
    out = preds
    while True:
        confident_out = torch.mul((1 - mask), torch.abs(0.5 - out))
        confident_k = torch.topk(confident_out, k=1, dim=1)
        mask = torch.scatter(mask, dim=1, index=confident_k.indices, value=1)
        # out = torch.scatter(out, dim=1, index=confident_k.indices, src=out[:, confident_k.indices.flatten()].round())
        flag = sum(map(lambda x: torch.is_nonzero(x), (1 - mask).flatten()))
        if flag == 0:
            break
        # print(flag)
        out = model(out, mask)
    return out

def run(boardSz, epoch, model, optimizer, logger, dataset, batchSz, to_train=False, unperm=None):

    loss_final, err_final = 0, 0

    loader = DataLoader(dataset, batch_size=batchSz)
    tloader = tqdm(loader, total=len(loader))

    for i,(data,is_input,label) in enumerate(tloader):
        if to_train:
            optimizer.zero_grad()
        preds = model(data.contiguous(), is_input.contiguous())
        preds = recursive_inference(preds, is_input, model)
        preds_round = preds.round()
        print(is_input.size())
        print(torch.sum(preds_round == label, dim=1))
        print(torch.sum(is_input, dim=1))

        loss = nn.functional.binary_cross_entropy(preds, label)

        if to_train:
            loss.backward()
            optimizer.step()

        err = computeErr(preds.data, boardSz, unperm)/batchSz
        print(err)
        # exit(0)

        tloader.set_description('Epoch {} {} Loss {:.4f} Err: {:.4f}'.format(epoch, ('Train' if to_train else 'Test '), loss.item(), err))
        loss_final += loss.item()
        err_final += err

    loss_final, err_final = loss_final/len(loader), err_final/len(loader)
    logger.log((epoch, loss_final, err_final))

    if not to_train:
        print('TESTING SET RESULTS: Average loss: {:.4f} Err: {:.4f}'.format(loss_final, err_final))

    #print('memory: {:.2f} MB, cached: {:.2f} MB'.format(torch.cuda.memory_allocated()/2.**20, torch.cuda.memory_cached()/2.**20))
    torch.cuda.empty_cache()

def train(args, epoch, model, optimizer, logger, dataset, batchSz, unperm=None):
    run(args, epoch, model, optimizer, logger, dataset, batchSz, True, unperm)

@torch.no_grad()
def test(args, epoch, model, optimizer, logger, dataset, batchSz, unperm=None):
    run(args, epoch, model, optimizer, logger, dataset, batchSz, False, unperm)

@torch.no_grad()
def computeErr(pred_flat, n, unperm):
    if unperm is not None: pred_flat[:,:] = pred_flat[:,unperm]

    nsq = n ** 2
    pred = pred_flat.view(-1, nsq, nsq, nsq)

    batchSz = pred.size(0)
    s = (nsq-1)*nsq//2 # 0 + 1 + ... + n^2-1
    I = torch.max(pred, 3)[1].squeeze().view(batchSz, nsq, nsq)

    def invalidGroups(x):
        valid = (x.min(1)[0] == 0)
        valid *= (x.max(1)[0] == nsq-1)
        valid *= (x.sum(1) == s)
        return valid.logical_not()

    boardCorrect = torch.ones(batchSz).type_as(pred)
    for j in range(nsq):
        # Check the jth row and column.
        boardCorrect[invalidGroups(I[:,j,:])] = 0
        boardCorrect[invalidGroups(I[:,:,j])] = 0

        # Check the jth block.
        row, col = n*(j // n), n*(j % n)
        M = invalidGroups(I[:,row:row+n,col:col+n].contiguous().view(batchSz,-1))
        boardCorrect[M] = 0

        if boardCorrect.sum() == 0:
            return batchSz

    return float(batchSz-boardCorrect.sum())

if __name__=='__main__':
    main()