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
    def __init__(self, n_state, n_actions, aux, m):
        super(SudokuSolver, self).__init__()
        self.sat = satnet.SATNet(n_state+n_actions, m, aux, max_iter=100, eps=1e-6)

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
    parser.add_argument('--data_dir', type=str, default='sudoku')
    parser.add_argument('--difficulty', type=str, default='')
    parser.add_argument('--boardSz', type=int, default=3)
    parser.add_argument('--batchSz', type=int, default=200)
    parser.add_argument('--testBatchSz', type=int, default=800)
    parser.add_argument('--aux', type=int, default=100)
    parser.add_argument('--m', type=int, default=400)
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
            args.data_dir, '_action', args.aux, args.m, args.lr, args.batchSz)
    if args.save:
        save = '{}-{}'.format(args.save, save)
    save = os.path.join('logs', save)
    # if os.path.isdir(save):
    #     shutil.rmtree(save)
    os.makedirs(save, exist_ok=True)

    print_header('Loading data')

    with open(os.path.join(args.data_dir, args.difficulty + '_st.pt'), 'rb') as f:
        X_in = torch.load(f)
    with open(os.path.join(args.data_dir, args.difficulty + '_at.pt'), 'rb') as f:
        Y_in = torch.load(f)
    with open(os.path.join(args.data_dir, args.difficulty + '_scope.pt'), 'rb') as f:
        Scope = torch.load(f)


    N = Scope.size()[0]
    nTrain = int(N*(1.-args.testPct))
    nTest = N-nTrain
    nTrain = Scope[nTrain-1]
    # assert(nTrain % args.batchSz == 0)
    # assert(nTest % args.testBatchSz == 0)

    print_header('Forming inputs')
    X, Y, is_input = process_inputs(X_in, Y_in)
    data = X
    if args.cuda:
        data, is_input, Y = data.cuda(), is_input.cuda(), Y.cuda()

    train_set = TensorDataset(data[:nTrain], is_input[:nTrain], Y[:nTrain])
    test_set = TensorDataset(data[nTrain:], is_input[nTrain:], Y[nTrain:])
    print(len(train_set), len(test_set))

    print_header('Building model')
    n_states = X_in.size()[1]
    n_actions = Y_in.size()[1]
    model = SudokuSolver(n_states, n_actions, args.aux, args.m)

    if args.cuda:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # args.model = f"{save}/it41.pth"
    if args.model:
        print(f"{args.model} loaded")
        model.load_state_dict(torch.load(args.model))

    train_logger = CSVLogger(os.path.join(save, 'train.csv'))
    test_logger = CSVLogger(os.path.join(save, 'test.csv'))
    fileds = ['epoch', 'loss', 'err']
    train_logger.log(fileds)
    test_logger.log(fileds)

    # test(args.boardSz, 0, model, optimizer, test_logger, test_set, args.testBatchSz)
    for epoch in range(1, args.nEpoch+1):
        train(args.boardSz, epoch, model, optimizer, train_logger, train_set, args.batchSz)
        test(args.boardSz, epoch, model, optimizer, test_logger, test_set, args.testBatchSz)
        torch.save(model.state_dict(), os.path.join(save, 'it'+str(epoch)+'.pth'))

def process_inputs(X, Y):
    N, states = X.size()
    _, actions = Y.size()
    is_input = torch.cat([torch.ones(size=(N, states)), torch.zeros(size=(N, actions))], dim=1).int()
    Y = torch.cat([X, Y], dim=1)
    X = torch.cat([X, torch.zeros(size=(N, actions))], dim=1)
    return X, Y, is_input

def run(boardSz, epoch, model, optimizer, logger, dataset, batchSz, to_train=False):
    loss_final = 0
    err_final = 0

    loader = DataLoader(dataset, batch_size=batchSz, shuffle=True)
    tloader = tqdm(loader, total=len(loader))

    for i,(data,is_input,label) in enumerate(tloader):
        if to_train:
            optimizer.zero_grad()

        preds = model(data.contiguous(), is_input.contiguous())
        loss = nn.functional.binary_cross_entropy(preds, label)
        loss_final += loss.item() * data.size()[0]

        if to_train:
            loss.backward()
            optimizer.step()

        err = computeErr(preds, label)
        err_final += err * data.size()[0]
        tloader.set_description('Epoch {} {} Loss {:.4f}, Err: {:.4f}'.format(epoch, ('Train' if to_train else 'Test '), loss.item(), err))

    loss_final = loss_final / len(dataset)
    err_final = err_final / len(dataset)
    logger.log((epoch, loss_final, err_final))

    #print('memory: {:.2f} MB, cached: {:.2f} MB'.format(torch.cuda.memory_allocated()/2.**20, torch.cuda.memory_cached()/2.**20))
    torch.cuda.empty_cache()

def train(args, epoch, model, optimizer, logger, dataset, batchSz):
    run(args, epoch, model, optimizer, logger, dataset, batchSz, True)

@torch.no_grad()
def test(args, epoch, model, optimizer, logger, dataset, batchSz):
    run(args, epoch, model, optimizer, logger, dataset, batchSz, False)

@torch.no_grad()
def computeErr(preds, label):
    batchSz = preds.size()[0]
    err = 0
    for one_pred, one_ans in zip(preds.round(), label):
        # print(one_pred, one_ans)
        if not torch.equal(one_pred, one_ans):
            err += 1
    return err / batchSz

def to_soduku_batch(X):
    X = X.reshape(-1, 10, 9, 9)
    X = torch.argmax(X, dim=1)
    return X

if __name__=='__main__':
    main()