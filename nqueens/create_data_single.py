import numpy as np
import torch
import os
import random

random.seed(0)

def place_queen(n, pre):
    if len(pre) == n:
        return [pre]
    else:
        possible = []
        all_ans = []
        for col in range(n):
            success_flag = True
            for pre_row, pre_col in enumerate(pre):
                if pre_col == col:
                    success_flag = False
                    break
                row = len(pre)
                if abs(pre_col - col) == row - pre_row:
                    success_flag = False
                    break
            if success_flag:
                possible.append(col)
        for one in possible:
            new_pre = pre.copy()
            new_pre.append(one)
            all_ans.extend(plance_queen(n, new_pre))
        return all_ans

def to_boad(ans):
    n = len(ans)
    board = np.zeros(shape=(n,n), dtype=np.int)
    for i, x in enumerate(ans):
        board[i, x] = 1
    return board

def get_queen_col_from_fixed_row(board, row):
    return np.argmax(board[row])

def get_diagnal(n, row, col):
    direction = [(1,1), (1,-1), (-1,1), (-1,-1)]
    for delta_x, delta_y in direction:
        new_x, new_y = row, col
        while 0 < new_x and new_x < n - 1 and 0 < new_y and new_y < n - 1:
            new_x += delta_x
            new_y += delta_y
            yield new_x, new_y

def to_mask(ans, fixed, n):
    board = np.zeros(shape=(n,n), dtype=np.int)
    for row in fixed:
        col = get_queen_col_from_fixed_row(ans, row)
        # board[row, col] = 1
        board[row, :] = 1
        board[:, col] = 1
        for r, c in get_diagnal(n, row, col):
            board[r, c] = 1
    return board

def get_single_problem(example, one_ans, pre_fixed_bits):
    flexible_bits = {i for i in range(len(example[0]))}
    flexible_bits = flexible_bits.difference(set(pre_fixed_bits))
    flexible_bits = list(flexible_bits)
    bit = random.choice(flexible_bits)
    one_appeared = one_ans[bit]
    satisfied = list(filter(lambda x: x[bit] == one_appeared, example))
    if len(satisfied) > 1:
        new_fixed_bits = pre_fixed_bits + [bit]
        return get_single_problem(satisfied, one_ans, new_fixed_bits)
    else:
        return set(pre_fixed_bits + [bit])

for n in range(12, 13):
    ans = plance_queen(n, [])
    if len(ans) > 0:
        n_example = len(ans)
        print(f"Board Size: {n}, Num Solution: {len(ans)}, Num Blur: {n_example}")

        dir = str(n)
        if dir not in os.listdir(os.getcwd()):
            os.makedirs(dir, exist_ok=True)

        X = torch.zeros(size=(n_example, n, n), dtype=torch.float32)
        is_input = torch.zeros(size=(n_example, n, n), dtype=torch.int32)

        for i, one_ans in enumerate(ans):
            one_fixed = get_single_problem(ans, one_ans, [])
            board = to_boad(one_ans)
            X[i] = torch.from_numpy(board)
            is_input[i] = torch.from_numpy(to_mask(board, one_fixed, n))
        torch.save(X, dir + "/features.pt")
        torch.save(is_input, dir + "/is_input.pt")
