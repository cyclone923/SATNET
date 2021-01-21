import numpy as np
import torch
import os

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

def to_mask(ans, fixed, n):
    board = np.zeros(shape=(n,n), dtype=np.int)
    for row in fixed:
        col = get_queen_col_from_fixed_row(ans, row)
        board[row, col] = 1
    return board

def get_problem(example, pre_fixed_bits):
    example_fixed = [set() for _ in range(len(example))]
    flexible_bits = {i for i in range(len(example[0]))}
    flexible_bits = flexible_bits.difference(set(pre_fixed_bits))
    for bit in flexible_bits:
        all_appeared = set((map(lambda x: x[bit], example)))

        for i, one_appeared in enumerate(all_appeared):
            satisfied = list(filter(lambda x: x[bit] == one_appeared, example))
            if len(satisfied) > 1:
                new_fixed_bits = pre_fixed_bits + [bit]
                for i, one_satisfied in enumerate(satisfied):
                    ori_idx = example.index(one_satisfied)
                    satisfied_fixed = get_problem(satisfied, new_fixed_bits)
                    example_fixed[ori_idx] = example_fixed[ori_idx].union(satisfied_fixed[i])
            else:
                idx = example.index(satisfied[0])
                example_fixed[idx].add(tuple(pre_fixed_bits + [bit]))
    return example_fixed



for n in range(4, 9):
    ans = plance_queen(n, [])
    if len(ans) > 0:
        fixed = get_problem(ans, [])
        n_example = sum([len(item) for item in fixed])
        print(f"Board Size: {n}, Num Solution: {len(ans)}, Num Blur: {n_example}")

        dir = str(n)
        if dir not in os.listdir(os.getcwd()):
            os.makedirs(dir, exist_ok=True)

        X = torch.zeros(size=(n_example, n, n), dtype=torch.float32)
        is_input = torch.zeros(size=(n_example, n, n), dtype=torch.int32)
        cnt = 0
        for one_ans, one_ans_fixed in zip(ans, fixed):
            print(one_ans, len(one_ans_fixed))
            board = to_boad(one_ans)
            X[cnt : cnt+len(one_ans_fixed)] = torch.from_numpy(board)
            # print(X[cnt])
            for one_fixed in one_ans_fixed:
                is_input[cnt] = torch.from_numpy(to_mask(board, one_fixed, n))
                # print(is_input[cnt])
                # print(one_fixed)
                cnt += 1
        assert cnt == n_example
        torch.save(X, dir + "/features.pt")
        torch.save(is_input, dir + "/is_input.pt")

