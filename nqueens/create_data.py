import numpy as np
import torch
import os

def plance_queen(n, pre):
    if len(pre) == n:
        return [pre]
    else:
        possible = []
        all_ans = []
        for row in range(n):
            success_flag = True
            for pre_col, pre_row in enumerate(pre):
                if pre_row == row:
                    success_flag = False
                    break
                col = len(pre)
                if abs(pre_row - row) == col - pre_col:
                    success_flag = False
                    break
            if success_flag:
                possible.append(row)
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

def to_mask(ans, n):
    board = np.zeros(shape=(n,n), dtype=np.int)
    for row in ans:
        board[row] = 1
    return board

def get_problem(example, pre_fixed_bits):
    example_blur = [set() for _ in range(len(example))]
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
                    satisfied_blur = get_problem(satisfied, new_fixed_bits)
                    example_blur[ori_idx] = example_blur[ori_idx].union(satisfied_blur[i])
            else:
                idx = example.index(satisfied[0])
                example_blur[idx].add(tuple(pre_fixed_bits + [bit]))
    return example_blur


for n in range(4, 9):
    ans = plance_queen(n, [])
    if len(ans) > 0:
        blur = get_problem(ans, [])
        n_example = sum([len(item) for item in blur])
        print(f"Board Size: {n}, Num Solution: {len(ans)}, Num Blur: {n_example}")

        dir = str(n)
        if dir not in os.listdir(os.getcwd()):
            os.makedirs(dir, exist_ok=True)

        X = torch.zeros(size=(n_example, n, n), dtype=torch.float32)
        is_input = torch.zeros(size=(n_example, n, n), dtype=torch.int32)
        cnt = 0
        for one_ans, one_ans_blur in zip(ans, blur):
            print(one_ans, len(one_ans_blur))
            X[cnt:cnt+len(one_ans)] = torch.from_numpy(to_boad(one_ans))
            for one_blur in one_ans_blur:
                is_input[cnt] = torch.from_numpy(to_mask(one_blur, n))
                cnt += 1
        assert cnt == n_example
        torch.save(X, dir + "/features.pt")
        torch.save(is_input, dir + "/is_input.pt")
