import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class Sequence:
    def __init__(self, length):
        self.length = length
        self.optional_starts = None

    def __len__(self):
        return self.length


class Array:
    @classmethod
    def load_array(cls, idx, bit_array):
        sequences = []
        sequence_length = 0
        for bit in bit_array:
            if bit == 0:
                sequence_length += 1
            elif sequence_length:
                sequences.append(Sequence(sequence_length))
                sequence_length = 0
        if sequence_length:
            sequences.append(Sequence(sequence_length))
        return cls(idx, len(bit_array), sequences)

    def __init__(self, idx, length, sequences):
        self.idx = idx
        self.length = length
        self.sequences = sequences
        self.constraints = np.ones(length) * np.nan
        self.n_combinations = None
        self.init_sequence_optional_starts()

    def grid_idx(self, idx):
        raise NotImplemented

    def init_sequence_optional_starts(self):
        for i, sequence in enumerate(self.sequences):
            min_start = sum(len(seq) for seq in self.sequences[:i]) + i
            max_start = self.length - sum(len(seq) for seq in self.sequences[i:]) - len(self.sequences) + i + 2
            sequence.optional_starts = set(range(min_start, max_start))
        self.update_n_combinations()

    def update_n_combinations(self):
        self.n_combinations = np.prod([len(sequence.optional_starts) for sequence in self.sequences])

    def set_constraint(self, idx, value):
        self.constraints[idx] = value
        drop_condition = (lambda start, seq: idx in [start - 1, start + len(seq)]) if value == 1 \
            else (lambda start, seq: start <= idx < start + len(seq))

        for sequence in self.sequences:
            for start in list(filter(lambda start: drop_condition(start, sequence), sequence.optional_starts)):
                sequence.optional_starts.discard(start)

        self.update_n_combinations()

    def get_combinations(self, level=0, min_start=0):
        if level == len(self.sequences):
            if not np.any(self.constraints[min_start: self.length] == 1):
                yield np.zeros(self.length)
        else:
            sequence = self.sequences[level]
            for start in sequence.optional_starts:
                end = start + len(sequence)
                if (min_start <= start
                        and not np.any(self.constraints[min_start: start] == 1)
                        and not np.any(self.constraints[start: end] == 0)
                        and not np.any(self.constraints[end: end + 1] == 1)):
                    for combination in self.get_combinations(level + 1, end + 1):
                        combination[start: start + len(sequence)] = 1
                        yield combination

    def find_constraints(self):
        random_combination = next(iter(self.get_combinations()))
        optional_new_constraints = np.isnan(self.constraints)
        optional_new_constraint_indexes, = np.where(optional_new_constraints)
        constraint_values = random_combination[optional_new_constraint_indexes]

        for combination in self.get_combinations():
            diverse_indexes, = np.where(combination[optional_new_constraint_indexes] != constraint_values)
            if diverse_indexes.size > 0:
                optional_new_constraint_indexes = np.delete(optional_new_constraint_indexes, diverse_indexes)
                constraint_values = random_combination[optional_new_constraint_indexes]
                if constraint_values.size == 0:
                    break

        new_constraints = dict(zip(optional_new_constraint_indexes, constraint_values))
        for idx, value in new_constraints.items():
            self.set_constraint(idx, value)

        return new_constraints


class Row(Array):
    def __repr__(self):
        return f'Row {self.idx}'

    def grid_idx(self, idx):
        return self.idx, idx


class Col(Array):
    def __repr__(self):
        return f'Col {self.idx}'

    def grid_idx(self, idx):
        return idx, self.idx


class Nonogram:
    @classmethod
    def load_img(cls, img_path):
        img = mpimg.imread(img_path)
        n_rows = img.shape[0]
        n_cols = img.shape[1]
        rows = [Row.load_array(i, img[i, :]) for i in range(n_rows)]
        cols = [Col.load_array(i, img[:, i]) for i in range(n_cols)]
        return cls(rows, cols)

    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.sol = None
        self.n_constraints = 0

    def solve(self, progress_bar=False):
        self.sol = 0.5 * np.ones([len(self.rows), len(self.cols)])
        updates = set(self.rows + self.cols)
        n_slots = int.__mul__(*self.sol.shape)
        gen = tqdm(range(n_slots)) if progress_bar else range(n_slots)
        for i in gen:
            if i < self.n_constraints:
                continue
            while updates:
                array = min(updates, key=lambda arr: arr.n_combinations)
                updates.discard(array)
                new_constraints = array.find_constraints()
                for idx, value in new_constraints.items():
                    grid_idx = array.grid_idx(idx)
                    self.sol[grid_idx] = value
                    affected_array = self.cols[idx] if type(array) is Row else self.rows[idx]
                    affected_array.set_constraint(array.idx, value)
                    updates.add(affected_array)
                if new_constraints:
                    self.n_constraints += len(new_constraints)
                    break

    def preview(self):
        plt.imshow(self.sol, cmap='Greys')
        plt.show()

    def export_csv(self, filepath, with_solution=False):
        maxrow = max(len(row.sequences) for row in self.rows)
        maxcol = max(len(col.sequences) for col in self.cols)

        tbl = [[''] * (len(self.cols) + maxrow) for _ in range(len(self.rows) + maxcol)]

        for i, row in enumerate(self.rows):
            indent = maxrow - len(row.sequences)
            for k, seq in enumerate(row.sequences):
                tbl[maxcol + i][indent + k] = str(seq)

        for j, col in enumerate(self.cols):
            indent = maxcol - len(col.sequences)
            for k, seq in enumerate(col.sequences):
                tbl[indent + k][maxrow + j] = str(seq)

        if with_solution and self.sol:
            for i in range(self.sol.shape[0]):
                for j in range(self.sol.shape[1]):
                    tbl[maxcol + i][maxrow + j] = '0' if self.sol[i][j] else '1'

        with open(filepath, 'w') as fp:
            for line in tbl:
                fp.write(','.join(line) + '\n')
