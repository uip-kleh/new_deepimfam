import os
import yaml
import json
import numpy as np
import matplotlib.pyplot as plt

class Common:
    def __init__(self, config_path) -> None:
        self.set_config(config_path)

    def set_config(self, config_path):
        # TODO: THERE MAY BE BETTER WAY
        with open(config_path, "r") as f: 
            args = yaml.safe_load(f)
            # PATH
            self.data_direcotry = self.join_home(args["data_directory"], True)
            self.results_directory = self.join_home(args["results_directory"], True)
            self.aaindex1_path = self.join_home(args["aaindex1_path"])
            self.amino_train_path = self.join_home(args["amino_train_data"])
            self.amino_test_path = self.join_home(args["amino_test_data"])

            # EXPERIMENT INDEX
            self.method_directory = self.join_home(args["method_directory"], True)
            self.index1 = args["index1"]
            self.index2 = args["index2"]
            index_combination = "_".join([self.index1, self.index2])
            self.experiment_directory = self.make_directory(os.path.join(self.method_directory, index_combination))
            self.coordinates_directory = self.make_directory(os.path.join(self.experiment_directory, "coordinates"))
            self.images_directory = self.make_directory(os.path.join(self.experiment_directory, "images"))
            self.results_directory = self.make_directory(os.path.join(self.experiment_directory, "results"))


    def join_home(self, fname, is_dir=False):
        fname = os.path.join(os.environ["HOME"], fname)
        if not os.path.exists(fname) and is_dir: 
            os.mkdir(fname)
        return fname
    
    def make_directory(self, fname):
        if not os.path.exists(fname):
            os.mkdir(fname)
        return fname
    
    def save_obj(self, obj, fname):
        with open(fname, "w") as f:
            json.dump(obj, f, indent=2)

    # LOAD AAINDEX1 FROM JSON
    def load_aaindex1(self):
        with open(self.aaindex1_path, "r") as f:
            self.aaindex1 = json.load(f)

    def save_figure(self, fname):
        plt.tight_layout()
        plt.savefig(fname, transparent=True)
        plt.cla()
        plt.clf()
        plt.close()

# CALC CORR
class AAindex1(Common):
    def __init__(self, config_path) -> None:
        Common.__init__(self, config_path)
        self.load_aaindex1()

    def calc(self):
        results = []
        
        # ELIMINATE INDEX WHICH HAS SAME VALUE
        keys = []
        for key, values in self.aaindex1.items():
            if self.has_same_value(np.array(list(values.values()))): 
                continue
            keys.append(key)

        N = len(keys)
        print("REST OF INDEX IS ", N)
        keys.sort()

        for i in range(N):
            for j in range(i+1, N):
                key1, key2 = keys[i], keys[j]

                # TODO: THERE MAY BE BETTER WAY
                values1 = np.array(list(self.aaindex1[key1].values()))
                values2 = np.array(list(self.aaindex1[key2].values()))

                corr = np.corrcoef(values1, values2)
                results.append([abs(corr[0][1]), key1, key2])
        
        results.sort()
        fname = os.path.join(self.results_directory, "corr.json")
        self.save_obj(results, fname)

    def has_same_value(self, values):
        print(values, not values.size == np.unique(values).size)
        return not values.size == np.unique(values).size

    def disp(self, key1, key2):
        print(self.aaindex1[key1])
        print(self.aaindex1[key2])

class DeepImFam(Common):
    def __init__(self, config_path) -> None:
        Common.__init__(self, config_path)

    def calc_coordinate(self):
        vectors = self.generate_std_vectors()
        self.draw_vectors(vectors)
        
    # GENERATE STANDRIZED VECTOR
    def generate_std_vectors(self):
        self.load_aaindex1()
        keys = self.aaindex1[self.index1].keys()
        values1 = np.array(list(self.aaindex1[self.index1].values()))
        values2 = np.array(list(self.aaindex1[self.index2].values()))
        std_values1 = self.standarize(values1)
        std_values2 = self.standarize(values2)

        vectors = {}
        for i, key in enumerate(keys):
            vectors[key] = [std_values1[i], std_values2[i]]
        return vectors

    def standarize(self, values: np.array):
        return (values - np.mean(values)) / np.std(values) 
    
    def draw_vectors(self, vectors):
        plt.figure()
        for key, items in vectors.items():
            plt.plot([0, items[0]], [0, items[1]])
            plt.text(items[0], items[1], key)
        plt.title("_".join([self.index1, self.index1]))
        fname = os.path.join(self.experiment_directory, "vectors.pdf")
        self.save_figure(fname)

    # CALCURATE COORDINATE
    def calc_coordinate(self):
        vectors = self.generate_std_vectors()
        sequences = self.read_sequences(self.amino_train_path) + self.read_sequences(self.amino_test_path)

        for i, seq in enumerate(sequences):
            fname = os.path.join(self.coordinates_directory, str(i) + ".dat")
            with open(fname, "w") as f:
                x, y = 0, 0
                print("{}, {}".format(x, y), file=f)
                for aa in seq:
                    if not aa in vectors:
                        continue
                    x += vectors[aa][0]
                    y += vectors[aa][1]
                    print("{}, {}".format(x, y), file=f)


    def read_sequences(self, path):
        sequences = []
        with open(path, "r") as f:
            for l in f.readlines():
                fam, seq = l.split()
                sequences.append(seq)
        return sequences

if __name__ == "__main__":
    # TEST: AAindex
    # aaindex1 = AAindex1(config_path="config.yaml")    
    # aaindex1.calc()
    # aaindex1.disp("NAKH900107", "PALJ810108")

    deepimfam = DeepImFam(config_path="config.yaml")
    deepimfam.calc_coordinate()