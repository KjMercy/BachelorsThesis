from PyEMD import EMD, EEMD
import numpy  as np
import numpy.linalg as LG
from scipy.special import rel_entr as KL_dist
import pygad

class SignalCleaner:

    def __init__(
            self, 
            signal_list, 
            ensemble, 
            generations_per_signal, 
            parents_per_signal, 
            mutation_percent
            ):
        
        if ensemble:
            self.decomposer = EEMD()
        else:
            self.decomposer = EMD()

        self.signals = signal_list
        # each element of signals contains: 
        # (1) the signal and (2 )the time domain

        self.gen_per_signal = generations_per_signal
        self.parents_per_signal = parents_per_signal
        self.mutation_percent = mutation_percent

    def decompose(self):
        for i, signal in enumerate(self.signals):
            self.decomposer.eemd(signal.y, signal.t)
            self.imfs[i], self.res[i] = self.decomposer.get_imfs_and_residue()

            # imfs[i] is a list of the imfs of the i-eth signal
            # res[i] is the residual of the decomposition of the i-eth signal

    def __calc_distances(self, metric):
        """Calculating distances between the signal and it's IMFs"""
        
        for i, signal in enumerate(self.signals):
            self.distances[i] = [metric(signal.y, imf) for imf in self.imfs[i]]

    def imf_selection(self):
        """Separating the IMFs into one noise dominant group and one signal 
        dominant group by calculating the index of the boundary imf for each signal"""

        self.__calc_distances(KL_dist)

        for i, dist in enumerate(self.distances):
            self.j_boundary[i] = np.argmax(dist) + 1


    def __hard_threshold(self, imfs, thresholds):
        res = []
        for i, imf in enumerate(imfs):
            if LG.norm(imf) > thresholds[i]:
                res.append(imf)
            else:
                res.append(0)
        return res
    
    def __soft_threshold(self, imfs, thresholds):
        res = []
        for i, imf in enumerate(imfs):
            if LG.norm(imf) > thresholds[i]:
                res.append( (np.sign(imf))*(LG.norm(imf) - thresholds[i]) )
            else:
                res.append(0)
        return res

    def apply_thresholding(self, soft = False):
        for i, _ in enumerate(self.signals):

            if soft:
                self.tresholded_imfs[i] = self.__soft_threshold(
                    self.imfs[i, :self.j_boundary[i]],
                    self.thresholds[i]
                    )
            else:
                self.tresholded_imfs[i] = self.__hard_threshold(
                    self.imfs[i, :self.j_boundary[i]],
                    self.thresholds[i]
                    )

    def __calc_single_signal_thresholds(self, imfs, C, BETA, RHO):
        
        energy = []
        energy.append( np.sum(np.square(imfs[0])) )
        for k in range(1, len(imfs)):
            # solution[1] corresponds to BETA
            energy.append( (energy[0]/BETA) / (RHO**(-k)) )

        thresholds = []
        # TODO: capire cosa deve essere "n" (--> nel paper c'è scritto 'number
        # of samples in the signal being subjected to noise removal')
        # Per adesso supporro sia il numero di IMF soggette a thresholding
        n = len(imfs)
        for _ in imfs:
            thresholds.append( C*np.sqrt(energy[0]*2*np.log(n)) )

        return thresholds

    def calc_tresholds(self):
        """Calculate thresholds to be used for imf 'cleaning' """

        for i, _ in enumerate(self.signals):
            self.tresholds[i] = self.__calc_single_signal_thresholds(
                self.imfs[i, :self.j_boundary[i]],
                self.C[i], 
                self.BETA[i], 
                self.RHO[i]
            )

    def i_eth_fitness(self, signal_index):
        """Here we 'generate' the fitness function of the specific signal"""

        def fitness(ga_instance, solution, solution_idx):

            i = signal_index
            boundary = self.j_boundary[i]
            sum_signal_dominant_imfs = np.sum(self.imfs[i, boundary:], axis=0)

            thresholded_imfs = self.__hard_threshold(
                self.imfs[i, :boundary],
                self.__calc_single_signal_thresholds(
                    self.imfs[i, :boundary],
                    solution[0], # Gene reppresenting C
                    solution[1], # Gene reppresenting BETA
                    solution[2] # Gene reppresenting RHO
                )
            )
            sum_thresholded_imfs = np.sum(thresholded_imfs, axis=0)

            # x: noisy ECG signal
            x = np.sum(self.imfs[i, :], axis=0) + self.res[i]

            # y: original ECG signal ---> TODO: qual'è la differenza tra x e y?
            y = self.signals[i].y # !!! Credo sia sbagliato, siccome sarebbe uguale a x

            # y_pred: reconstructed denoised ECG signal calculated with
            y_pred =  sum_thresholded_imfs + sum_signal_dominant_imfs + self.res[i]


            return 10*np.log10( np.sum(np.square(x - y)) / np.sum(np.square(y_pred - y)) )
        
        return fitness

    def GA(self):
        for i, signal in enumerate(self.signals):

            # We get the fitness function of the current signal
            fitness = self.i_eth_fitness(i)
            ga = pygad.GA(
                num_generations=self.gen_per_signal,
                num_parents_mating=self.parents_per_signal,
                num_genes=3, # i.e. C, BETA and RHO 
                mutation_percent_genes=self.mutation_percent,
                fitness_func=fitness
            )
            ga.run()
            solution, solution_fitness, solution_idx = ga.best_solution()
            self.C[i] = solution[0]
            self.BETA[i] = solution[1]
            self.RHO[i] = solution[2]

    def run(self, soft_thresholding = True):
        self.decompose()
        self.imf_selection()

        self.GA()
        self.calc_tresholds()
        self.apply_thresholding(soft_thresholding)
        