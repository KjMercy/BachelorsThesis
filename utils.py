from PyEMD import EMD, CEEMDAN
import numpy  as np
import numpy.linalg as LG
from scipy.special import rel_entr as KL_dist
from scipy.stats import gaussian_kde
import pygad

def add_AWGN(signal, desired_SNR):
    noise_power = np.var(signal) / (10 ** (desired_SNR/10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))

    return signal + noise

def SNR_improvement(noisy_signal, signal, predicted):
    return 10*np.log10( np.sum(np.square(noisy_signal - signal)) / np.sum(np.square(predicted - signal)) )

def signal_MSE(signal, predicted):
    return np.mean(np.square(predicted - signal))

def estimate_pdf(signal_data):
    """Function to estimate the PDFs of the signals"""
        
    n = len(signal_data)
    sigma = np.std(signal_data, ddof=1)
    
    # Silverman's Rule of Thumb for bandwidth
    bandwidth = 1.06 * sigma * n**(-1/5)
    
    # Sturges' Rule for the number of bins
    num_bins = int(np.ceil(np.log2(n) + 1))
    
    kde = gaussian_kde(signal_data, bw_method=bandwidth)
    bin_centers = np.linspace(signal_data.min(), signal_data.max(), num_bins)
    pdf = kde(bin_centers)
    pdf = pdf/pdf.sum()
    return pdf

class SignalCleaner:
    """Class for the denoising of a signal using EEMD and GA

    Denoising of a signal by calculating it's IMFs (Intrinsic mode functions)
    through either EEMD (Ensemble Empirical Mode Decomposition) or EMD (Ensemble 
    Empirical Mode Decomposition). Through the Kullback-Leibler divergence, it is
    calculated the boundary IMF that stands between the signal-dominant IMFs and
    the noise-dominant IMFs. A GA (Genetic Algorithm) based adaptive thresholding 
    is applied to the noise-dominant set of IMFs. In the end, the denoised 
    noise-dominant IMFs are added to the signal-dominant IMFs to produce the clean signal

    """


    def __init__(
            self, 
            signal_list, 
            ensemble, 
            generations_per_signal, 
            parents_per_signal, 
            mutation_percent,
            SNR_input,
            hard_threshold=False,
            ):
        
        """
        Attributes:
            signal_list: A list of the signals to which to apply the denoising
            ensemble: boolean to indicate whether to use ensemgle or standard 
            empirical mode decompisotion
            generations_per_signal: integer indicating the number of generations for
            the calculating of the thresholds using the GA
            parents_per_signal: integer indicating the number of parents for the 
            calculating of the thresholds using the GA
            mutation_percent: float indicating the probability of mutation of the of
            a gene (here parameter for calculating the threshold)
            SNR_input: float indicating the input Signal to Noise Ratio
            hard_threshold: boolean to indicate the usage of either hard or soft thresholding
        """
        
        if ensemble:
            self.decomposer = CEEMDAN()
        else:
            self.decomposer = EMD()

        self.original_signals = signal_list # which will act as ground truth
        self.signals = [None]*len(signal_list) # which will be the signals to be cleaned
        
        self.SNR_input = SNR_input
        
        # AWGN is added to each signal
        for i, sig in enumerate(self.original_signals):
            self.signals[i] = add_AWGN(signal_list[i], self.SNR_input)
            
        # The density functions of the signals are computed    
        self.signal_pdfs = [None]*len(self.signals)
        for i, sig in enumerate(self.signals):
            self.signal_pdfs[i] = estimate_pdf(sig)
        
        self.gen_per_signal = generations_per_signal
        self.parents_per_signal = parents_per_signal
        self.mutation_percent = mutation_percent
        
    def decompose(self):
        """Decomposes the signal into multiple IMFs using the specified decomposer"""
        self.imfs = [None]*len(self.signals)
        self.res = [None]*len(self.signals)
        for i, signal in enumerate(self.signals):
            self.decomposer.ceemdan(signal, list( range(len(signal)) ))
            self.imfs[i], self.res[i] = self.decomposer.get_imfs_and_residue()

            # imfs[i] is a list of the imfs of the i-eth signal
            # res[i] is the residual of the decomposition of the i-eth signal
            
    def calc_imf_pdfs(self):
        """Calculates the density function of each imf of each signal"""
        
        self.imf_pdfs = [None]*len(self.imfs)
        for i, signal in enumerate(self.signals):
            self.imf_pdfs[i] = [None]*len(self.imfs[i])
            for j, imf in enumerate(self.imfs[i]):
                self.imf_pdfs[i][j] = self.imfs[i][j].copy()
                self.imf_pdfs[i][j] = estimate_pdf(self.imfs[i][j])

    def __calc_distances(self, metric):
        """Calculates distances between the signal and it's IMFs"""
        
        #self.distances = [None]*len(self.signals)
        #for i, signal in enumerate(self.signals):
            #self.distances[i] = [metric(signal, imf) for imf in self.imfs[i]]
            
        self.distances = [None]*len(self.signal_pdfs)
        for i, sig_pdf in enumerate(self.signal_pdfs):
            self.distances[i] = [sum(metric(sig_pdf, imf_pdf)) for imf_pdf in self.imf_pdfs[i]]

    def imf_selection(self):
        """Separates the IMFs into one noise dominant group and one signal 
        dominant group by calculating the index of the boundary imf for each signal"""

        self.__calc_distances(KL_dist)

        self.j_boundary = [None]*len(self.signals)
        for i, dist in enumerate(self.distances):
            self.j_boundary[i] = np.argmax(dist) + 1

    def __hard_threshold(self, imfs, thresholds):
        """Applies hard thresholding to the specified imfs using the thresholds
        passed as arguments"""

        res = []
        for i, imf in enumerate(imfs):
            if LG.norm(imf) > thresholds[i]:
                res.append(imf)
            else:
                res.append(0)
        return res
    
    def __soft_threshold(self, imfs, thresholds):
        """Applies soft thresholding to the specified imfs using the thresholds
        passed as arguments"""

        res = []
        for i, imf in enumerate(imfs):
            if LG.norm(imf) > thresholds[i]:
                res.append( (np.sign(imf))*(LG.norm(imf) - thresholds[i]) )
            else:
                res.append(0)
        return res

    def apply_thresholding(self, soft = False):
        """Applies thresholding to all the noise-dominant IMF groups (i.e the noise
        dominant IMFs for each signal). If soft is True, it applies soft-thresholding"""

        self.thresholded_imfs = [None]*len(self.signals)
        for i, _ in enumerate(self.signals):

            if soft:
                self.thresholded_imfs[i] = self.__soft_threshold(
                    self.imfs[i][:self.j_boundary[i]],
                    self.thresholds[i]
                    )
            else:
                self.thresholded_imfs[i] = self.__hard_threshold(
                    self.imfs[i][:self.j_boundary[i]],
                    self.thresholds[i]
                    )

    def __calc_single_signal_thresholds(self, imfs, C, BETA, RHO):
        
        energy = []
        energy.append( np.sum(np.square(imfs[0])) )
        for k in range(1, len(imfs)):
            energy.append( (energy[0]/BETA) / (RHO**(-k)) )

        thresholds = []
        n = len(imfs[0]) #number of samples in the signal being subjected to noise removal
        for i in range(len(imfs)):
            thresholds.append( C*np.sqrt(energy[i]*2*np.log(n)) )
            
        return thresholds

    def calc_thresholds(self):
        """Calculate thresholds to be used for imf 'cleaning' """

        self.thresholds = [None]*len(self.signals)
        for i, _ in enumerate(self.signals):
            self.thresholds[i] = self.__calc_single_signal_thresholds(
                self.imfs[i][:self.j_boundary[i]],
                self.C[i], 
                self.BETA[i], 
                self.RHO[i]
            )

    def i_eth_fitness(self, signal_index):
        """Here we 'generate' the fitness function of the specific signal"""

        def fitness(ga_instance, solution, solution_idx):
            """The Signal To Noise Improvement (SNR improvement) is used as
            an objective function, or fitness"""
            
            thresholding_func = self.__hard_threshold if self.hard_threshold else self.__soft_threshold

            i = signal_index
            boundary = self.j_boundary[i]
            #sum_signal_dominant_imfs = np.sum(self.imfs[i, boundary:], axis=0)
            sum_signal_dominant_imfs = np.sum(self.imfs[i][boundary:], axis=0)

            thresholded_imfs = self.thresholding_func(
                self.imfs[i][:boundary],
                self.__calc_single_signal_thresholds(
                    self.imfs[i][:boundary],
                    solution[0], # Gene reppresenting C
                    solution[1], # Gene reppresenting BETA
                    solution[2] # Gene reppresenting RHO
                )
            )
            sum_thresholded_imfs = np.sum(thresholded_imfs, axis=0)

            # x: noisy ECG signal
            #x = add_AWGN(self.signals[i], self.SNR_input) TODO: delete this line if the one below works
            x = self.signals[i]

            # y: original ECG signal
            # y = self.signals[i] TODO: delete this line if the one below works
            y = self.original_signals[i]

            # y_pred: reconstructed denoised ECG signal calculated with
            y_pred =  sum_thresholded_imfs + sum_signal_dominant_imfs + self.res[i]


            # return 10*np.log10( np.sum(np.square(x - y)) / np.sum(np.square(y_pred - y)) )
            return SNR_improvement(x, y, y_pred)
        
        return fitness

    def GA(self):
        self.C = [None]*len(self.signals)
        self.BETA = [None]*len(self.signals)
        self.RHO = [None]*len(self.signals)
        for i, signal in enumerate(self.signals):

            # We get the fitness function of the current signal
            fitness = self.i_eth_fitness(i)
            ga = pygad.GA(
                num_generations=self.gen_per_signal,
                num_parents_mating=self.parents_per_signal,
                num_genes=3, # i.e. C, BETA and RHO 
                mutation_percent_genes=self.mutation_percent,
                fitness_func=fitness,
                sol_per_pop=10,
                gene_space=[ {'low': 0.5, 'high': 1.0},{'low': 0.5, 'high': 1.5},{'low': 1.5, 'high': 3.0} ]
            )
            ga.run()
            solution, solution_fitness, solution_idx = ga.best_solution()
            self.C[i] = solution[0]
            self.BETA[i] = solution[1]
            self.RHO[i] = solution[2]
            #ga.plot_fitness()

    def predict(self):
        
        self.y_pred = [None]*len(self.signals)
        for i, signal in enumerate(self.signals):
                         
            sum_thresholded_imfs = np.sum(self.thresholded_imfs[i], axis=0)
            boundary = self.j_boundary[i]
            sum_signal_dominant_imfs = np.sum(self.imfs[i][boundary:], axis=0)

            self.y_pred[i] = sum_thresholded_imfs + sum_signal_dominant_imfs + self.res[i]

    def run(self, hard_thresholding = False):
        self.decompose()
        self.calc_imf_pdfs()
        self.imf_selection()

        self.GA()
        self.calc_thresholds()
        self.apply_thresholding(soft_thresholding)
        self.predict()
        