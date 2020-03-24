########################################
# CS/CNS/EE 155 2018
# Problem Set 6
#
# Author:       Andrew Kang
# Description:  Set 6 skeleton code
########################################

# You can use this (optional) skeleton code to complete the HMM
# implementation of set 5. Once each part is implemented, you can simply
# execute the related problem scripts (e.g. run 'python 2G.py') to quickly
# see the results from your code.
#
# Some pointers to get you started:
#
#     - Choose your notation carefully and consistently! Readable
#       notation will make all the difference in the time it takes you
#       to implement this class, as well as how difficult it is to debug.
#
#     - Read the documentation in this file! Make sure you know what
#       is expected from each function and what each variable is.
#
#     - Any reference to "the (i, j)^th" element of a matrix T means that
#       you should use T[i][j].
#
#     - Note that in our solution code, no NumPy was used. That is, there
#       are no fancy tricks here, just basic coding. If you understand HMMs
#       to a thorough extent, the rest of this implementation should come
#       naturally. However, if you'd like to use NumPy, feel free to.
#
#     - Take one step at a time! Move onto the next algorithm to implement
#       only if you're absolutely sure that all previous algorithms are
#       correct. We are providing you waypoints for this reason.
#
# To get started, just fill in code where indicated. Best of luck!

import random

class HiddenMarkovModel:
    '''
    Class implementation of Hidden Markov Models.
    '''

    def __init__(self, A, O):
        '''
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0. 
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state.

        Arguments:
            A:          Transition matrix with dimensions L x L.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.

            O:          Observation matrix with dimensions L x D.
                        The (i, j)^th element is the probability of
                        emitting observation j given state i.

        Parameters:
            L:          Number of states.
            
            D:          Number of observations.
            
            A:          The transition matrix.
            
            O:          The observation matrix.
            
            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        '''

        self.L = len(A)
        self.D = len(O[0])
        self.A = A
        self.O = O
        self.A_start = [1. / self.L for _ in range(self.L)]


    def viterbi(self, x):
        '''
        Uses the Viterbi algorithm to find the max probability state 
        sequence corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            max_seq:    State sequence corresponding to x with the highest
                        probability.
        '''

        M = len(x)      # Length of sequence.

        # The (i, j)^th elements of probs and seqs are the max probability
        # of the prefix of length i ending in state j and the prefix
        # that gives this probability, respectively.
        #
        # For instance, probs[1][0] is the probability of the prefix of
        # length 1 ending in state 0.
        probs = [[0. for _ in range(self.L)] for _ in range(M + 1)]
        seqs = [['' for _ in range(self.L)] for _ in range(M + 1)]
        
        ### TODO: Insert Your Code Here (2A)
        
        # Initialize the state when length = 1
        for state in range(self.L):
            probs[1][state] = self.A_start[state] * self.O[state][x[0]]
            seqs[1][state] = str(state)

        # Iterate over the prefixes of length 2 to M
        for length in range(2, M + 1):
            # Iterate over current states
            for state_cur in range(self.L):
                # Initialization for iteration
                prefix = ''
                max_proba = 0
                
                #Iterate over previous states
                for state_prev in range(self.L):
                    # Calulate the probability given by previous states
                    proba = probs[length - 1][state_prev] * self.A[state_prev][state_cur] * self.O[state_cur][x[length - 1]]
                    
                    # Update the sequence with larger probability by comparsion
                    if proba >= max_proba:
                        # Store the corresponding sequence and probability
                        prefix = seqs[length - 1][state_prev]
                        max_proba = proba
                
                # Update the entire sequence and corresponding probability to be the final results
                probs[length][state_cur] = max_proba
                seqs[length][state_cur] = prefix + str(state_cur)    
        
        # Find the sequence with maximized probability
        max_seq = seqs[M][probs[M].index(max(probs[M]))]
        return max_seq



    def forward(self, x, normalize=False):
        '''
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            alphas:     Vector of alphas.

                        The (i, j)^th element of alphas is alpha_j(i),
                        i.e. the probability of observing prefix x^1:i
                        and state y^i = j.

                        e.g. alphas[1][0] corresponds to the probability
                        of observing x^1:1, i.e. the first observation,
                        given that y^1 = 0, i.e. the first state is 0.
        '''

        M = len(x)      # Length of sequence.
        alphas = [[0. for _ in range(self.L)] for _ in range(M + 1)]
 
        ### TODO: Insert Your Code Here (2Bi)
        
        # Initialize the state when length = 1
        for state in range(self.L):
            alphas[1][state] = self.A_start[state] * self.O[state][x[0]]

        # Iterate over the prefixes of length 2 to M
        for length in range(2, M + 1):
            # Iterate over current states
            for state_cur in range(self.L):
                # Initialization for iteration
                sum_proba = 0
                
                #Iterate over previous states
                for state_prev in range(self.L):
                    # Calculating the total probability of the transition step by accumulation
                    sum_proba += alphas[length - 1][state_prev] * self.A[state_prev][state_cur]
                
                # Update the alpha matrix (for state_cur) by multiplying the observed probability and the transition probaility
                alphas[length][state_cur] = self.O[state_cur][x[length - 1]] * sum_proba
            
            # Possible normalization
            if normalize:
                alphas[length] = [alpha / sum(alphas[length]) for alpha in alphas[length]]
                
        return alphas


    def backward(self, x, normalize=False):
        '''
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            betas:      Vector of betas.

                        The (i, j)^th element of betas is beta_j(i), i.e.
                        the probability of observing prefix x^(i+1):M and
                        state y^i = j.

                        e.g. betas[M][0] corresponds to the probability
                        of observing x^M+1:M, i.e. no observations,
                        given that y^M = 0, i.e. the last state is 0.
        '''

        M = len(x)      # Length of sequence.
        betas = [[0. for _ in range(self.L)] for _ in range(M + 1)]
        
        ### TODO: Insert Your Code Here (2Bii)
        
        # Initialize the state when length = M
        betas[M] = [1. for _ in range(self.L)]
        
        # Iterate over the prefixes of length M - 1 to 1
        for length in range(1, M)[::-1]:
            # Iterate over current states
            for state_back in range(self.L):
                # Initialization for iteration
                sum_proba = 0
                
                #Iterate over previous states
                for state_up in range(self.L):
                    # Calculating the total probability of the transition step by accumulation
                    sum_proba += betas[length + 1][state_up] * self.A[state_back][state_up] * self.O[state_up][x[length]]
                
                # Update the beta matrix (for state_back)
                betas[length][state_back] = sum_proba
            
            # Possible normalization
            if normalize:
                betas[length] = [beta / sum(betas[length]) for beta in betas[length]]
                
        return betas


    def supervised_learning(self, X, Y):
        '''
        Trains the HMM using the Maximum Likelihood closed form solutions
        for the transition and observation matrices on a labeled
        datset (X, Y). Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to D - 1. In other words, a list of
                        lists.

            Y:          A dataset consisting of state sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to L - 1. In other words, a list of
                        lists.

                        Note that the elements in X line up with those in Y.
        '''

        ### TODO: Insert Your Code Here (2C)
        
        # Extracting the size of X (initialization)
        N = len(X)
        
        # Calculate each element of A using the M-step formulas.
        # Iterate over the A matrix 
        for a in range(self.L):
            for b in range(self.L):
                # Initialize the stored values
                denominator = 0
                numerator = 0
                # Iterate over the Y matrix
                for j in range(N):
                    # Starting from 1 here (given by slides) in the iteration
                    for i in range(1, len(Y[j])):
                         # Check the first indicator
                        if Y[j][i - 1] == a:
                            denominator += 1
                             # Check the second indicator
                            if Y[j][i] == b:
                                numerator += 1
                # Update the A matrix with stored values                   
                self.A[a][b] = numerator / denominator

        # Calculate each element of O using the M-step formulas.
        # Iterate over the O matrix 
        for z in range(self.L):
            for w in range(self.D):
                # Initialize the stored values
                denominator = 0
                numerator = 0
                # Iterate over the Y matrix
                for j in range(N):
                    # Starting from 0 here (given by slides) in the iteration
                    for i in range(len(Y[j])):
                        # Check the first indicator
                        if Y[j][i] == z:
                            denominator += 1
                            # Check the second indicator
                            if X[j][i] == w:
                                numerator += 1
                # Update the O matrix with stored values                
                self.O[z][w] = numerator / denominator


    def unsupervised_learning(self, X, N_iters):
        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        dataset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of length M, consisting of integers ranging
                        from 0 to D - 1. In other words, a list of lists.

            N_iters:    The number of iterations to train on.
        '''
        
        ### TODO: Insert Your Code Here (2D)
        N = len(X)
        
        # Initialize the two matrices of marginal probabilities
        # For the first margin matrix, entries are given by margin[i][j][a] = P(y_{i}^j = a, x_i) (i: sample, j: step, a: choices)
        margin = [[[0. for _ in range(self.L)] for _ in range(len(sample))] for sample in X]
        # For the second margin matrix, entries are given by trans_margin[i][j][a] = P(y_{i}^j = a, y_i^(j+1) = b, x_i)
        # (i: sample, j, j+1: step transition, a, b: choices)
        trans_margin = [[[[0. for _ in range(self.L)] for _ in range(self.L)] for _ in range(len(sample) - 1)] for sample in X]

        # Iterate over the training for N_iters times
        for training in range(N_iters):
            print(training)
            if training % 5 == 0:
                print("Iteration: " + str(training))
            
            # E-step: Calculate the two marginal probability matrices by iterating over the training examples
            for i in range(N):
                # Initialization with forward/backward algorithm implemented above
                alphas = self.forward(X[i], True)
                betas = self.backward(X[i], True)
                
                # Iterate over the training steps
                for j in range(1, len(X[i]) + 1):
                    # Store unnormalized data
                    margin_ij = [0. for _ in range(self.L)]
                    # Iterate over the possible choices
                    for a in range(self.L):
                        # Store marginal value
                        margin_ij[a] = alphas[j][a] * betas[j][a]
                    # Normalization
                    margin[i][j - 1] = [prob / sum(margin_ij) for prob in margin_ij]
                    # Boundary case
                    if j == len(X[i]):
                        break

                    # Store unnormalized data
                    trans_margin_ij = [[0. for _ in range(self.L)] for _ in range(self.L)]
                    denominator_sum = 0
                    # Iterate over the possible choices (2 choices, which represent a transition)
                    for a in range(self.L):
                        for b in range(self.L):
                            # Store marginal value and update the sum of all probabilities by accumulation
                            trans_margin_ij[a][b] = (alphas[j][a] * self.O[b][X[i][j]] * self.A[a][b] * betas[j + 1][b])
                            denominator_sum += trans_margin_ij[a][b]
                    # Normalization
                    trans_margin[i][j - 1] = [[trans / denominator_sum for trans in choice] for choice in trans_margin_ij]
        
            # M-step: update A, O matrices
            # Iterate over the A matrix 
            for a in range(self.L):
                for b in range(self.L):
                    # Initialize the stored values
                    denominator = 0
                    numerator = 0
                    # Iterate over the marginal matrices
                    for i in range(N):
                        for j in range(1, len(X[i])):
                            denominator += margin[i][j - 1][a]
                            numerator += trans_margin[i][j - 1][a][b]
                    # Update the A matrix with stored values                   
                    self.A[a][b] = numerator / denominator
            
            # Iterate over the O matrix
            for z in range(self.L):
                for w in range(self.D):
                    # Initialize the stored values
                    denominator = 0
                    numerator = 0
                    # Iterate over the Y matrix
                    for i in range(N):
                        for j in range(len(X[i])):
                            denominator += margin[i][j][z]
                            # Check the indicator
                            if X[i][j] == w:
                                numerator += margin[i][j][z]
                    # Update the O matrix with stored values                
                    self.O[z][w] = numerator / denominator
    
    def find_state(self, given_emission):
        # Extract the list P(y | x) for a given x
        proba_list = []
        for row in self.O:
            proba_list.append(row[given_emission])  
        # Sample from the probability list
        valid_state = random.choices(range(len(proba_list)), weights=proba_list)[0]
        return valid_state
    
    def find_syllable_real(self, word, syllable_dict, remain):
        key = word.lower()
        # Extract the real and end syllable lists
        real_syllable = syllable_dict[key][0]
        
        # Check if the word's real syllable satisfies our requirement
        for i in range(len(real_syllable)):
            if real_syllable[i] <= remain:
                return random.choice(real_syllable[i:])
        
        # If there's no valid syllable within the range, return 11 > 10
        return 11
    
    def find_syllable_end(self, word, syllable_dict):
        key = word.lower()
        # Extract the real and end syllable lists
        real_syllable = syllable_dict[key][0]
        end_syllable = syllable_dict[key][1]
        
        if len(end_syllable) > 0:
            return end_syllable[0]
        
        # If there's no valid real syllable, check if there's valid end syllable
        else:
             return random.choice(real_syllable)
        
        return 11
        

    def generate_emission(self, M_syllable, word_map, syllable_dict):
        '''
        Generates an emission of length M (syllable length), assuming that the starting state
        is chosen uniformly at random. 

        Arguments:
            M:          Length of the emission to generate.

        Returns:
            emission:   The randomly generated emission as a list.

            states:     The randomly generated states as a list.
        '''
        # Initialization
        emission = []
        states = []
        remain = M_syllable

        ### TODO: Insert Your Code Here (2F)

        # Iterate over possible syllable combinations
        while remain > 0:
            if remain == M_syllable:
                # Initialize y_{0} by sampling from the uniformly distribution
                states.append(random.choice(range(self.L)))
            else:
            # Sample y_{i} from the transition matrxi A (P(y_{i} | y_{i-1})
                next_state = random.choices(range(self.L), weights=self.A[states[-1]])[0]
                states.append(next_state)
                
            while True:
                # Sample x_{i} from the observation matrix O (P(x_{i} | y_{i}))
                next_observation = random.choices(range(self.D), weights=self.O[states[-1]])[0]
                # Extract word and check whether it's valid or not
                word = word_map[next_observation]
                syllable = self.find_syllable_real(word, syllable_dict, remain)
                if syllable != 11:
                    remain -= syllable
                    emission.append(next_observation)
                    break
        
        return emission, states

    def generate_emission_rhyme(self, given_word, given_state, M_syllable, word_map, obs_map, syllable_dict):
        # Initialization
        emission = []
        states = []
        remain = M_syllable

        ### TODO: Insert Your Code Here (2F)

        # Initialize y_{0} by sampling from the uniformly distribution
        states.append(given_state)
        emission.append(obs_map[given_word])
        given_syllable = self.find_syllable_end(given_word, syllable_dict)
        remain -= given_syllable

        # Iterate over possible syllable combinations
        while remain > 0:
            # Sample y_{i} from the transition matrxi A (P(y_{i} | y_{i-1})
            next_state = random.choices(range(self.L), weights=self.A[states[-1]])[0]
            states.append(next_state)
            while True:
                # Sample x_{i} from the observation matrix O (P(x_{i} | y_{i}))
                next_observation = random.choices(range(self.D), weights=self.O[states[-1]])[0]
                # Extract word and check whether it's valid or not
                word = word_map[next_observation]
                syllable = self.find_syllable_real(word, syllable_dict, remain)
                if syllable != 11:
                    remain -= syllable
                    emission.append(next_observation)
                    break
                    
        return emission, states
    
    def check_stress(self, word1, word2, length1, length2, stress_dict):
        word1_breakdown = []
        word2_breakdown = []
        for word in stress_dict:
            if word == word1 or word == word2:
                pronounce = stress_dict[word]
                for choice in pronounce:
                    stress_digit = [char for sub in choice for char in sub if char.isdigit()]
                    if word == word1 and len(stress_digit) == length1:
                        word1_breakdown = stress_digit
                    if word == word2 and len(stress_digit) == length2:
                        word2_breakdown = stress_digit             
        if len(word1_breakdown) > 0 and len(word2_breakdown) > 0:
            if int(word2_breakdown[-1]) == int(word1_breakdown[0]):
                return False
        else:
            return True
    
    
    def generate_emission_rhyme_stress(self, given_word, given_state, M_syllable, word_map, obs_map, syllable_dict, stress_dict):
        # Initialization
        emission = []
        states = []
        syllable_history = []
        remain = M_syllable

        ### TODO: Insert Your Code Here (2F)

        # Initialize y_{0} by sampling from the uniformly distribution
        states.append(given_state)
        emission.append(obs_map[given_word])
        given_syllable = self.find_syllable_end(given_word, syllable_dict)
        syllable_history.append(given_syllable)
        remain -= given_syllable

        # Iterate over possible syllable combinations
        while remain > 0:
            # Sample y_{i} from the transition matrxi A (P(y_{i} | y_{i-1})
            next_state = random.choices(range(self.L), weights=self.A[states[-1]])[0]
            states.append(next_state)
            while True:
                # Sample x_{i} from the observation matrix O (P(x_{i} | y_{i}))
                next_observation = random.choices(range(self.D), weights=self.O[states[-1]])[0]
                # Extract word and check whether it's valid or not
                word = word_map[next_observation]
                syllable = self.find_syllable_real(word, syllable_dict, remain)
                if syllable != 11:
                    adjacent_word = word_map[emission[-1]]
                    adjacent_length = syllable_history[-1]
                    result = self.check_stress(adjacent_word, word, adjacent_length, syllable, stress_dict)
                    if result == True:
                        remain -= syllable
                        emission.append(next_observation)
                        syllable_history.append(syllable)
                        break
                    
        return emission, states
    

    def probability_alphas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the forward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        # Calculate alpha vectors.
        alphas = self.forward(x)

        # alpha_j(M) gives the probability that the state sequence ends
        # in j. Summing this value over all possible states j gives the
        # total probability of x paired with any state sequence, i.e.
        # the probability of x.
        prob = sum(alphas[-1])
        return prob


    def probability_betas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the backward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        betas = self.backward(x)

        # beta_j(1) gives the probability that the state sequence starts
        # with j. Summing this, multiplied by the starting transition
        # probability and the observation probability, over all states
        # gives the total probability of x paired with any state
        # sequence, i.e. the probability of x.
        prob = sum([betas[1][j] * self.A_start[j] * self.O[j][x[0]] \
                    for j in range(self.L)])

        return prob


def supervised_HMM(X, Y):
    '''
    Helper function to train a supervised HMM. The function determines the
    number of unique states and observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for supervised learning.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        Y:          A dataset consisting of state sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to L - 1. In other words, a list of lists.
                    Note that the elements in X line up with those in Y.
    '''
    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Make a set of states.
    states = set()
    for y in Y:
        states |= set(y)
    
    # Compute L and D.
    L = len(states)
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with labeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.supervised_learning(X, Y)

    return HMM

def unsupervised_HMM(X, n_states, N_iters):
    '''
    Helper function to train an unsupervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for unsupervised learing.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        n_states:   Number of hidden states to use in training.
        
        N_iters:    The number of iterations to train on.
    '''

    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)
    
    # Compute L and D.
    L = n_states
    D = len(observations)
    
    # Randomly initialize and normalize matrix A.
    
    # SEED
    random.seed(2020)
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    # Randomly initialize and normalize matrix O.
    
    # SEED
    random.seed(155)
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.unsupervised_learning(X, N_iters)

    return HMM
