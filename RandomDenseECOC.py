import matplotlib.pyplot as plt
import numpy as np



'''
Notes : 

1 - if k is the number of classes , then the size of the matrix should be (k , 10*log(k))
2 - we generate 10*log(k) matrices with random elements with values : {-1, 1}. 
3 - we have to choose a probability distribution for the whole problem, it's best the data has the same distribution of probability as the matrices. 
4 - the author says its best we generate 10^4 random matrices. 

'''

class RandomDenseECOC : 

    def __init__(self,f, X, y,  num_classes, num_matrices = 10**4, ratio = 0.8):

        self.number_classes = num_classes
        self.number_matrices = num_matrices
        self.number_dichotomizers = 10*np.log(self.number_classes)
        self.ratio = ratio
        self.X, self.y = X, y # out data
        self.classifier = f # our margin based classifier (SOM SVM in our case)
        self.matrices = []  # our num_matrices generated shape (num_matrices , num_classes , num_dichotomizers), initilized to []
        self.best_matrix = []

    
    def train_test_split(self):    
        '''
        function to split our data for training and testing
        ratio : % of data that goes to the training set
        '''
        n = len(self.X)
        n_samples = int(np.around(n*self.ratio, 0))
        n_data = np.arange(n)
        np.random.shuffle(n_data)
        idx_train = np.random.choice(n_data, n_samples, replace=False)
        idx_test = list(set(n_data) - set(idx_train))    
        X_train , X_test , Y_train, Y_test = self.X[idx_train, ], self.X[idx_test, ], self.y[idx_train, ], self.y[idx_test, ]
        return X_train , X_test , Y_train, Y_test

    def generate_matrix(self):
        '''
        for a distribution D (uniform in this case), we generate a matrix of dimensions (n_row , n_col) 
        '''
        M = np.empty(shape=(self.number_classes,self.number_dichotomizers))
        for row in range(self.number_dichotomizers):
            arr = np.random.randint(2, size = self.number_dichotomizers)
            M[row] = [-1 if item == 0 else 1 for item in arr]
        
        return M.tolist()

    
    def Coding_matrices(self):
        '''
        with this method, we generate the asked number of matrices
        number_classes : number of classes
        number_matrices : number of matrices to generate 
        '''
        for matrix in range(self.number_matrices):
            M = self.generate_matrix()
            self.matrices.append(M)
        self.matrices = np.array(self.matrices)
    

    def Laplacian_decoder(self, x, y, K):
        '''
        "the chosen decoder for our Random dense Ecoc "
        
        (x, y) : two vectors to compare their similarity
        K : the number of classes ??
        alpha : the number of matched features between x and y 
        beta : the number of mismatched features between x and y without taking into account 0 if the matrix is formed with {-1, 0, 1} (sparse random ecoc case)

        '''
        if len(x) != len(y) :
            raise Exception("invalid comparison , please check if the sizes of the vectors are the same")

        x, y = np.array(x), np.array(y)
        alpha = len(np.where(x == y)[0])

        #beta = len(np.where((x == y) == False)[0])
        # in this case: (Dense random ECOC), we have alpha + beta == len(x) == len(y) 
        
        output = (alpha + 1) / (len(x) + K)
        return output
    

    def construct_binary_problems(self,M, Y): 
        '''
        this function creates all the subsets of data for each dichotomizer (column vector) of the coding matrix M
        output : indexes of the labels (positive and negative) for each dichotomizer
        '''
        cls = np.unique(self.y)
        binary_indexes = []
        for i in range(M.shape[1]):
            # Let's determine the positive and negative classes
            pos_class_idx = np.where(M[:,i] == 1)
            neg_class_idx = np.where(M[:,i] != 1)
            pos_classes = cls[pos_class_idx]
            neg_classes = cls[neg_class_idx]
            #Let's now create our subdatasets for each binary problem
            pos_cond = [Y == pos_classes[i] for i in range(len(pos_classes))]
            neg_cond = [Y == neg_classes[i] for i in range(len(neg_classes))]
            pos_idx = np.where(np.logical_or.reduce(pos_cond))[0]  
            neg_idx = np.where(np.logical_or.reduce(neg_cond))[0] 
            # Create a tuple of both positive and negative indexes
            tmp = (pos_idx, neg_idx)
            binary_indexes.append(tmp)    
        return binary_indexes


    def train(self):
        X_train , X_test , Y_train, Y_test = self.train_test_split()
        self.Coding_matrices() # we generate our test coding matrices
        best_matrix_test = []
        for M in self.matrices :
            binary_idx = self.construct_binary_problems(M, Y_train)
            X_tmp = np.copy(X_train)
            model_list = []
            for tup in binary_idx : 
                y_tmp = np.copy(Y_train)
                y_tmp[tup[0]] = 1 
                y_tmp[tup[1]] = -1
                res = self.f(X_tmp, y_tmp) #returns the final weights/predicting parameters/lagrange multipliers(in case of SOM)
                model_list.append(res)
            #Let's evaluate the test set using all the hypothesis to create a binary sequence
            
            y_pred = np.zeros(Y_test.shape)
            i = 0
            for x in X_test : 
                print(i)
                #print(x)
                tmp_x = np.array([1, x[0], x[1]])
                mod_Vect = []  # a vector of size L
                for res in model_list :
                    dot = np.dot(tmp_x , res)
                    sigmoid = 1 / (1 + np.exp(-dot))
                    if sigmoid > 0.5 : 
                        mod_Vect.append(1)
                    else :
                        mod_Vect.append(-1)
                mod_Vect = np.array(mod_Vect)

            # we now extract the class corresponding to the input instance
                distance_list = []
                for row in M : 
                    d = self.Laplacian_decoder(mod_Vect, row, self.number_classes)
                    distance_list.append(d)
                
                min_idx = np.argmax(distance_list) # is the index corresponding to the predicted class
                result_class = np.unique(self.y)[min_idx]
                y_pred[i] = result_class
                print('actual class : ', Y_test[i] ,' ****** predicted class : ',result_class )
                i+= 1
            
            # now we have predicted labels for the test set. 

            # let's now evaluate the f1 score of this matrix : 

            true_positives = len(np.where((y_pred == Y_test) & (y_pred == 1) )[0])
            false_positives = len(np.where((y_pred != Y_test) & (y_pred == 1) )[0])
            false_negatives = len(np.where((y_pred != Y_test) & (y_pred == -1) )[0])

            f1_score = (true_positives) / [true_positives + 1/2*(false_positives + false_negatives)]

            tmp = [M, f1_score]
            best_matrix_test.append(tmp)
        
        # Now let's determine the best matrix !!!

        self.best_matrix, best_score = sorted(best_matrix_test, key=lambda x : x[1], reverse=True)[0]
        
        print(f'the best f1 score of all the generated matrices is :{best_score}')

        return 



if __name__ == '__main__':
    """DR = RandomDenseECOC(4, 20)
    print()"""