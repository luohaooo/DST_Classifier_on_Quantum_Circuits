import numpy as np
from combine_classical import combine_BPAS_classical
from combine_quantum import combineBPAs_quantum
from train_GMMs import learn_GMM_coefficients
from generate_BPAs import  generate_BPAs
from train_test_data import save_training_testing_data
from decision_making import decision_making

# simulate and record accuracy
def conduct_simulation(pro_data, train_ratio_list, components_num, N):
    mean_accuracy_quantum_32 = np.empty_like(train_ratio_list)
    mean_accuracy_quantum_1024 = np.empty_like(train_ratio_list)
    mean_accuracy_classical = np.empty_like(train_ratio_list)

    for k in range(len(train_ratio_list)):
        train_ratio = train_ratio_list[k]
        print("Train Ratio: "+ str(train_ratio))

        accuracy_record_quantum_32 = np.zeros(N)
        accuracy_record_quantum_1024 = np.zeros(N)
        accuracy_record_classical = np.zeros(N)

        for iternum in range(N):
            if iternum % 10 == 0:
                print("Iteration number: "+str(iternum))

            training_set, testing_set = save_training_testing_data(pro_data, train_ratio)

            testing_num = 0
            error_num_quantum_32 = 0
            error_num_quantum_1024 = 0
            error_num_classical = 0

            types_num = len(testing_set)

            weights, means, covariances = learn_GMM_coefficients(training_set, components_num)
            
            for ii in range(types_num):
                testing_num += len(testing_set[ii])
                for testing_data in testing_set[ii]:
                    BPA, alphas = generate_BPAs(testing_data, weights, means, covariances)
                
                    # quantum: 32768 shots
                    combined_BPA_quantum_32 = combineBPAs_quantum(alphas, 32768)
                    classification_result_quantum_32 = decision_making(combined_BPA_quantum_32)
                    if  classification_result_quantum_32 != ii:
                        error_num_quantum_32 += 1

                    # quantum: 1024 shots
                    combined_BPA_quantum_1024 = combineBPAs_quantum(alphas, 1024)
                    classification_result_quantum_1024 = decision_making(combined_BPA_quantum_1024)
                    if  classification_result_quantum_1024 != ii:
                        error_num_quantum_1024 += 1
                    
                    # classical
                    combined_BPA_classical = combine_BPAS_classical(BPA)
                    classification_result_classical = decision_making(combined_BPA_classical)
                    if  classification_result_classical != ii:
                        error_num_classical += 1
                    
            accuracy_record_quantum_32[iternum] = 1 - error_num_quantum_32/testing_num
            accuracy_record_quantum_1024[iternum] = 1 - error_num_quantum_1024/testing_num
            accuracy_record_classical[iternum] = 1 - error_num_classical/testing_num

        mean_accuracy_quantum_32[k] = np.mean(accuracy_record_quantum_32)
        mean_accuracy_quantum_1024[k] = np.mean(accuracy_record_quantum_1024)
        mean_accuracy_classical[k] = np.mean(accuracy_record_classical)

    return mean_accuracy_classical, mean_accuracy_quantum_32, mean_accuracy_quantum_1024