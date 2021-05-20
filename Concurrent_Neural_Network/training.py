from Concurrent_Neural_Network.submodel import Multi_layer_feed_forward_model
from Concurrent_Neural_Network.models import Concurrent_Module
import pdb

architectures = [[4],[8],[16],[32],[4,2],[4,4],[8,4],[16,8],[8,4,2],[16,8,4]]

def eval_architechures(data_train_loader,data_valid_loader,data_test_loader, n_input, output_file = None, verbose=False, **kwargs):
    list_error_model_valid=[]
    list_error_model_test=[]
    string_arch_list = []
    for arch in architectures:
        string_arch = ':'.join(list(map(str, arch)))
        string_arch_list.append(string_arch)
        if verbose:
            print('Modele :' + string_arch)
        model, sub_model = direct_Concurrent_NN_training(data_train_loader,data_valid_loader,arch,n_input,verbose=verbose, **kwargs)
        model_error_valid = float(model.eval(data_valid_loader, return_MAPE=True))
        model_error_test =  float(model.eval(data_test_loader, return_MAPE=True))
        list_error_model_valid.append(model_error_valid)
        list_error_model_test.append(model_error_test)
    best_model_index = list_error_model_valid.index(min(list_error_model_valid))
    if verbose:
        print('Best model : ' + string_arch_list[best_model_index])
        print("Best MAPE :  %.2f" % list_error_model_test[best_model_index])
    if output_file is not None:
        output_file.write('Best model NN: ' + string_arch_list[best_model_index] + '\n')
        output_file.write("Best MAPE :  %.2f \n" % list_error_model_test[best_model_index])



def direct_Concurrent_NN_training(data_train_loader, data_valid_loader, architecture, n_input, verbose = True, **kwargs):
    epochs = int(kwargs.get('epochs',100))
    early_stopping =  int(kwargs.get("early_stopping"))
    learning_rate = float(kwargs.get('learning_rate', 0.005))
    loss = kwargs.get('loss', 'L1')
    nb_max_attempt = kwargs.get('nb_max_attempt', 25) # Max number of training attempt
    nb_attempt = 0
    while nb_attempt < nb_max_attempt:
        submodel = Multi_layer_feed_forward_model(n_input, architecture)
        model= Concurrent_Module(submodel, sum_factor=1, loss=loss, learning_rate =learning_rate)
        mape_ini = model.eval(data_valid_loader,return_MAPE=True)
        model.train(data_train_loader, eval_dataset=data_valid_loader,max_epochs=10, batch_print=10, early_stopping=early_stopping)
        mape_end = model.eval(data_valid_loader,return_MAPE=True)
        if mape_ini - mape_end > 4 :
            break
        else:
            nb_attempt +=1
    if nb_attempt == nb_max_attempt:
        raise Exception("Too many attempt without increase of performance")

    batch_prt= (epochs //10 ) if verbose else (epochs+1)

    model.train(data_train_loader, eval_dataset=data_valid_loader,max_epochs=epochs, batch_print=batch_prt)
    if verbose:
        print("Training attempt : %s" %(nb_attempt+1))
    return model, submodel