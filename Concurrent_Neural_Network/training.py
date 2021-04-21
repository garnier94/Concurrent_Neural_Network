from Concurrent_Neural_Network.submodel import Multi_layer_feed_forward_model
from Concurrent_Neural_Network.models import Concurrent_Module
import pdb

architectures = [[2],[4],[8],[2,2],[4,2],[4,4],[8,4],[2,2,2],[4,2,2],[8,4,2],[8,4,2]]

def eval_architechures(data_train_loader,data_valid_loader,data_test_loader, n_input, output_file = None, verbose=False, **kwargs):
    ls_mape =[]
    string_arch_list = []
    for arch in architectures:
        string_arch = ':'.join(list(map(str, arch)))
        string_arch_list.append(string_arch)
        if verbose:
            print('Modele :' + string_arch)
        model, sub_model = direct_Concurrent_NN_training(data_train_loader,data_valid_loader,arch,n_input,verbose=verbose, **kwargs)
        model_error = float(model.eval(data_test_loader, return_MAPE=True))
        if output_file is not None:
            output_file.write('Modèle :' + string_arch + "\n")
            output_file.write('MAPE modèle Test: %.3f \n' % model_error)
        ls_mape.append(model_error)
    if verbose :
        best_model_index = ls_mape.index(min(ls_mape))
        print('Best model : ' + string_arch_list[best_model_index])
        print("Best MAPE :  %.2f" % ls_mape[best_model_index])
    return model_error


def direct_Concurrent_NN_training(data_train_loader, data_valid_loader, architecture, n_input, verbose = True, **kwargs):
    epochs = kwargs.get('epoch',100)
    learning_rate = kwargs.get('learning_rate', 0.005)
    nb_max_attempt = kwargs.get('nb_max_attempt', 10) # Max number of training attempt
    nb_attempt = 0
    while nb_attempt < nb_max_attempt:
        submodel = Multi_layer_feed_forward_model(n_input, architecture)
        model= Concurrent_Module(submodel, sum_factor=1, loss='L1', learning_rate =learning_rate)
        mape_ini = model.eval(data_valid_loader,return_MAPE=True)
        model.train(data_train_loader, eval_dataset=data_valid_loader,max_epochs=10, batch_print=10)
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