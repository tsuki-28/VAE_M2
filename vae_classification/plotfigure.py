from vae_classification.vae_classify import m2,test_data,test_set,alpha,cuda
from vae_classification.vae_m2 import FFNN
import torch
import numpy as np
from collections import defaultdict
import pandas as pd
def plot_test_M2(model, dataloader, alpha, cuda_available: bool, test_data: defaultdict, verbose=True):
    import torch.nn.functional as F
    # init
    running_loss = 0
    correct = 0
    total = 0
    loss_class = 0
    saved = False
    matix=[]
    with torch.no_grad():
        model.eval()
        for i, (input, target) in enumerate(dataloader):
            if cuda_available:
                model = model.cuda()
                input = input.cuda()
                target = target.cuda()
            input=torch.Tensor.float(input)
            input=input.view(input.size(0), 2, 500)
            # loss_lab, _, _, _ = model(input, target)

            output = model(input)
            classifier_loss = F.cross_entropy(output, target)

            loss_class += classifier_loss.item()

            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum()

            # running_loss += loss.item()
            matix.append([target.numpy(),predicted.numpy()])
            matix=matix[0]
            matix=np.array(matix)
        pd.DataFrame(matix).round(2).to_csv('./matrix_antenna1.csv',header=False,index=False)
        try:
            max_acc = max(test_data["test_accuracy"])
        except:
            max_acc = 0
        i=i+1
        # test_data["Tot_loss"] += [100*running_loss /(i+1) ]
        test_data["classifier_loss"] += [100*loss_class /(i+1) ]
        test_data["test_accuracy"] += [100 * correct.true_divide(total).item()]

        current_acc = test_data["test_accuracy"][-1]
        if current_acc >= max_acc:
            # torch.save(model.classifier.state_dict(), "./state_dict_classifier.pt")
            max_acc = current_acc
            saved = True

    if verbose:
        print("Test :")
        print(
            "Classifier loss : {}, Classifier accuracy : {}".format( round(test_data["classifier_loss"][-1],3),
                                                                     round(test_data["test_accuracy"][-1],2)))
        if saved:
            print(f"Saved Checkpoint with accuracy {max_acc}")

if __name__ == '__main__':
    m2=m2.classifier
    print('------------------分类器-------------------------')
    print(m2)
    m2.load_state_dict(torch.load('../vae_classification/state_dict_classifier_los_55.pt'))
    p_data=plot_test_M2(m2,test_set,alpha,cuda,test_data)
    print('123')