import matplotlib.pyplot as plt

def plot_result(predict,actual):    
    sample=[1,100,200,400,800]
    for i in sample:
        epoch = range(1,97)
        plt.figure(figsize=(15,10))
        plt.plot(epoch,predict[i,:],'red',marker='.' ,label='predict_value')
        plt.plot(epoch,actual[i,:],'blue', label='actual_value')
        plt.xlabel('epoch',size=15)
        plt.ylabel('value',size=15)
        plt.legend()
        plt.show()
        plt.clf()
        
def plot_loss(loss, val_loss ,savepath):
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.savefig(savepath+'.png')
    plt.show()