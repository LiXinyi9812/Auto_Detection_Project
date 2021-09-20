import torch
import numpy as np
from sklearn import metrics
from sklearn.metrics import fbeta_score
from torch import optim, nn
import visdom
import torchvision
from torch.utils.data import DataLoader

from RLBD import RLBD
from resnet import ResNet18

batchsz = 256
lr = 1e-3
epochs = 10

torch.manual_seed(1234)

train_db = RLBD('RLBD', 224, mode='train')
val_db = RLBD('RLBD', 224, mode='val')
test_db = RLBD('RLBD', 224, mode='test')
train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True,
                          num_workers=4)
val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=2)
test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=2)

viz = visdom.Visdom()


def evalute(model, loader):
    model.eval()
    correct = 0
    total = len(loader.dataset)

    for x,y in loader:
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
        correct += torch.eq(pred, y).sum().float().item()

    return correct / total

def F1_score(model, loader):
    model.eval()

    #total = len(loader.dataset)
    for x, y in loader:
        with torch.no_grad():
            logits = model(x)#logits是一个4*10的Tensor，可以理解成4张图片，每张图片有10维向量的预测，然后对每一张照片的输出值执行softmax和argmax（dim=1），得出预测标签，与真实标签比较，得出准确率。
            pred = logits.argmax(dim=1)#dim=1意味着第二个维度，也就是标签值

    f1s = metrics.f1_score(y, pred, average='weighted')
    return f1s

# def F2_score(model, loader):
#     model.eval()
#
#     #total = len(loader.dataset)
#     for x, y in loader:
#         with torch.no_grad():
#             logits = model(x)#logits是一个4*10的Tensor，可以理解成4张图片，每张图片有10维向量的预测，然后对每一张照片的输出值执行softmax和argmax（dim=1），得出预测标签，与真实标签比较，得出准确率。
#             pred = logits.argmax(dim=1)#dim=1意味着第二个维度，也就是标签值
#
#     f2s = fbeta_score(y, pred, beta=2, average='weighted')
#     return f2s



#可以换个loss函数（是否是针对图片数据集的？）
#def focal_loss(labels, logits, gamma, alpha):
#labels = tf.cast(labels, tf.float32)
#probs = tf.sigmoid(logits)
#ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
#alpha_t = tf.ones_like(logits) * alpha
#alpha_t = tf.where(labels > 0, alpha_t, 1.0 - alpha_t)
#probs_t = tf.where(labels > 0, probs, 1.0 - probs)
#focal_matrix = alpha_t * tf.pow((1.0 - probs_t), gamma)

#loss = focal_matrix * ce_loss
#loss = tf.reduce_mean(loss)
#return loss
#然后再改下面的crossentropy

#focal loss 多分类（没搞定）
# def criterion(y_pred, y_true, weight=None, alpha=0.25, gamma=2):
#     sigmoid_p = nn.Sigmoid(y_pred)
#     zeros = torch.zeros_like(sigmoid_p)
#     pos_p_sub = torch.where(y_true > zeros,y_true - sigmoid_p,zeros)
#     neg_p_sub = torch.where(y_true > zeros,zeros,sigmoid_p)
#     per_entry_cross_ent = -alpha * (pos_p_sub ** gamma) * torch.log(torch.clamp(sigmoid_p,1e-8,1.0))-(1-alpha)*(neg_p_sub ** gamma)*torch.log(torch.clamp(1.0-sigmoid_p,1e-8,1.0))
#     return per_entry_cross_ent.sum()


def main():

    model = ResNet18(4)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss()

    best_F1,best_acc, best_epoch = 0, 0, 0
    global_step = 0
    viz.line([0], [-1], win='loss', opts=dict(title='loss', xlabel = 'Step', ylabel = 'Loss'))
    viz.line([0], [-1], win='val_acc', opts=dict(title='val_acc', xlabel = 'epoch', ylabel = 'Accuracy'))
    viz.line([0], [-1], win='val_F1', opts=dict(title='val_F1', xlabel = 'epoch', ylabel = 'F1 score'))


    for epoch in range(epochs):
        for step, (x,y) in enumerate(train_loader):
            model.train()
            logits = model(x)
            loss = criteon(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            viz.line([loss.item()], [global_step], win='loss', update='append')
            global_step += 1

        if epoch % 1 == 0:

            val_F1 = F1_score(model, val_loader)
            val_acc = evalute(model, val_loader)
            if val_acc > best_acc:
                best_acc = val_acc
                viz.line([val_acc], [global_step], win='val_acc', update='append')
            if val_F1 > best_F1:
                best_epoch = epoch
                best_F1 = val_F1

                torch.save(model.state_dict(), 'best.mdl')

                viz.line([val_F1], [global_step], win='val_F1', update='append')





    print('best F1_score', best_F1, 'best epoch', best_epoch)
    print('acc', best_acc)


    model.load_state_dict(torch.load('best.mdl'))
    print('loaded from ckpt!')

    test_F = F1_score(model, test_loader)
    test_acc = evalute(model, val_loader)
    print('test F1-score:', test_F)
    print('test acc:', test_acc)








if __name__ == '__main__':
    main()