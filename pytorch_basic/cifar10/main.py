import argparse
import training
import datasets
from tsne import tsne

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='CIFAR10 image classification')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--epoch', default=101, type=int, help='training epoch')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--l2', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--model_name', default='resnet18', type=str, help='model name')
    parser.add_argument('--pretrained', default=None, type=str, help='model path')
    parser.add_argument('--train', default='train', type=str, help='train and eval')
    args = parser.parse_args()
    print(args)

    # 데이터 불러오기
    trainloader, testloader = datasets.dataloader(args.batch_size)
    print('Completed loading your datasets.')

    # 모델 불러오기 및 학습하기
    learning = training.SupervisedLearning(trainloader, testloader, args.model_name, args.pretrained)

    if args.train == 'train':
        learning.train(args.epoch, args.lr, args.l2)
    else:
        train_acc = learning.eval(trainloader)
        test_acc = learning.eval(testloader)
        print(f' Train Accuracy: {train_acc}, Test Accuraccy: {test_acc}')

        # t-SNE graph
        tsne(testloader, args.model_name, args.pretrained)

