import os
import torch
import torch.utils
import torch.utils.data
import torchvision
from torch.utils.model_zoo import load_url
from torchvision import transforms
from cirtorch.networks.imageretrievalnet import init_network, extract_vectors, extract_vectors_by_arrays , extract_vectors_by_arrays2 , extract_db_array
from cirtorch.utils.general import get_data_root
from cirtorch.layers.pooling import MAC, SPoC, GeM, GeMmp, RMAC, Rpool
from cirtorch.datasets.datahelpers import im_resize
from solar_global.utils.networks import load_network
from PIL import Image
import torch.nn.functional as F
import cv2
import pickle
from tqdm import tqdm
import argparse
from custom_datasets.imagefolder import ImageFolderDataset

from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.reducers import ThresholdReducer
from pytorch_metric_learning.regularizers import LpRegularizer
from pytorch_metric_learning import losses, miners


PRETRAINED = {
    'rSfM120k-tl-resnet50-gem-w': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet50-gem-w-97bf910.pth',
    'rSfM120k-tl-resnet101-gem-w': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet101-gem-w-a155e54.pth',
    'rSfM120k-tl-resnet152-gem-w': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet152-gem-w-f39cada.pth',
    'gl18-tl-resnet50-gem-w': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet50-gem-w-83fdc30.pth',
    'gl18-tl-resnet101-gem-w': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet101-gem-w-a4d43db.pth',
    'gl18-tl-resnet152-gem-w': 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/gl18/gl18-tl-resnet152-gem-w-21278d5.pth',
}

parser = argparse.ArgumentParser(description='PyTorch CNN Image Retrieval Training')
parser.add_argument('--loss-margin', '-lm', metavar='LM', default=0.7, type=float,
                    help='loss margin: (default: 0.7)')
parser.add_argument('--use_solar', '-sl', action='store_true')
parser.add_argument('--use_rmac', '-rmac', action='store_true')
parser.add_argument('--train_folder', type=str, default='')
parser.add_argument('--val_folder', type=str, default='')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_epochs', type=int, default=32)
parser.add_argument('--optimizer', type=str, default='sgd')
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--save_ckpts', type=str, default='checkpoints')


def train_epoch(model, train_loader, optimizer, criterion, miner, device):
    model.train()
    running_loss = 0.0
    for idx, (image, label) in tqdm(enumerate(train_loader)):
        image, label = image.to(device), label.to(device)
        optimizer.zero_grad()
        outputs = model(image)
        hard_pairs = miner(outputs, label)
        # outputs = F.softmax(outputs, dim=1)
        loss = criterion(outputs, label, hard_pairs)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        #if idx > 2000:
         #   break
    epoch_loss = running_loss / idx#len(train_loader)

    return epoch_loss

def eval(model, loader, criterion, miner,device):
    model.eval()
    running_loss = 0
    for idx, (image, label) in enumerate(loader):
        image, label = image.to(device), label.to(device)
        outputs = model(image)
        if miner is not None:
            hard_pairs = miner(outputs, label)
            # outputs = F.softmax(outputs, dim=1)
            loss = criterion(outputs, label, hard_pairs)
        else:
            loss = criterion(outputs, label)
        running_loss += loss.item()
        # _, predicted = torch.max(outputs.data, 1)
        # total += label.size(0)
    
        # correct += (predicted == label).cpu().sum().item()

    return running_loss/len(loader)#, correct / total

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = parser.parse_args()
    network = 'rSfM120k-tl-resnet101-gem-w'
    state = load_url(PRETRAINED[network], model_dir=os.path.join(get_data_root(), 'networks'))
    if args.use_solar:
        if torch.cuda.is_available():
            state = torch.load('data/networks/model_best.pth.tar')
        else:
            state = torch.load('data/networks/model_best.pth.tar', map_location=torch.device('cpu'))
    net_params = {}
    net_params['architecture'] = state['meta']['architecture']
    net_params['pooling'] = state['meta']['pooling']
    net_params['local_whitening'] = state['meta'].get('local_whitening', False)
    net_params['regional'] = state['meta'].get('regional', False)
    net_params['whitening'] = state['meta'].get('whitening', True)

    net_params['mean'] = state['meta']['mean']
    net_params['std'] = state['meta']['std']
    net_params['pretrained'] = False
    if args.use_solar:
        net = load_network('model_best.pth.tar')
    else:
        net = init_network(net_params)
    net.load_state_dict(state['state_dict'])

    if args.use_rmac:
        net.pool = RMAC(3)
    
    net = net.to(device=device)

    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    try:
        dataset = ImageFolderDataset(args.train_folder, transform=transform)
        # val_dataset = ImageFolderDatasetFolder(args.val_folder, transform=transform)
        train_size = int(0.9 * len(dataset))
        valid_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])
        
        print("Number of train samples: ", len(train_dataset))
        print("Number of vaild samples: ", len(val_dataset))
        print("Detected Classes TrainDS are: ", len(dataset.class_to_idx.keys()))
        # print("Detected Classes ValDS are: ", len(val_dataset.class_to_idx.keys()))
        
        miner = miners.MultiSimilarityMiner()
        criterion = losses.TripletMarginLoss(margin=5.)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=8)
        valid_loader  = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

        print('Infor loader: ')
        print('Train_loader: {0}, valid_loader: {1}'.format(len(train_loader), len(valid_loader)))
    except:
        print('Missing data!!!')
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    best_eval_loss = 20
    for epoch in range(args.num_epochs):
        print(f'EPOCH {epoch+1}: ')
        train_loss = train_epoch(net,
                                train_loader,
                                optimizer,
                                criterion,
                                miner,
                                device
                            )
        eval_loss= eval(net, valid_loader, criterion, None,device)
        exp_lr_scheduler.step()
        if (epoch + 1) % 2 != 0:
            continue
        torch.save(net.state_dict(), os.path.join(args.save_ckpts, 'epochs-{}-train-{}-val-{}.pt'.format(epoch, round(train_loss, 2), round(eval_loss, 2))))
        print(f'eval_loss from {best_eval_loss} --> {eval_loss}: Saved model!!')
        best_eval_loss = eval_loss

        # print(f"Epoch [{epoch + 1}/{cfg.EPOCHS}], Train loss: {train_loss:.4f}, Eval loss: {eval_loss:.4f}, eval_loss: {eval_loss:.4f}")

    print('TRAIN FINISH!!!')
    # print('Test...')
    # test_loss = eval(model,test_loader, criterion,  None, device)
    # print(f"Test loss: {test_loss:.4f}, Test_acc: {test_acc:.4f}")

    # torch.save(model.state_dict(), last_path)

if __name__ == '__main__':
    main()