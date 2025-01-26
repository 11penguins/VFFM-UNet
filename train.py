from utils import *
from dataset.dataset import Datasets
from torch.utils.data import DataLoader
from model import Net
from engine import *


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    seed(3407)
    torch.cuda.empty_cache()

    # data_path =
    # epoch_nums =
    # head_nums =
    # batch_size =

    train_dataset = Datasets(data_path, 'isic2018', train=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=0)
    val_dataset = Datasets(data_path, 'isic2018', train=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=False, num_workers=0, drop_last=True)

    model = Net(img_size=256, head_num=head_nums, depth=[8, 16, 24, 32, 48, 64], channel=3)
    model = model.cuda()

    optimizer = get_optimizer('AdamW', model)
    criterion = Loss(0.6, 0.4)
    scheduler = get_scheduler('CosineAnnealingLR', optimizer)
   
    print("#————————————Start train!————————————#")

    for epoch in range(1, epoch_nums):
        torch.cuda.empty_cache()

        loss_train, step = train_one_epoch(model, train_loader, optimizer, criterion, scheduler, step)
        torch.cuda.empty_cache()

        loss ,miou, all = val_one_epoch(model, val_loader, criterion, epoch)


    print("#————————————Start test!————————————#")

    # model.load_state_dict(torch.load()) choose the best model for testing
    # test_one_epoch(val_loader, model, criterion)

if __name__ == '__main__':
    main()