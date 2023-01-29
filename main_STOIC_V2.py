"""
    IMPORTING LIBS
"""
import numpy as np
import os
import time
import random
import glob
import argparse, json
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
import wandb


class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


"""
    IMPORTING CUSTOM MODULES/METHODS
"""

from nets.load_net import gnn_model  # import GNNs
from data.data import LoadData  # import dataset

"""
    GPU Setup
"""


def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:', torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device


"""
    VIEWING MODEL CONFIG AND PARAMS
"""


def view_model_param(MODEL_NAME, net_params, tresh):
    model = gnn_model(MODEL_NAME, net_params, tresh)
    total_param = 0
    print("MODEL DETAILS:\n")
    # print(model)
    for param in model.parameters():
        # print(param.data.size())
        total_param += np.prod(list(param.data.size()))
    print('MODEL/Total parameters:', MODEL_NAME, total_param)
    return total_param


"""
    TRAINING CODE
"""


def train_val_pipeline(MODEL_NAME, DATASET_NAME, params, net_params, dirs, tresh):
    avg_test_acc = []
    avg_train_acc = []
    avg_val_acc = []
    avg_val_loss = []
    avg_epochs = []

    t0 = time.time()
    per_epoch_time = []

    dataset = LoadData(DATASET_NAME)

    if MODEL_NAME in ['GCN', 'GAT']:
        if net_params['self_loop']:
            print("[!] Adding graph self-loops for GCN/GAT models (central node trick).")
            dataset._add_self_loops()

    root_log_dir, root_ckpt_dir, write_file_name, write_config_file = dirs
    device = net_params['device']

    # Write the network and optimization hyper-parameters in folder config/
    with open(write_config_file + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n\nTotal Parameters: {}\n\n""".format(
            DATASET_NAME, MODEL_NAME, params, net_params, net_params['total_param']))

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        if params['cross_val']:
            for split_number in range(4):
                t0_split = time.time()
                log_dir = os.path.join(root_log_dir, "RUN_" + str(split_number))
                writer = SummaryWriter(log_dir=log_dir)
                # setting seeds
                random.seed(params['seed'])
                np.random.seed(params['seed'])
                torch.manual_seed(params['seed'])
                if device == 'cuda':
                    torch.cuda.manual_seed(params['seed'])

                print("RUN NUMBER: ", split_number)
                trainset, valset, testset = dataset.train[split_number], dataset.val[split_number], dataset.test[
                    split_number]
                print("Training Graphs: ", len(trainset))
                print("Validation Graphs: ", len(valset))
                print("Test Graphs: ", len(testset))
                print("Number of Classes: ", net_params['n_classes'])

                model = gnn_model(MODEL_NAME, net_params, tresh)
                model = model.to(device)
                optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                 factor=params['lr_reduce_factor'],
                                                                 patience=params['lr_schedule_patience'],
                                                                 verbose=True, min_lr=params['min_lr'])

                epoch_train_losses, epoch_val_losses = [], []
                epoch_train_accs, epoch_val_accs = [], []

                # batching exception for Diffpool
                drop_last = True if MODEL_NAME == 'DiffPool' else False
                # drop_last = False

                # import train functions for all other GCNs
                from train.train_STOIC import train_epoch_sparse as train_epoch, \
                    evaluate_network_sparse as evaluate_network

                train_loader = DataLoader(trainset, batch_size=params['batch_size'], shuffle=True, drop_last=drop_last,
                                          collate_fn=dataset.collate)
                val_loader = DataLoader(valset, batch_size=params['batch_size'], shuffle=False, drop_last=drop_last,
                                        collate_fn=dataset.collate)
                test_loader = DataLoader(testset, batch_size=params['batch_size'], shuffle=False, drop_last=drop_last,
                                         collate_fn=dataset.collate)

                with tqdm(range(params['epochs'])) as t:
                    for epoch in t:

                        t.set_description('Epoch %d' % epoch)

                        start = time.time()
                        epoch_train_loss, epoch_train_acc, optimizer, train_comb_loss, A_train, _ = train_epoch(model, optimizer, device,
                                                                                   train_loader, epoch)

                        epoch_val_loss, epoch_val_acc, val_comb_loss, A_val, _ = evaluate_network(model, device, val_loader, epoch)

                        epoch_test_loss, epoch_test_acc, test_comb_loss, A_test, _ = evaluate_network(model, device, test_loader, epoch)

                        epoch_train_losses.append(epoch_train_loss)
                        epoch_val_losses.append(epoch_val_loss)
                        epoch_train_accs.append(epoch_train_acc)
                        epoch_val_accs.append(epoch_val_acc)
                        current_epoch = {'val_acc': epoch_val_acc,
                                         'val_loss': epoch_val_loss,
                                         'train_acc': epoch_train_acc,
                                         'train_loss': epoch_train_loss,
                                         'test_acc': epoch_test_acc,
                                         'test_loss': epoch_test_loss,
                                         'epoch': epoch}

                        if epoch == 0:
                            no_change = 0
                            save = True
                            best_epoch = {'val_acc': epoch_val_acc,
                                          'val_loss': epoch_val_loss,
                                          'train_acc': epoch_train_acc,
                                          'train_loss': epoch_train_loss,
                                          'test_acc': epoch_test_acc,
                                          'test_loss': epoch_test_loss,
                                          'epoch': epoch}
                        else:
                            if not (current_epoch['val_acc'] >= best_epoch['val_acc']):
                                no_change += 1
                                save = False
                            else:
                                best_epoch = current_epoch
                                no_change = 0
                                save = True
                        wandb.define_metric('val_acc', summary="mean")
                        wandb.define_metric('val_loss', summary="mean")
                        wandb.define_metric('train_acc', summary="mean")
                        wandb.define_metric('train_loss', summary="mean")
                        wandb.define_metric('test_acc', summary="mean")
                        wandb.log(best_epoch, step=epoch)
                        """wandb.init(project='PrognosisGraph', group = 'cross_val' , job_type="fold_".format(str(split_number)))
                        wandb.log(best_epoch, step = epoch)
                        wandb.finish()"""
                        #wandb.watch(model)
                        writer.add_scalar('train/_loss', epoch_train_loss, epoch)
                        writer.add_scalar('val/_loss', epoch_val_loss, epoch)
                        writer.add_scalar('train/_acc', epoch_train_acc, epoch)
                        writer.add_scalar('val/_acc', epoch_val_acc, epoch)
                        writer.add_scalar('test/_acc', epoch_test_acc, epoch)
                        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

                        epoch_train_acc = 100. * epoch_train_acc
                        epoch_test_acc = 100. * epoch_test_acc
                        epoch_val_acc = 100. * epoch_val_acc

                        t.set_postfix(time=time.time() - start, lr=optimizer.param_groups[0]['lr'],
                                      train_loss=epoch_train_loss, val_loss=epoch_val_loss,
                                      train_acc=epoch_train_acc, val_acc=epoch_val_acc,
                                      test_acc=epoch_test_acc)

                        per_epoch_time.append(time.time() - start)

                        # Saving checkpoint
                        if save:
                            ckpt_dir = os.path.join(root_ckpt_dir, "RUN_" + str(split_number))
                            if not os.path.exists(ckpt_dir):
                                os.makedirs(ckpt_dir)
                            torch.save(model.state_dict(), '{}.pkl'.format(ckpt_dir + "/epoch_" + str(epoch)))

                            files = glob.glob(ckpt_dir + '/*.pkl')
                            for file in files:
                                epoch_nb = file.split('_')[-1]
                                epoch_nb = int(epoch_nb.split('.')[0])
                                if epoch_nb < epoch - 1:
                                    os.remove(file)

                        scheduler.step(epoch_val_loss)
                        if no_change > params['lr_schedule_patience']*2:
                            print('Best epoch since since {} epochs '.format(str(no_change)))
                            break
                        if optimizer.param_groups[0]['lr'] <= params['min_lr']:
                            print('LR min')
                            break

                        # Stop training after params['max_time'] hours
                        if time.time() - t0_split > params[
                            'max_time'] * 3600 / 5:  # Dividing max_time by 5, since there are 5 runs
                            print('-' * 89)
                            print(
                                "Max_time for one train-val-test split experiment elapsed {:.3f} hours, so stopping".format(
                                    params['max_time'] / 10))
                            break

                """_, test_acc = evaluate_network(model, device, test_loader, epoch)
                _, train_acc = evaluate_network(model, device, train_loader, epoch)
                _, val_acc = evaluate_network(model, device, val_loader, epoch)"""

                avg_test_acc.append(best_epoch['test_acc'])
                avg_train_acc.append(best_epoch['train_acc'])
                avg_val_acc.append(best_epoch['val_acc'])
                avg_val_loss.append(best_epoch['val_loss'])
                avg_epochs.append(epoch)

                '''plt.figure()
                plt.plot(epoch_train_accs)
                plt.plot(epoch_val_accs)
                plt.legend(['train', 'val'])
                plt.ylabel('acc')
                plt.xlabel('epoch')
                plt.title('Training summary fold '.format(split_number))
                plt.savefig(log_dir +'/acc')
                plt.figure()
                plt.plot(epoch_train_losses)
                plt.plot(epoch_val_losses)
                plt.legend(['train', 'val'])
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.title('Training summary fold '.format(split_number))
                plt.savefig(log_dir + '/loss')'''

                print("Test Accuracy [BEST EPOCH]: {:.4f}".format(best_epoch['test_acc'] * 100))
                print("Val Accuracy [BEST EPOCH]: {:.4f}".format(best_epoch['val_acc'] * 100))
                print("Train Accuracy [LAST EPOCH]: {:.4f}".format(best_epoch['train_acc'] * 100))
        else:
            split_number = params['split_num']
            t0_split = time.time()
            log_dir = os.path.join(root_log_dir, "RUN_" + str(split_number))
            writer = SummaryWriter(log_dir=log_dir)
            # setting seeds
            random.seed(params['seed'])
            np.random.seed(params['seed'])
            torch.manual_seed(params['seed'])
            """if device.type == 'cuda':
                torch.cuda.manual_seed(params['seed'])"""

            print("RUN NUMBER: ", split_number)
            trainset, valset, testset = dataset.train[split_number], dataset.val[split_number], dataset.test[
                split_number]
            print("Training Graphs: ", len(trainset))
            print("Validation Graphs: ", len(valset))
            print("Test Graphs: ", len(testset))
            print("Number of Classes: ", net_params['n_classes'])

            model = gnn_model(MODEL_NAME, net_params, tresh)
            model = model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                             factor=params['lr_reduce_factor'],
                                                             patience=params['lr_schedule_patience'],
                                                             verbose=True, min_lr=params['min_lr'])

            epoch_train_losses, epoch_val_losses = [], []
            epoch_train_accs, epoch_val_accs = [], []

            # batching exception for Diffpool
            drop_last = True if MODEL_NAME == 'DiffPool' else False
            # drop_last = False

            # import train functions for all other GCNs
            from train.train_STOIC import train_epoch_sparse as train_epoch, \
                evaluate_network_sparse as evaluate_network

            train_loader = DataLoader(trainset, batch_size=params['batch_size'], shuffle=True, drop_last=drop_last,
                                      collate_fn=dataset.collate)
            val_loader = DataLoader(valset, batch_size=params['batch_size'], shuffle=False, drop_last=drop_last,
                                    collate_fn=dataset.collate)
            test_loader = DataLoader(testset, batch_size=params['batch_size'], shuffle=False, drop_last=drop_last,
                                     collate_fn=dataset.collate)

            with tqdm(range(params['epochs'])) as t:
                for epoch in t:

                    t.set_description('Epoch %d' % epoch)

                    start = time.time()
                    epoch_train_loss, epoch_train_acc, optimizer, train_comb_loss, A_train, feat = train_epoch(model,
                                                                                                         optimizer,
                                                                                                         device,
                                                                                                         train_loader,
                                                                                                         epoch)

                    epoch_val_loss, epoch_val_acc, val_comb_loss, A_val, feat = evaluate_network(model, device, val_loader,
                                                                                           epoch)

                    epoch_test_loss, epoch_test_acc, test_comb_loss, A_test, feat = evaluate_network(model, device,
                                                                                               test_loader, epoch)


                    current_epoch = {'val_acc': epoch_val_acc,
                                     'val_loss': epoch_val_loss,
                                     'train_acc': epoch_train_acc,
                                     'train_loss': epoch_train_loss,
                                     'test_acc': epoch_test_acc,
                                     'test_loss': epoch_test_loss,
                                     'epoch': epoch,
                                     'test': epoch_test_acc,
                                     'sparsity_train_loss': train_comb_loss['sparsity'],
                                     'connectivity_train_loss': train_comb_loss['connectivity'],
                                     'smooth_train_loss': train_comb_loss['smooth'],
                                     'grl_loss': val_comb_loss['grl'],
                                     'gnn_loss': val_comb_loss['gnn'],
                                     'tr_loss': val_comb_loss['transformer'],
                                     'tr_acc': test_comb_loss['tr_acc']}
                    if epoch == 0:
                        no_change = 0
                        save = True
                        best_epoch = current_epoch
                    else:
                        if not (current_epoch['val_acc'] >= best_epoch['val_acc']):
                            no_change += 1
                            save = False
                        else:
                            best_epoch = current_epoch
                            no_change = 0
                            save = True
                    epoch_train_losses.append(current_epoch['gnn_loss'].detach().numpy())
                    epoch_val_losses.append(current_epoch['tr_loss'].detach().numpy())
                    epoch_train_accs.append(current_epoch['test_acc'])
                    epoch_val_accs.append(current_epoch['tr_acc'])
                    wandb.log(current_epoch)
                    img = A_train
                    min = float(img.min())
                    max = float(img.max())
                    img -= img.min(0, keepdim=True)[0]
                    img /= img.max(0, keepdim=True)[0]
                    img = img.detach().numpy()

                    #img = Image.fromarray(np.uint8(cm.gist_earth(img) * 255))
                    images = wandb.Image(img, caption="A epoch :{} min : {} max : {}"
                                         .format(str(epoch), str(round(min, 4)), str(round(max, 4))))
                    wandb.log({"A_train": images})
                    wandb.watch(model)
                    writer.add_scalar('train/_loss', epoch_train_loss, epoch)
                    writer.add_scalar('val/_loss', epoch_val_loss, epoch)
                    writer.add_scalar('train/_acc', epoch_train_acc, epoch)
                    writer.add_scalar('val/_acc', epoch_val_acc, epoch)
                    writer.add_scalar('test/_acc', epoch_test_acc, epoch)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

                    epoch_train_acc = 100. * epoch_train_acc
                    epoch_test_acc = 100. * epoch_test_acc
                    epoch_val_acc = 100. * epoch_val_acc

                    t.set_postfix(time=time.time() - start, lr=optimizer.param_groups[0]['lr'],
                                  train_loss=epoch_train_loss, val_loss=epoch_val_loss,
                                  train_acc=epoch_train_acc, val_acc=epoch_val_acc,
                                  test_acc=epoch_test_acc)

                    per_epoch_time.append(time.time() - start)

                    # Saving checkpoint
                    if save:
                        ckpt_dir = os.path.join(root_ckpt_dir, "RUN_" + str(split_number))
                        if not os.path.exists(ckpt_dir):
                            os.makedirs(ckpt_dir)
                        torch.save(model.state_dict(), '{}.pkl'.format(ckpt_dir + "/epoch_" + str(epoch)))

                        files = glob.glob(ckpt_dir + '/*.pkl')
                        for file in files:
                            epoch_nb = file.split('_')[-1]
                            epoch_nb = int(epoch_nb.split('.')[0])
                            if epoch_nb < epoch - 1:
                                os.remove(file)

                    scheduler.step(epoch_val_loss)
                    """if no_change > 2*params['lr_schedule_patience']+3:
                        print('Best epoch since since {} epochs '.format(str(no_change)))
                        break"""
                    if optimizer.param_groups[0]['lr'] <= params['min_lr']:
                        print('LR min')
                        break

                    # Stop training after params['max_time'] hours
                    if time.time() - t0_split > params[
                        'max_time'] * 3600 / 5:  # Dividing max_time by 5, since there are 5 runs
                        print('-' * 89)
                        print(
                            "Max_time for one train-val-test split experiment elapsed {:.3f} hours, so stopping".format(
                                params['max_time'] / 10))
                        break
                plt.figure()
                plt.plot(epoch_train_losses)
                plt.plot(epoch_val_losses)
                plt.legend(['gnn', 'transformer'])
                plt.ylabel('loss')
                plt.xlabel('epoch')
                wandb.log({"chart": wandb.Image(plt)})
                plt.figure()
                plt.plot(epoch_train_accs)
                plt.plot(epoch_val_accs)
                plt.legend(['gnn', 'transformer'])
                plt.ylabel('acc')
                plt.xlabel('epoch')
                wandb.log({"chart": wandb.Image(plt)})

            """_, test_acc = evaluate_network(model, device, test_loader, epoch)
            _, train_acc = evaluate_network(model, device, train_loader, epoch)
            _, val_acc = evaluate_network(model, device, val_loader, epoch)"""
            avg_test_acc.append(best_epoch['test_acc'])
            avg_train_acc.append(best_epoch['train_acc'])
            avg_val_acc.append(best_epoch['val_acc'])
            avg_epochs.append(epoch)

            """plt.figure()
            plt.plot(epoch_train_accs)
            plt.plot(epoch_val_accs)
            plt.legend(['train', 'val'])
            plt.ylabel('acc')
            plt.xlabel('epoch')
            plt.title('Training summary fold '.format(split_number))
            plt.savefig(log_dir + '/acc')
            plt.figure()
            plt.plot(epoch_train_losses)
            plt.plot(epoch_val_losses)
            plt.legend(['train', 'val'])
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.title('Training summary fold '.format(split_number))
            plt.savefig(log_dir + '/loss')"""

            print("Test Accuracy [BEST EPOCH]: {:.4f}".format(best_epoch['test_acc'] * 100))
            print("Val Accuracy [BEST EPOCH]: {:.4f}".format(best_epoch['val_acc'] * 100))
            print("Train Accuracy [LAST EPOCH]: {:.4f}".format(best_epoch['train_acc'] * 100))


    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')
    print("TOTAL TIME TAKEN: {:.4f}hrs".format((time.time() - t0) / 3600))
    print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))

    # Final test accuracy value averaged over 5-fold
    print("""\n\n\nFINAL RESULTS\n\nTEST ACCURACY averaged: {:.4f} with s.d. {:.4f}""".format(
        np.mean(np.array(avg_test_acc)) * 100, np.std(avg_test_acc) * 100))
    print("\nAll splits Test Accuracies:\n", avg_test_acc)
    print("""\n\n\nFINAL RESULTS\n\nVAL ACCURACY averaged: {:.4f} with s.d. {:.4f}""".format(
        np.mean(np.array(avg_val_acc)) * 100, np.std(avg_val_acc) * 100))
    print("\nAll splits Train Accuracies:\n", avg_val_acc)
    print("""\n\n\nFINAL RESULTS\n\nTRAIN ACCURACY averaged: {:.4f} with s.d. {:.4f}""".format(
        np.mean(np.array(avg_train_acc)) * 100, np.std(avg_train_acc) * 100))
    print("\nAll splits Train Accuracies:\n", avg_train_acc)

    writer.close()

    """
        Write the results in out/results folder
    """
    with open(write_file_name + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n{}\n\nTotal Parameters: {}\n\n
    FINAL RESULTS\nTEST ACCURACY averaged: {:.3f}\n with test acc s.d. {:.3f}\nTRAIN ACCURACY averaged: {:.3f}\n with train s.d. {:.3f}\n\n
    Convergence Time (Epochs): {:.3f}\nTotal Time Taken: {:.3f} hrs\nAverage Time Per Epoch: {:.3f} s\n\n\nAll Splits Test Accuracies: {}\n\nAll Splits Train Accuracies: {}""" \
                .format(DATASET_NAME, MODEL_NAME, params, net_params, model, net_params['total_param'],
                        np.mean(np.array(avg_test_acc)) * 100, np.std(avg_test_acc) * 100,
                        np.mean(np.array(avg_train_acc)) * 100, np.std(avg_train_acc) * 100,
                        np.mean(np.array(avg_epochs)),
                        (time.time() - t0) / 3600, np.mean(per_epoch_time), avg_test_acc, avg_train_acc))

    return avg_val_acc, avg_train_acc, avg_val_loss, avg_test_acc


def main():
    """
        USER CONTROLS
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=os.getcwd() + "/configs/STOIC_V2.json",
                        help="Please give a config.json file with training/model/data/param details")
    parser.add_argument('--gpu_id', help="Please give a value for gpu id")
    parser.add_argument('--model', help="Please give a value for model name")
    parser.add_argument('--dataset', help="Please give a value for dataset name")
    parser.add_argument('--out_dir', help="Please give a value for out_dir")
    parser.add_argument('--seed', help="Please give a value for seed")
    parser.add_argument('--epochs', help="Please give a value for epochs")
    parser.add_argument('--batch_size', help="Please give a value for batch_size")
    parser.add_argument('--init_lr', help="Please give a value for init_lr")
    parser.add_argument('--lr_reduce_factor', help="Please give a value for lr_reduce_factor")
    parser.add_argument('--lr_schedule_patience', help="Please give a value for lr_schedule_patience")
    parser.add_argument('--min_lr', help="Please give a value for min_lr")
    parser.add_argument('--weight_decay', help="Please give a value for weight_decay")
    parser.add_argument('--print_epoch_interval', help="Please give a value for print_epoch_interval")
    parser.add_argument('--L', help="Please give a value for L")
    parser.add_argument('--feature_dim', help="Please give a value for feature_dim")
    parser.add_argument('--similarity_dim', help="Please give a value for feature_dim")
    parser.add_argument('--mlp_dim', help="Please give a value for out_dim")
    parser.add_argument('--residual', help="Please give a value for residual")
    parser.add_argument('--edge_feat', help="Please give a value for edge_feat")
    parser.add_argument('--readout', help="Please give a value for readout")
    parser.add_argument('--kernel', help="Please give a value for kernel")
    parser.add_argument('--n_heads', help="Please give a value for n_heads")
    parser.add_argument('--n_transformers', help="Please give a value for n_heads")
    parser.add_argument('--cls_token', help="Please give a value for cls_token")
    parser.add_argument('--gated', help="Please give a value for gated")
    parser.add_argument('--in_feat_dropout', help="Please give a value for in_feat_dropout")
    parser.add_argument('--dropout', help="Please give a value for dropout")
    parser.add_argument('--graph_norm', help="Please give a value for graph_norm")
    parser.add_argument('--batch_norm', help="Please give a value for batch_norm")
    parser.add_argument('--sage_aggregator', help="Please give a value for sage_aggregator")
    parser.add_argument('--data_mode', help="Please give a value for data_mode")
    parser.add_argument('--num_pool', help="Please give a value for num_pool")
    parser.add_argument('--gnn_per_block', help="Please give a value for gnn_per_block")
    parser.add_argument('--embedding_dim', help="Please give a value for embedding_dim")
    parser.add_argument('--pool_ratio', help="Please give a value for pool_ratio")
    parser.add_argument('--linkpred', help="Please give a value for linkpred")
    parser.add_argument('--cat', help="Please give a value for cat")
    parser.add_argument('--self_loop', help="Please give a value for self_loop")
    parser.add_argument('--max_time', help="Please give a value for max_time")
    parser.add_argument('--pos_enc_dim', help="Please give a value for pos_enc_dim")
    parser.add_argument('--cross_val', help="If you want to make cross validation")
    parser.add_argument('--split_num', help="To select the fold for training")
    parser.add_argument('--threshold_value', help="To select the threshold to apply on edge")
    parser.add_argument('--similarity', help="To select the edge similarity representation")
    parser.add_argument('--type_of_thresh', help="To select of to apply threshold")

    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    # device
    if args.gpu_id is not None:
        config['gpu']['id'] = int(args.gpu_id)
        config['gpu']['use'] = True
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
    # model, dataset, out_dir
    if args.model is not None:
        MODEL_NAME = args.model
    else:
        MODEL_NAME = config['model']
    if args.dataset is not None:
        DATASET_NAME = args.dataset
    else:
        DATASET_NAME = config['dataset']
    # dataset = LoadData(DATASET_NAME)
    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = config['out_dir']
    # parameters
    params = config['params']
    if args.cross_val is not None:
        params['cross_val'] = bool(args.cross_val)
    if args.similarity is not None:
        params['similarity'] = args.similarity
    if args.type_of_thresh is not None:
        params['type_of_thresh'] = str(args.type_of_thresh)
    if args.threshold_value is not None:
        params['threshold_value'] = float(args.threshold_value)
    if args.split_num is not None:
        params['split_num'] = int(args.split_num)
    if args.seed is not None:
        params['seed'] = int(args.seed)
    if args.epochs is not None:
        params['epochs'] = int(args.epochs)
    if args.batch_size is not None:
        params['batch_size'] = int(args.batch_size)
    if args.init_lr is not None:
        params['init_lr'] = float(args.init_lr)
    if args.lr_reduce_factor is not None:
        params['lr_reduce_factor'] = float(args.lr_reduce_factor)
    if args.lr_schedule_patience is not None:
        params['lr_schedule_patience'] = int(args.lr_schedule_patience)
    if args.min_lr is not None:
        params['min_lr'] = float(args.min_lr)
    if args.weight_decay is not None:
        params['weight_decay'] = float(args.weight_decay)
    if args.print_epoch_interval is not None:
        params['print_epoch_interval'] = int(args.print_epoch_interval)
    if args.max_time is not None:
        params['max_time'] = float(args.max_time)
    # network parameters
    net_params = config['net_params']
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']
    net_params['batch_size'] = params['batch_size']
    if args.L is not None:
        net_params['L'] = int(args.L)
    if args.feature_dim is not None:
        net_params['feature_dim'] = int(args.feature_dim)
    if args.similarity_dim is not None:
        net_params['similarity_dim'] = int(args.similarity_dim)
    if args.mlp_dim is not None:
        net_params['mlp_dim'] = int(args.mlp_dim)
    if args.residual is not None:
        net_params['residual'] = True if args.residual == 'True' else False
    if args.edge_feat is not None:
        net_params['edge_feat'] = True if args.edge_feat == 'True' else False
    if args.cls_token is not None:
        net_params['cls_token'] = True if args.cls_token == 'True' else False
    if args.readout is not None:
        net_params['readout'] = args.readout
    if args.kernel is not None:
        net_params['kernel'] = int(args.kernel)
    if args.n_heads is not None:
        net_params['n_heads'] = int(args.n_heads)
    if args.n_transformers is not None:
        net_params['n_transformers'] = int(args.n_transformers)
    if args.gated is not None:
        net_params['gated'] = True if args.gated == 'True' else False
    if args.in_feat_dropout is not None:
        net_params['in_feat_dropout'] = float(args.in_feat_dropout)
    if args.dropout is not None:
        net_params['dropout'] = float(args.dropout)
    if args.graph_norm is not None:
        net_params['graph_norm'] = True if args.graph_norm == 'True' else False
    if args.batch_norm is not None:
        net_params['batch_norm'] = True if args.batch_norm == 'True' else False
    if args.sage_aggregator is not None:
        net_params['sage_aggregator'] = args.sage_aggregator
    if args.data_mode is not None:
        net_params['data_mode'] = args.data_mode
    if args.num_pool is not None:
        net_params['num_pool'] = int(args.num_pool)
    if args.gnn_per_block is not None:
        net_params['gnn_per_block'] = int(args.gnn_per_block)
    if args.embedding_dim is not None:
        net_params['embedding_dim'] = int(args.embedding_dim)
    if args.pool_ratio is not None:
        net_params['pool_ratio'] = float(args.pool_ratio)
    if args.linkpred is not None:
        net_params['linkpred'] = True if args.linkpred == 'True' else False
    if args.cat is not None:
        net_params['cat'] = True if args.cat == 'True' else False
    if args.self_loop is not None:
        net_params['self_loop'] = True if args.self_loop == 'True' else False
    if args.pos_enc_dim is not None:
        net_params['pos_enc_dim'] = int(args.pos_enc_dim)

    net_params['in_dim'] = 1024  # node_dim (feat is an integer)
    net_params['n_classes'] = 2

    root_log_dir = out_dir + 'logs/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(
        config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    root_ckpt_dir = out_dir + 'checkpoints/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(
        config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_file_name = out_dir + 'results/result_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(
        config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_config_file = out_dir + 'configs/config_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(
        config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    dirs = root_log_dir, root_ckpt_dir, write_file_name, write_config_file

    if not os.path.exists(out_dir + 'results'):
        os.makedirs(out_dir + 'results')

    if not os.path.exists(out_dir + 'configs'):
        os.makedirs(out_dir + 'configs')

    def Merge(dict1, dict2):
        res = {**dict1, **dict2}
        return res

    params = Merge(params, net_params)
    """os.environ["WANDB_API_KEY"] = 'f6a1a7a209231118c99d5b6078fc26a2941ce3c3'
    wandb.login()"""

    wandb.init(config=params, allow_val_change=True)#, group = 'cross_val' , job_type="optimize"
    net_params = wandb.config
    params = wandb.config
    tresh=None
    net_params['total_param'] = view_model_param(MODEL_NAME, net_params, tresh)
    val_acc, train_acc, val_loss, test_acc = train_val_pipeline(MODEL_NAME, DATASET_NAME, params, net_params, dirs, tresh)
    wandb.log(dict({'avg_val_acc': np.mean(val_acc), 'avg_train_acc': np.mean(train_acc),
                    "avg_val_loss": np.mean(val_loss), "avg_test_acc": np.mean(test_acc)}))



main()

