import builtins
import json
import os
import time
import warnings

from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
import torch.distributed as dist
import torch.utils.data.distributed
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter

import test
from opts import parse_opts
from model import generate_model
from dataset import get_training_set, get_validation_set, get_test_set
from datasets.ucf101 import get_video_names_and_annotations
from utils import Logger, accuracy, AverageMeter
from loss.NCECriterion import NCECriterion


def main():
    opt = parse_opts()
    assert opt.phase in [
        'pretraining', 'finetuning'], "Only support pretraining and finetuning."

    if opt.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if opt.dist_url == "env://" and opt.world_size == -1:
        opt.world_size = int(os.environ["WORLD_SIZE"])

    opt.distributed = opt.world_size > 1 or opt.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if opt.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        opt.world_size = ngpus_per_node * opt.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        if not opt.test:
            mp.spawn(main_worker, nprocs=ngpus_per_node,
                     args=(ngpus_per_node, opt))
        else:
            ctx = mp.get_context('spawn')
            test_results = ctx.Queue()
            mp.spawn(main_worker, nprocs=ngpus_per_node,
                     args=(ngpus_per_node, opt, test_results))

    else:
        # Simply call main_worker function
        main_worker(opt.gpu, ngpus_per_node, opt)


def main_worker(gpu, ngpus_per_node, opt, test_results=None):
    opt.gpu = gpu

    # suppress printing if not master

    if opt.multiprocessing_distributed and opt.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if opt.gpu is not None:
        print("Use GPU: {} for training".format(opt.gpu))

    if opt.distributed:
        if opt.dist_url == "env://" and opt.rank == -1:
            opt.rank = int(os.environ["RANK"])
        if opt.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            opt.rank = opt.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=opt.dist_backend, init_method=opt.dist_url,
                                world_size=opt.world_size, rank=opt.rank)
        opt.batch_size = int(opt.batch_size / ngpus_per_node)
        opt.n_threads = int(
            (opt.n_threads + ngpus_per_node - 1) / ngpus_per_node)

    if opt.rank % ngpus_per_node == 0:
        if not os.path.exists(opt.result_path):
            os.makedirs(opt.result_path)
        opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
        with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
            json.dump(vars(opt), opt_file)

    if not opt.no_train:
        training_data = get_training_set(opt)
        opt.N_data = len(training_data)

        if opt.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                training_data)
        else:
            train_sampler = None
        train_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=opt.batch_size,
            shuffle=(train_sampler is None),
            num_workers=opt.n_threads,
            pin_memory=True,
            drop_last=True,
            sampler=train_sampler)

        model, parameters = generate_model(opt)

        if opt.phase == 'finetuning':
            criterion = nn.CrossEntropyLoss().cuda(opt.gpu)
        elif opt.phase == 'pretraining':
            criterion = NCECriterion(len(training_data)).cuda(opt.gpu)
        else:
            raise NotImplementedError(
                'not implement {} phase'.format(opt.phase))

        train_logger = Logger(
            os.path.join(opt.result_path, 'train.log.rank{}'.format(opt.rank)),
            ['epoch', 'loss', 'acc', 'lr'])
        train_batch_logger = Logger(
            os.path.join(opt.result_path,
                         'train_batch.log.{}'.format(opt.rank)),
            ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])

        optimizer = optim.SGD(parameters, lr=opt.learning_rate,
                              momentum=opt.momentum,
                              weight_decay=opt.weight_decay)

        if opt.phase == 'finetuning':
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer, 'max', patience=opt.lr_patience,
                min_lr=1e-6, factor=opt.lr_factor)

    if not opt.no_val:
        validation_data = get_validation_set(opt)
        if opt.distributed:
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                validation_data)
        else:
            val_sampler = None
        val_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=opt.batch_size,
            shuffle=(val_sampler is None),
            num_workers=opt.n_threads,
            pin_memory=True,
            drop_last=True,
            sampler=val_sampler)
        val_logger = Logger(
            os.path.join(opt.result_path, 'val.log.rank{}'.format(opt.rank)),
            ['epoch', 'acc1', 'acc5'] if opt.phase == 'finetuning' else ['epoch', 'recall@1', 'recall@10'])
    
    if opt.test:
        model, parameters = generate_model(opt)

        test_data = get_test_set(opt)
        idx_to_labels = test_data.get_idx_to_label()
        if opt.distributed:
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_data, shuffle=False)
        else:
            test_sampler = None
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=opt.batch_size,
            shuffle=(test_sampler is None),
            num_workers=opt.n_threads,
            pin_memory=True,
            drop_last=False,
            sampler=test_sampler)

    if opt.resume_path:
        print('==>loading checkpoint {}'.format(opt.resume_path))
        if opt.gpu is None:
            checkpoint = torch.load(opt.resume_path)
        else:
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(opt.gpu)
            checkpoint = torch.load(opt.resume_path, map_location=loc)

        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if not opt.no_train:
            optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        opt.begin_epoch = 1

    torch.backends.cudnn.benchmark = True
    if opt.rank % ngpus_per_node == 0:
        summary_writer = SummaryWriter(log_dir=opt.result_path)
    else:
        summary_writer = None
    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        if not opt.no_train:
            if opt.distributed:
                train_sampler.set_epoch(i)

            train_epoch(i, train_loader, model, criterion, optimizer, opt,
                        train_logger, train_batch_logger, summary_writer)

        if not opt.no_val:
            if opt.phase == 'finetuning':
                val_acc = val_finetune_epoch(i, val_loader, model, opt,
                                             val_logger, summary_writer)
            elif opt.phase == 'pretraining':
                val_acc = val_pretrain_epoch(i, val_loader, model, opt,
                                             val_logger, summary_writer)
        if not opt.no_train and not opt.no_val:
            if opt.phase == 'finetuning':
                scheduler.step(val_acc)
            elif opt.phase == 'pretraining':
                adjuest_learning_rate(optimizer, i, opt)

    if opt.test:
        test.test(test_loader, model, opt, idx_to_labels, test_results)
        if opt.multiprocessing_distributed and opt.gpu == 0:
            result_json = {}
            finish_procs = 0
            while(finish_procs < ngpus_per_node):
                rst = test_results.get()
                if rst == -1:
                    finish_procs += 1
                else:
                    result_json[rst[0]] = rst[1]
            with open(
                os.path.join(opt.result_path, '{}.json'.format(opt.test_subset)),
                    'w') as f:
                json.dump({'results': result_json}, f)



def train_epoch(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, batch_logger, writer=None):
    print('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    end_time = time.time()
    for i, (inputs, targets, index) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        if opt.phase == 'finetuning':
            if not opt.no_cuda:
                inputs = inputs.cuda(opt.gpu, non_blocking=True)
                targets = targets.cuda(opt.gpu, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            acc = accuracy(outputs, targets)[0]
        elif opt.phase == 'pretraining':
            if not opt.no_cuda:
                vis_inputs = inputs.cuda(opt.gpu, non_blocking=True)
                text_inputs = targets.cuda(opt.gpu, non_blocking=True)
                index = index.cuda(opt.gpu, non_blocking=True)

            vis_out, text_out = model(
                vis_inputs, text_inputs, index, epoch > opt.ft_bert_ep)
            l_vp, l_vn = criterion(vis_out)
            l_tp, l_tn = criterion(text_out)
            loss = l_vp + l_vn + l_tp + l_tn
            bsz = vis_out.shape[0]
            acc = accuracy(vis_out.squeeze(),
                           torch.zeros([bsz]).cuda().long())[0]
        else:
            raise NotImplementedError(
                'not implement {} phase'.format(opt.phase))

        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc[0], inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        if opt.clip_gradient is not None:
            total_norm = nn.utils.clip_grad_norm_(
                model.parameters(), opt.clip_gradient)
            if total_norm > opt.clip_gradient:
                print("clipping gradient: {} with coef {}".format(
                    total_norm, opt.clip_gradient / total_norm))
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()
        if i % opt.print_frq == 0:
            batch_logger.log({
                'epoch': epoch,
                'batch': i + 1,
                'iter': (epoch - 1) * len(data_loader) + (i + 1),
                'loss': losses.val,
                'acc': accuracies.val,
                'lr': optimizer.param_groups[0]['lr']
            })
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Lr {3}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  'Acc {accuracies.val:.7f} ({accuracies.avg:.7f})'.format(
                      epoch,
                      i + 1,
                      len(data_loader),
                      optimizer.param_groups[0]['lr'],
                      batch_time=batch_time,
                      data_time=data_time,
                      losses=losses,
                      accuracies=accuracies))

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'acc': accuracies.avg,
        'lr': optimizer.param_groups[-1]['lr']
    })
    print('Epoch: [{0}]\t'
          'Lr {1}\t'
          'Loss {loss.avg:.4f}\t'
          'Acc {acc.avg:.7f}'.format(
              epoch,
              optimizer.param_groups[0]['lr'],
              loss=losses,
              acc=accuracies))
    if writer is not None:
        writer.add_scalar('scalar/train_loss', losses.avg, epoch)
        writer.add_scalar('scalar/train_acc', accuracies.avg, epoch)
        writer.flush()

    if epoch % opt.checkpoint == 0 and opt.rank == 0:
        save_file_path = os.path.join(opt.result_path,
                                      'save_{}.pth'.format(epoch))
        states = {
            'epoch': epoch + 1,
            'arch': opt.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_file_path)


def val_pretrain_epoch(epoch, data_loader, model, opt, logger, writer=None):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    video_feats = []
    text_feats = []

    for vis_inputs, text_inputs, _ in tqdm(data_loader, disable=opt.rank!=0):
        with torch.no_grad():
            if not opt.no_cuda:
                text_inputs = text_inputs.cuda(opt.gpu, non_blocking=True)
                vis_inputs = vis_inputs.cuda(opt.gpu, non_blocking=True)

            vis_out, text_out = model(vis_inputs, text_inputs)
            video_feats.append(vis_out)
            text_feats.append(text_out)

    video_feats = torch.cat(video_feats, dim=0)
    video_feats = concat_all_gather(video_feats)

    text_feats = torch.cat(text_feats, dim=0)
    text_feats = concat_all_gather(text_feats)

    rank_mat = torch.mm(text_feats, video_feats.t())
    target = torch.tensor(list(range(video_feats.size(0)))).cuda().long()
    top1, top10 = accuracy(rank_mat, target, topk=(1, 10))
    top1, top10 = top1[0], top10[0]

    print("recall@1: {}".format(top1))
    print("recall@10: {}".format(top10))

    logger.log({'epoch': epoch, 'recall@1': top1, 'recall@10': top10})
    if writer is not None:
        writer.add_scalar('scalar/recall@1', top1, epoch)
        writer.add_scalar('scalar/recall@5', top10, epoch)
        writer.flush()

    return top1


def val_finetune_epoch(epoch, data_loader, model, opt, logger, writer=None):
    print('validation at epoch {}'.format(epoch))

    model.eval()

    acc1_avg = AverageMeter()
    acc5_avg = AverageMeter()

    for inputs, targets, _ in tqdm(data_loader, disable=opt.rank!=0):
        with torch.no_grad():
            if not opt.no_cuda:
                targets = targets.cuda(opt.gpu, non_blocking=True)
                inputs = inputs.cuda(opt.gpu, non_blocking=True)

            outputs = model(inputs)
            outputs = concat_all_gather(outputs)
            targets = concat_all_gather(targets)
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            acc1_avg.update(acc1[0], inputs.size(0))
            acc5_avg.update(acc5[0], inputs.size(0))

    logger.log({'epoch': epoch, 'acc1': acc1_avg.avg, 'acc5': acc5_avg.avg})
    print('Epoch: [{0}]\t'
          'Acc@1 {acc1_avg.avg:.3f}\t'
          'Acc@5 {acc5_avg.avg:.3f}'.format(
              epoch,
              acc1_avg=acc1_avg,
              acc5_avg=acc5_avg))
    if writer is not None:
        writer.add_scalar('scalar/val_acc1', acc1_avg.avg, epoch)
        writer.add_scalar('scalar/val_acc5', acc5_avg.avg, epoch)
        writer.flush()
    return acc1_avg.avg


def adjuest_learning_rate(optimizer, epoch, opt):
    if (epoch + 1) == opt.ft_bert_ep:
        print('start updating bert parameters.')
        for param_group in optimizer.param_groups:
            param_group['lr'] = opt.lr_bert * \
                0.1 if param_group['lr'] == 0 else param_group['lr']
    gamma = 1
    if epoch == 100:
        gamma = 0.1
    elif epoch == 201:
        gamma = 0.1
    elif epoch == 250:
        gamma = 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] *= gamma


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


if __name__ == '__main__':
    main()
