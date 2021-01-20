from collections import OrderedDict

import torch
from torch import nn

from models import resnet, cpd


def generate_model(opt):
    assert opt.model in ['resnet']

    if opt.model == 'resnet':
        assert opt.model_depth in [10, 18, 34, 50]

        from models.resnet import get_fine_tuning_parameters

        if opt.model_depth == 10:
            model = resnet.resnet10(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=int(opt.sample_duration / opt.stride_size),
                dp=opt.dp
            )
        elif opt.model_depth == 18:
            model = resnet.resnet18(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=int(opt.sample_duration / opt.stride_size),
                dp=opt.dp
            )
        elif opt.model_depth == 34:
            model = resnet.resnet34(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=int(opt.sample_duration / opt.stride_size),
                dp=opt.dp
            )
        elif opt.model_depth == 50:
            model = resnet.resnet50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=int(opt.sample_duration / opt.stride_size),
                dp=opt.dp,
                freeze_bn=False
            )

    if opt.phase == 'pretraining':
        from models.cpd import get_fine_tuning_parameters
        model = cpd.CPD(visual_encoder=model,
                        N_data=opt.N_data,
                        emb_dim=opt.emb_dim,
                        dropout=opt.dp,
                        K=opt.nce_k,
                        T=opt.nce_t,
                        m=opt.nce_m,
                        gpu=opt.gpu)

    if not opt.no_cuda:
        if opt.distributed:
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices.
            if opt.gpu is not None:
                torch.cuda.set_device(opt.gpu)
                model.cuda(opt.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs we have
                model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[opt.gpu], find_unused_parameters=True)
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                model = torch.nn.parallel.DistributedDataParallel(
                    model, find_unused_parameters=True)
        else:
            # AllGather implementation (batch shuffle, queue update, etc.) in
            # this code only supports DistributedDataParallel.
            raise NotImplementedError(
                "Only DistributedDataParallel is supported.")

        if opt.phase == 'pretraining':
            parameters = get_fine_tuning_parameters(
                model, opt.learning_rate)
            return model, parameters
        elif opt.phase == 'finetuning':
            if opt.pretrain_path is not None:
                print('loading pretrained model {}'.format(opt.pretrain_path))
                pretrain = torch.load(opt.pretrain_path, map_location='cuda:{}'.format(opt.gpu))

                new_state_dict = OrderedDict()
                for name, weights in pretrain['state_dict'].items():
                    if 'visual_encoder' in name:
                        new_state_dict[name.replace(
                            'visual_encoder.', '')] = weights

                x = model.state_dict()
                for k in x.keys():
                    if k not in new_state_dict.keys():
                        print('Missing Keys: ', k)
                x.update(new_state_dict)
                model.load_state_dict(x)
            model.module.fc = nn.Linear(
                model.module.fc.in_features, opt.n_finetune_classes).cuda()
            parameters = get_fine_tuning_parameters(
                model, opt.ft_begin_index, opt.learning_rate)
            return model, parameters
        else:
            raise NotImplementedError(
                'not implement {} phase'.format(opt.phase))
    else:
        raise NotImplementedError("CPU version: not implemented!")
    return model, model.parameters()
