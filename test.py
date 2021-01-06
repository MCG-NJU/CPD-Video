import torch
import torch.nn.functional as F
import time
import os
import sys
import json
import scipy.io as sio

from utils import AverageMeter


def calculate_video_results(output_buffer, video_id, test_results, score_dict, class_names):
    video_outputs = torch.stack(output_buffer)
    average_scores = torch.mean(video_outputs, dim=0)
    sorted_scores, locs = torch.topk(average_scores, k=10)

    video_results = []
    for i in range(sorted_scores.size(0)):
        video_results.append({
            'label': class_names[locs[i].item()],
            'score': sorted_scores[i].item()
        })

    test_results['results'][video_id] = video_results
    score_dict[video_id] = average_scores.numpy()


def test(data_loader, model, opt, class_names):
    print('test')

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    end_time = time.time()
    output_buffer = []
    previous_video_id = ''
    test_results = {'results': {}}
    score_dict = {}
    for i, (inputs, targets, _, _) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        with torch.no_grad():
            outputs = model(inputs.cuda())

        for j in range(outputs.size(0)):
            if not (i == 0 and j == 0) and targets[j] != previous_video_id:
                calculate_video_results(output_buffer, previous_video_id,
                                        test_results, score_dict, class_names)
                output_buffer = []
            output_buffer.append(outputs[j].data.cpu())
            previous_video_id = targets[j]

        if (i % 100) == 0:
            with open(
                    os.path.join(opt.result_path, '{}.json'.format(
                        opt.test_subset)), 'w') as f:
                json.dump(test_results, f)

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        future_time = (len(data_loader) - i - 1) * batch_time.avg
        hh = int(future_time // 3600)
        mm = int((future_time - hh * 3600) // 60)
        ss = int(future_time - hh * 3600 - mm * 60)

        print('[{}/{}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Eta {hh}h:{mm}m:{ss}s\t'.format(
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time,
                  hh=hh,
                  mm=mm,
                  ss=ss))
    with open(
            os.path.join(opt.result_path, '{}.json'.format(opt.test_subset)),
            'w') as f:
        json.dump(test_results, f)

    sio.savemat(os.path.join(opt.result_path,
                             '{}.mat'.format(opt.test_subset)), score_dict)
