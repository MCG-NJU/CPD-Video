import torch
from tqdm import tqdm


def calculate_video_results(output_buffer, video_id, test_results, class_names):
    video_outputs = torch.stack(output_buffer)
    average_scores = torch.mean(video_outputs, dim=0)
    sorted_scores, locs = torch.topk(average_scores, k=1)

    video_results = []
    for i in range(sorted_scores.size(0)):
        video_results.append({
            'label': class_names[locs[i].item()],
            'score': sorted_scores[i].item()
        })

    test_results.put([video_id, video_results])


def test(data_loader, model, opt, class_names, test_results):
    print('test')

    model.eval()

    output_buffer = []
    previous_video_id = ''
    i = 0
    for (inputs, targets, _) in tqdm(data_loader, disable=opt.rank!=0):

        with torch.no_grad():
            outputs = model(inputs.cuda())

        for j in range(outputs.size(0)):
            if not (i == 0 and j == 0) and targets[j] != previous_video_id:
                calculate_video_results(output_buffer, previous_video_id,
                                        test_results, class_names)
                output_buffer = []
            output_buffer.append(outputs[j].data.cpu())
            previous_video_id = targets[j]
        i += 1
    test_results.put(-1)