from datasets.ucf101 import UcfHmdb
from datasets.instagram import Instagram


def get_training_set(opt):
    assert opt.dataset in ['ucf101', 'hmdb51', 'ins']

    if opt.dataset == 'ucf101':
        training_data = UcfHmdb(
            'UCF101',
            opt.video_path,
            opt.dataset_file,
            'training',
            testing=False,
            num_frames=opt.sample_duration // opt.stride_size,
            sample_stride=opt.stride_size,
            n_samples_for_each_video=1,
            n_samples_for_each_frame=1,
            crop_size=opt.sample_size)
    elif opt.dataset == 'hmdb51':
        training_data = UcfHmdb(
            'HMDB51',
            opt.video_path,
            opt.dataset_file,
            'training',
            testing=False,
            num_frames=opt.sample_duration // opt.stride_size,
            sample_stride=opt.stride_size,
            n_samples_for_each_video=1,
            n_samples_for_each_frame=1,
            crop_size=opt.sample_size)
    elif opt.dataset == 'ins':
        training_data = Instagram(
            opt.video_path,
            opt.dataset_file,
            'training',
            num_frames=opt.sample_duration // opt.stride_size,
            sample_stride=opt.stride_size,
            crop_size=opt.sample_size)

    return training_data


def get_validation_set(opt):
    assert opt.dataset in ['ucf101', 'hmdb51', 'ins']

    if opt.dataset == 'ucf101':
        validation_data = UcfHmdb(
            'UCF101',
            opt.video_path,
            opt.dataset_file,
            'validation',
            testing=False,
            num_frames=opt.sample_duration // opt.stride_size,
            sample_stride=opt.stride_size,
            n_samples_for_each_video=1,
            n_samples_for_each_frame=1,
            crop_size=opt.sample_size)
    elif opt.dataset == 'hmdb51':
        validation_data = UcfHmdb(
            'HMDB51',
            opt.video_path,
            opt.dataset_file,
            'validation',
            testing=False,
            num_frames=opt.sample_duration // opt.stride_size,
            sample_stride=opt.stride_size,
            n_samples_for_each_video=1,
            n_samples_for_each_frame=1,
            crop_size=opt.sample_size)
    elif opt.dataset == 'ins':
        validation_data = Instagram(
            opt.video_path,
            opt.dataset_file,
            'validation',
            num_frames=opt.sample_duration // opt.stride_size,
            sample_stride=opt.stride_size,
            crop_size=opt.sample_size)

    return validation_data


def get_test_set(opt):
    assert opt.dataset in ['ucf101', 'hmdb51']
    assert opt.test_subset in ['val', 'test']

    if opt.test_subset == 'val':
        subset = 'validation'
    elif opt.test_subset == 'test':
        subset = 'testing'

    if opt.dataset == 'ucf101':
        test_data = UcfHmdb(
            'UCF101',
            opt.video_path,
            opt.dataset_file,
            subset,
            testing=True,
            num_frames=opt.sample_duration // opt.stride_size,
            sample_stride=opt.stride_size,
            n_samples_for_each_video=10,
            n_samples_for_each_frame=3,
            crop_size=opt.sample_size)
    elif opt.dataset == 'hmdb51':
        test_data = UcfHmdb(
            'HMDB51',
            opt.video_path,
            opt.dataset_file,
            subset,
            testing=True,
            num_frames=opt.sample_duration // opt.stride_size,
            sample_stride=opt.stride_size,
            n_samples_for_each_video=10,
            n_samples_for_each_frame=3,
            crop_size=opt.sample_size)

    return test_data
