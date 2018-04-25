# coding: utf-8
import numpy as np
import glob
import os
import fnmatch


def load_proposal_file(filename):
    '''
    解析 (normalized_)proposal_list 文件
    注意: normalized_proposal 中的 start 和 end 是 开始的 frame 除以整段 video 的帧数, 也就是比例
    返回一个list，list中的每个元祖有四个元素:
        vid: video_name, 如: video_test_0000890
        n_frame: NUM_UNITS * FPS, FPS对于normalized是1, 正常的是frame数, 参见: https://github.com/yjxiong/action-detection/wiki/A-Description-of-the-Proposal-Files
        gt_boxes: list, 储存 n 个 ground_truth 的 boxes, 每个元素的格式: (CLASS START END)
        pr_boxes: list, 储存 n 个 proposal 的boxes, 每个元素的格式: (CLASS MAX_IOU MAX_OVERLAP START END)
    '''
    lines = list(open(filename))
    from itertools import groupby
    groups = groupby(lines, lambda x: x.startswith('#'))

    info_list = [[x.strip() for x in list(g)] for k, g in groups if not k]

    def parse_group(info):
        offset = 0
        vid = info[offset]  # vid_name: eg: video_test_0000890
        offset += 1

        n_frame = int(float(info[1]) * float(info[2]))  # NUM_UNITS * FPS
        n_gt = int(info[3])  # Ground truth count
        offset = 4

        gt_boxes = [x.split() for x in info[offset:offset+n_gt]]  # ground_truth info
        offset += n_gt

        n_pr = int(info[offset])  # proposal num count
        offset += 1
        pr_boxes = [x.split() for x in info[offset:offset+n_pr]]  # proposal info

        return vid, n_frame, gt_boxes, pr_boxes

    return [parse_group(l) for l in info_list]


def process_proposal_list(norm_proposal_list, out_list_name, frame_dict):
    '''
    输入:
        norm_proposal_list: normalized 的 proposal list
        out_list_name: 输出的 proposal list
        frame_dict: parse_directory() 函数的输出
    '''
    norm_proposals = load_proposal_file(norm_proposal_list)

    processed_proposal_list = []
    for idx, prop in enumerate(norm_proposals):
        vid = prop[0]  # VIDEO_ID
        frame_info = frame_dict[vid]  # VIDEO_ID 的三元组信息: frame_folder 绝对路径名、rgb 图像数量、flow_x 数量
        frame_cnt = frame_info[1]  # rgb 图像数量
        frame_path = frame_info[0]  # flow_x 数量

        # 每个 list 元素的组成: class, frame_start, frame_end
        gt = [[int(x[0]), int(float(x[1]) * frame_cnt), int(float(x[2]) * frame_cnt)] for x in prop[2]]

        # 每个 list 元素的组成: class, MAX_IOU, MAX_OVERLAP, frame_start, frame_end
        prop = [[int(x[0]), float(x[1]), float(x[2]), int(float(x[3]) * frame_cnt), int(float(x[4]) * frame_cnt)] for x
                in prop[3]]

        # 输出格式
        out_tmpl = "# {idx}\n{path}\n{fc}\n1\n{num_gt}\n{gt}{num_prop}\n{prop}"

        gt_dump = '\n'.join(['{} {:d} {:d}'.format(*x) for x in gt]) + ('\n' if len(gt) else '')
        prop_dump = '\n'.join(['{} {:.04f} {:.04f} {:d} {:d}'.format(*x) for x in prop]) + (
            '\n' if len(prop) else '')

        processed_proposal_list.append(out_tmpl.format(
            idx=idx, path=frame_path, fc=frame_cnt,
            num_gt=len(gt), gt=gt_dump,
            num_prop=len(prop), prop=prop_dump
        ))

    open(out_list_name, 'w').writelines(processed_proposal_list)


def parse_directory(path, key_func=lambda x: x[-11:],
                    rgb_prefix='img_', flow_x_prefix='flow_x_', flow_y_prefix='flow_y_'):
    """
    Parse directories holding extracted frames from standard benchmarks
    返回一个字典，key 是 folder_name 的关键部分的名字 VIDEO_ID, 如: video_test_0000890 (for thumos14)
    value是一个三元组，元素依次是 frame_folder 绝对路径名、rgb 图像数量、flow_x 数量
    """
    print('parse frames under folder {}'.format(path))
    # glob 返回的是绝对路径
    # 因此 frame_folders 是一个 list, 里面是 frame_folder 的绝对路径名
    frame_folders = glob.glob(os.path.join(path, '*'))

    def count_files(directory, prefix_list):
        lst = os.listdir(directory)
        cnt_list = [len(fnmatch.filter(lst, x+'*')) for x in prefix_list]
        return cnt_list

    # check RGB
    frame_dict = {}
    for i, f in enumerate(frame_folders):
        all_cnt = count_files(f, (rgb_prefix, flow_x_prefix, flow_y_prefix))
        # k: 指定视频的文件夹的名字, frame_folder's *key* name VID_ID: eg, video_test_0000890
        k = key_func(f)

        x_cnt = all_cnt[1]
        y_cnt = all_cnt[2]
        if x_cnt != y_cnt:
            raise ValueError('x and y direction have different number of flow images. video: '+f)
        if i % 200 == 0:
            print('{} videos parsed'.format(i))

        # k: VID_ID, eg: video_test_0000890
        # f: frame_folder 绝对路径名, all_cnt[0]: rgb 图像数量, x_cnt: x 光流数量
        frame_dict[k] = (f, all_cnt[0], x_cnt)

    print('frame folder analysis done')
    return frame_dict