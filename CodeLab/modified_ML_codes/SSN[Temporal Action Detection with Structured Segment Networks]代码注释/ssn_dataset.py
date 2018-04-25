# coding: utf-8
import torch.utils.data as data

import os
import os.path
from numpy.random import randint
from ops.io import load_proposal_file
from transforms import *
from ops.utils import temporal_iou


class SSNInstance:

    def __init__(self, start_frame, end_frame, video_frame_count,
                 fps=1, label=None,
                 best_iou=None, overlap_self=None):
        self.start_frame = start_frame
        self.end_frame = min(end_frame, video_frame_count)
        self._label = label
        self.fps = fps

        # prop部分长度占整个视频的比例
        self.coverage = (end_frame - start_frame) / video_frame_count

        self.best_iou = best_iou
        self.overlap_self = overlap_self

        self.loc_reg = None
        self.size_reg = None

    def compute_regression_targets(self, gt_list, fg_thresh):
        '''
        self(proposal) 依次与 gt_list 中的每个元素去计算 iou, 找到最大的那个 gt, 然后用该 prop 与 max_iou_gt 计算回归参量：
        (1) self.log_reg: 平移回归值
        (2) self.size_reg: 缩放回归值
        '''
        if self.best_iou < fg_thresh:
            # background proposals do not need this
            return

        # find the groundtruth instance with the highest IOU
        ious = [temporal_iou((self.start_frame, self.end_frame), (gt.start_frame, gt.end_frame)) for gt in gt_list]
        best_gt_id = np.argmax(ious)

        best_gt = gt_list[best_gt_id]

        # proposals 和 ground_truth(best) 的中心帧的 index
        prop_center = (self.start_frame + self.end_frame) / 2
        gt_center = (best_gt.start_frame + best_gt.end_frame) / 2

        # proposals 和 ground_truth 的帧的长度
        prop_size = self.end_frame - self.start_frame + 1
        gt_size = best_gt.end_frame - best_gt.start_frame + 1

        # get regression target:
        # (1). center shift propotional to the proposal duration
        # (2). logarithm of the groundtruth duration over proposal duraiton

        # 回归计算的核心代码，显然思路来源于 faster-rcnn
        # 两部分: location_reg: 平移参量; size_reg: 缩放参量
        self.loc_reg = (gt_center - prop_center) / prop_size
        try:
            self.size_reg = math.log(gt_size / prop_size)
        except:
            print(gt_size, prop_size, self.start_frame, self.end_frame)
            raise

    @property
    def start_time(self):
        return self.start_frame / self.fps

    @property
    def end_time(self):
        return self.end_frame / self.fps

    @property
    def label(self):
        return self._label if self._label is not None else -1

    # 回归目标参量: [self.loc_reg, self.size_reg]
    @property
    def regression_targets(self):
        return [self.loc_reg, self.size_reg] if self.loc_reg is not None else [0, 0]


class SSNVideoRecord:
    def __init__(self, prop_record):
        self._data = prop_record

        # 视频帧数
        frame_count = int(self._data[1])

        # build instance record
        self.gt = [
            SSNInstance(int(x[1]), int(x[2]), frame_count, label=int(x[0]), best_iou=1.0) for x in self._data[2]
            if int(x[2]) > int(x[1])
        ]

        self.gt = list(filter(lambda x: x.start_frame < frame_count, self.gt))

        self.proposals = [
            SSNInstance(int(x[3]), int(x[4]), frame_count, label=int(x[0]),
                        best_iou=float(x[1]), overlap_self=float(x[2])) for x in self._data[3] if int(x[4]) > int(x[3])
        ]

        self.proposals = list(filter(lambda x: x.start_frame < frame_count, self.proposals))

    @property
    def id(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    def get_fg(self, fg_thresh, with_gt=True):
        '''
        把 best_iou 大于指定阈值，如 0.7的挑出来
        返回的值是一个有了回归参量的fg的list
        '''
        fg = [p for p in self.proposals if p.best_iou > fg_thresh]
        if with_gt:
            fg.extend(self.gt)

        for x in fg:
            # 这一步后, fg 中的每个对象就有了两个回归属性: self.loc_reg, self.size_reg
            x.compute_regression_targets(self.gt, fg_thresh)
        return fg

    # 产生 bg 和 incomplete 部分
    def get_negatives(self, incomplete_iou_thresh, bg_iou_thresh,
                      bg_coverage_thresh=0.01, incomplete_overlap_thresh=0.7):

        tag = [0] * len(self.proposals)

        incomplete_props = []
        background_props = []

        # best_iou < incomplete_iou_thresh(0.3) 同时 overlap > incomplete_overlap_thresh(0.01) 就是 incomplete
        # overlap: 自己的 span 中有多少比例在 gt 中
        # TODO: 诡异的是，overlap 是 TAG 算法预测出来的！
        for i in range(len(tag)):
            if self.proposals[i].best_iou < incomplete_iou_thresh \
                    and self.proposals[i].overlap_self > incomplete_overlap_thresh:
                tag[i] = 1 # incomplete
                incomplete_props.append(self.proposals[i])

        # best_iou < bg_iou_thresh, 且视频不能太短(converage, prop占整个视频的比例大于一个阈值) 就是 bg
        for i in range(len(tag)):
            if tag[i] == 0 and \
                self.proposals[i].best_iou < bg_iou_thresh and \
                            self.proposals[i].coverage > bg_coverage_thresh:
                background_props.append(self.proposals[i])
        return incomplete_props, background_props


class SSNDataSet(data.Dataset):

    def __init__(self, root_path,
                 prop_file=None,
                 body_seg=5, aug_seg=2, video_centric=True,
                 new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 random_shift=True, test_mode=False,
                 prop_per_video=8, fg_ratio=1, bg_ratio=1, incomplete_ratio=6,
                 fg_iou_thresh=0.7,
                 bg_iou_thresh=0.01, incomplete_iou_thresh=0.3,
                 bg_coverage_thresh=0.02, incomplete_overlap_thresh=0.7,
                 gt_as_fg=True, reg_stats=None, test_interval=6, verbose=True,
                 exclude_empty=True, epoch_multiplier=1):

        self.root_path = root_path
        self.prop_file = prop_file  # proposal_file
        self.verbose = verbose

        self.body_seg = body_seg
        self.aug_seg = aug_seg
        self.video_centric = video_centric 
        # 调用时给了 True
        # 有的数据集比如 THUMOS14 中的一些视频, 是没有 ground truth 的
        self.exclude_empty = exclude_empty  
        
        # replicate the training set by N times in one epoch
        # train 的时候设置的值为 10
        self.epoch_multiplier = epoch_multiplier 

        # rgb: 1, flow/rgb_diff: 5
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.test_interval = test_interval

        self.fg_iou_thresh = fg_iou_thresh
        self.incomplete_iou_thresh = incomplete_iou_thresh
        self.bg_iou_thresh = bg_iou_thresh

        self.bg_coverage_thresh = bg_coverage_thresh
        self.incomplete_overlap_thresh = incomplete_overlap_thresh

        self.starting_ratio = 0.5
        self.ending_ratio = 0.5

        # gt 也加入到 fg 中去, 进行 get_fg (返回回归参量) 计算
        self.gt_as_fg = gt_as_fg

        # 每个mini_batch中 fg, bg, incomp 的比例，实验中分别取 1, 1, 6, 因此是 1: 1: 6
        denum = fg_ratio + bg_ratio + incomplete_ratio

        # TODO: 应该是每个 batch
        # fg, bg, incomp 三部分的 prop 数量
        self.fg_per_video = int(prop_per_video * (fg_ratio / denum))
        self.bg_per_video = int(prop_per_video * (bg_ratio / denum))
        self.incomplete_per_video = prop_per_video - self.fg_per_video - self.bg_per_video

        # reg_stats只在train下为 None, None才去计算它从而生效
        # regression 参量: loc_reg, size_reg
        # self.stats 计算的是 fg_prop 的 回归参量 的 mean 和 std
        self._parse_prop_file(stats=reg_stats)

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]
        elif self.modality == 'Flow':
            x_img = Image.open(os.path.join(directory, self.image_tmpl.format('x', idx))).convert('L')
            y_img = Image.open(os.path.join(directory, self.image_tmpl.format('y', idx))).convert('L')

            return [x_img, y_img]

    def _parse_prop_file(self, stats=None):
        prop_info = load_proposal_file(self.prop_file)

        self.video_list = [SSNVideoRecord(p) for p in prop_info]

        # 删除没有有效gt的
        if self.exclude_empty:
            self.video_list = list(filter(lambda x: len(x.gt) > 0, self.video_list))

        # v.id: video_name, 如: video_test_0000890
        self.video_dict = {v.id: v for v in self.video_list}

        # construct three pools:
        # 1. Foreground
        # 2. Background
        # 3. Incomplete

        # 注意以下三个是用 extend 拼起来的，因此没有内嵌
        self.fg_pool = []  # 依次存储 video_name, 有了回归参量的 前景prop
        self.bg_pool = [] # 依次存储 video_name, bg_prop
        self.incomp_pool = [] # 依次存储 video_name, incomplete_prop

        for v in self.video_list:
            self.fg_pool.extend([(v.id, prop) for prop in v.get_fg(self.fg_iou_thresh, self.gt_as_fg)])

            incomp, bg = v.get_negatives(self.incomplete_iou_thresh, self.bg_iou_thresh,
                                         self.bg_coverage_thresh, self.incomplete_overlap_thresh)

            self.incomp_pool.extend([(v.id, prop) for prop in incomp])
            self.bg_pool.extend([(v.id, prop) for prop in bg])

        if stats is None:
            self._compute_regresssion_stats()
        else:
            self.stats = stats

        if self.verbose:
            print("""
            
            SSNDataset: Proposal file {prop_file} parsed.
            
            There are {pnum} usable proposals from {vnum} videos.
            {fnum} foreground proposals
            {inum} incomplete_proposals
            {bnum} background_proposals
            
            Sampling config:
            FG/BG/INC: {fr}/{br}/{ir}
            Video Centric: {vc}
            
            Epoch size multiplier: {em}
            
            Regression Stats:
            Location: mean {stats[0][0]:.05f} std {stats[1][0]:.05f}
            Duration: mean {stats[0][1]:.05f} std {stats[1][1]:.05f}
            """.format(prop_file=self.prop_file, pnum=len(self.fg_pool) + len(self.bg_pool) + len(self.incomp_pool),
                       fnum=len(self.fg_pool), inum=len(self.incomp_pool), bnum=len(self.bg_pool),
                       fr=self.fg_per_video, br=self.bg_per_video, ir=self.incomplete_per_video, vnum=len(self.video_dict),
                       vc=self.video_centric, stats=self.stats, em=self.epoch_multiplier))
        else:
            print("""
                        SSNDataset: Proposal file {prop_file} parsed.   
            """.format(prop_file=self.prop_file))


    def _video_centric_sampling(self, video):

        fg = video.get_fg(self.fg_iou_thresh, self.gt_as_fg)
        incomp, bg = video.get_negatives(self.incomplete_iou_thresh, self.bg_iou_thresh,
                                     self.bg_coverage_thresh, self.incomplete_overlap_thresh)

        def sample_video_proposals(proposal_type, video_id, video_pool, requested_num, dataset_pool):
            if len(video_pool) == 0:
                # if there is nothing in the video pool, go fetch from the dataset pool
                return [(dataset_pool[x], proposal_type) for x in np.random.choice(len(dataset_pool), requested_num, replace=False)]
            else:
                replicate = len(video_pool) < requested_num
                idx = np.random.choice(len(video_pool), requested_num, replace=replicate)  # 不够就重复采样
                # TODO: 这里格式不就不统一了吗
                return [((video_id, video_pool[x]), proposal_type) for x in idx]

        out_props = []
        out_props.extend(sample_video_proposals(0, video.id, fg, self.fg_per_video, self.fg_pool))  # sample foreground
        out_props.extend(sample_video_proposals(1, video.id, incomp, self.incomplete_per_video, self.incomp_pool))  # sample incomp.
        out_props.extend(sample_video_proposals(2, video.id, bg, self.bg_per_video, self.bg_pool))  # sample background

        return out_props

    def _random_sampling(self):
        out_props = []

        out_props.extend([(x, 0) for x in np.random.choice(self.fg_pool, self.fg_per_video, replace=False)])
        out_props.extend([(x, 1) for x in np.random.choice(self.incomp_pool, self.incomplete_per_video, replace=False)])
        out_props.extend([(x, 2) for x in np.random.choice(self.bg_pool, self.bg_per_video, replace=False)])

        return out_props

    def _sample_indices(self, valid_length, num_seg):
        """
        对 valid_length 随机取 num_seg 段
        :param record: VideoRecord
        :return: list
        """

        average_duration = (valid_length + 1) // num_seg
        if average_duration > 0:
            # normal cases
            offsets = np.multiply(list(range(num_seg)), average_duration) \
                             + randint(average_duration, size=num_seg)
        elif valid_length > num_seg:
            offsets = np.sort(randint(valid_length, size=num_seg))
        else:
            offsets = np.zeros((num_seg, ))

        return offsets

    def _get_val_indices(self, valid_length, num_seg):

        if valid_length > num_seg:
            tick = valid_length / float(num_seg)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(num_seg)])
        else:
            offsets = np.zeros((num_seg,))

        return offsets

    def _sample_ssn_indices(self, prop, frame_cnt):
        '''
        返回值:
        offsets: list, 有 aug_seg + body_seg + aug_seg 个值，记录的是每段的开始
        starting_scale, ending_scale: 有效长度占 预设长度(一半的prop_duration) 的比例
        stage_split: start, course, end 的比例, list, 如[2, 7, 9], 表示 2: 5: 2
        '''
        start_frame = prop.start_frame + 1
        end_frame = prop.end_frame

        duration = end_frame - start_frame + 1
        assert duration != 0, (prop.start_frame, prop.end_frame, prop.best_iou)
        valid_length = duration - self.new_length

        # 确保参数有效性
        valid_starting = max(1, start_frame - int(duration * self.starting_ratio))
        valid_ending = min(frame_cnt - self.new_length + 1, end_frame + int(duration * self.ending_ratio))

        valid_starting_length = (start_frame - valid_starting - self.new_length + 1)
        valid_ending_length = (valid_ending - end_frame - self.new_length + 1)

        # 有效长度 占 固定比例长度(一半) 的比例
        starting_scale = (valid_starting_length + self.new_length - 1) / (duration * self.starting_ratio)
        ending_scale = (valid_ending_length + self.new_length - 1) / (duration * self.ending_ratio)

        # get starting
        starting_offsets = (self._sample_indices(valid_starting_length, self.aug_seg) if self.random_shift
                            else self._get_val_indices(valid_starting_length, self.aug_seg)) + valid_starting
        course_offsets = (self._sample_indices(valid_length, self.body_seg) if self.random_shift
                          else self._get_val_indices(valid_length, self.body_seg)) + start_frame
        ending_offsets = (self._sample_indices(valid_ending_length, self.aug_seg) if self.random_shift
                          else self._get_val_indices(valid_ending_length, self.aug_seg)) + end_frame

        offsets = np.concatenate((starting_offsets, course_offsets, ending_offsets))
        # [2, 7, 9] 
        stage_split = [self.aug_seg, self.aug_seg + self.body_seg, self.aug_seg * 2 + self.body_seg]
        return offsets, starting_scale, ending_scale, stage_split

    def _load_prop_data(self, prop):

        # read frame count
        frame_cnt = self.video_dict[prop[0][0]].num_frames

        # sample segment indices
        # 分别对应: offsets, starting_scale, ending_scale, stage_split
        prop_indices, starting_scale, ending_scale, stage_split = self._sample_ssn_indices(prop[0][1], frame_cnt)

        # turn prop into standard format

        # get label
        if prop[1] == 0:
            label = prop[0][1].label
        elif prop[1] == 1:
            label = prop[0][1].label  # incomplete
        elif prop[1] == 2:
            label = 0  # background
        else:
            raise ValueError()
        frames = []
        for idx, seg_ind in enumerate(prop_indices):
            p = int(seg_ind)
            for x in range(self.new_length):
                frames.extend(self._load_image(prop[0][0], min(frame_cnt, p+x)))

        # get regression target
        # 等于 0 是 foreground
        if prop[1] == 0:
            reg_targets = prop[0][1].regression_targets
            # regression targets 要 标准化 (减均值除以方差)
            reg_targets = (reg_targets[0] - self.stats[0][0]) / self.stats[1][0], \
                          (reg_targets[1] - self.stats[0][1]) / self.stats[1][1]
        else:
            reg_targets = (0.0, 0.0)

        return frames, label, reg_targets, starting_scale, ending_scale, stage_split, prop[1]

    def _compute_regresssion_stats(self):
        if self.verbose:
            print("computing regression target normalizing constants")
        targets = []
        for video in self.video_list:
            fg = video.get_fg(self.fg_iou_thresh, False)
            for p in fg:
                targets.append(list(p.regression_targets))
        # targets: list, n 个 [loc_reg, size_reg]
        # 计算均值，方差
        self.stats = np.array((np.mean(targets, axis=0), np.std(targets, axis=0)))

    def get_test_data(self, video, test_interval, gen_batchsize=4):
        props = video.proposals
        video_id = video.id
        frame_cnt = video.num_frames
        frame_ticks = np.arange(0, frame_cnt - self.new_length, test_interval, dtype=np.int) + 1

        num_sampled_frames = len(frame_ticks)

        # avoid empty proposal list
        if len(props) == 0:
            props.append(SSNInstance(0, frame_cnt - 1, frame_cnt))

        # process proposals to subsampled sequences
        rel_prop_list = []
        proposal_tick_list = []
        scaling_list = []
        for proposal in props:
            rel_prop = proposal.start_frame / frame_cnt, proposal.end_frame / frame_cnt
            rel_duration = rel_prop[1] - rel_prop[0]
            rel_starting_duration = rel_duration * self.starting_ratio
            rel_ending_duration = rel_duration * self.ending_ratio
            rel_starting = rel_prop[0] - rel_starting_duration
            rel_ending = rel_prop[1] + rel_ending_duration

            real_rel_starting = max(0.0, rel_starting)
            real_rel_ending = min(1.0, rel_ending)

            starting_scaling = (rel_prop[0] - real_rel_starting) / rel_starting_duration
            ending_scaling = (real_rel_ending - rel_prop[1]) / rel_ending_duration

            proposal_ticks = int(real_rel_starting * num_sampled_frames), int(rel_prop[0] * num_sampled_frames), \
                             int(rel_prop[1] * num_sampled_frames), int(real_rel_ending * num_sampled_frames)

            rel_prop_list.append(rel_prop)
            proposal_tick_list.append(proposal_ticks)
            scaling_list.append((starting_scaling, ending_scaling))

        # load frames
        # Since there are many frames for each video during testing, instead of returning the read frames,
        # we return a generator which gives the frames in small batches, this lower the memory burden
        # and runtime overhead. Usually setting batchsize=4 would fit most cases.
        def frame_gen(batchsize):
            frames = []
            cnt = 0
            for idx, seg_ind in enumerate(frame_ticks):
                p = int(seg_ind)
                for x in range(self.new_length):
                    frames.extend(self._load_image(video_id, min(frame_cnt, p+x)))
                cnt += 1

                if cnt % batchsize == 0:
                    frames = self.transform(frames)
                    yield frames
                    frames = []

            if len(frames):
                frames = self.transform(frames)
                yield frames

        return frame_gen(gen_batchsize), len(frame_ticks), torch.from_numpy(np.array(rel_prop_list)), \
               torch.from_numpy(np.array(proposal_tick_list)), torch.from_numpy(np.array(scaling_list))

    def get_training_data(self, index):
        # 顺序取
        if self.video_centric:
            video = self.video_list[index]
            props = self._video_centric_sampling(video)
        else:
            # 随机取
            props = self._random_sampling()

        out_frames = []
        out_prop_len = []
        out_prop_scaling = []
        out_prop_type = []
        out_prop_labels = []
        out_prop_reg_targets = []
        out_stage_split = []
        for idx, p in enumerate(props):
            prop_frames, prop_label, reg_targets, starting_scale, ending_scale, stage_split, prop_type = self._load_prop_data(
                p)

            processed_frames = self.transform(prop_frames)
            out_frames.append(processed_frames)
            out_prop_len.append(self.body_seg + 2 * self.aug_seg)
            out_prop_scaling.append([starting_scale, ending_scale])
            out_prop_labels.append(prop_label)
            out_prop_reg_targets.append(reg_targets)
            out_prop_type.append(prop_type)
            out_stage_split.append(stage_split)

        out_prop_len = torch.from_numpy(np.array(out_prop_len))
        out_prop_scaling = torch.from_numpy(np.array(out_prop_scaling, dtype=np.float32))
        out_prop_labels = torch.from_numpy(np.array(out_prop_labels))
        out_prop_reg_targets = torch.from_numpy(np.array(out_prop_reg_targets, dtype=np.float32))
        out_prop_type = torch.from_numpy(np.array(out_prop_type))
        out_stage_split = torch.from_numpy(np.array(out_stage_split))
        out_frames = torch.cat(out_frames)
        return out_frames, out_prop_len, out_prop_scaling, out_prop_type, out_prop_labels, \
               out_prop_reg_targets, out_stage_split

    def get_all_gt(self):
        gt_list = []
        for video in self.video_list:
            vid = video.id
            gt_list.extend([[vid, x.label - 1, x.start_frame / video.num_frames,
                             x.end_frame / video.num_frames] for x in video.gt])
        return gt_list

    def __getitem__(self, index):
        real_index = index % len(self.video_list)
        if self.test_mode:
            return self.get_test_data(self.video_list[real_index], self.test_interval)
        else:
            return self.get_training_data(real_index)

    def __len__(self):
        return len(self.video_list) * self.epoch_multiplier