# -*- coding: utf-8 -*-

import glob
import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

tkimg = None  # the tkimg being showing, use in this way for a bug of ImageTk module

class VideoLabelTool(object):

    def __init__(self, window):
        global tkimg
        self.window = window
        self.window.title("Powerful Video Label Tool by Yang Chen")
        self.frame = tk.Frame(window)
        self.frame.pack(fill='both', expand=1)
        self.window.bind("<Escape>", self.cancelbox)
        self.window.bind("d", self.nextImage)
        self.window.bind("a", self.prevImage)

        # GUI stuff
        # first line: input video dir and load it
        self.load_label = tk.Label(self.frame, text ='Video Dir:')
        self.load_label.grid(row=0, column=0, sticky=tk.E)
        self.dir_entry = tk.Entry(self.frame)
        self.dir_entry.grid(row=0, column=1, sticky=tk.W+tk.E)
        self.load_button = tk.Button(self.frame, text = "Load", command=self.loadDir)
        self.load_button.grid(row=0, column=2, sticky=tk.W+tk.E)

        # main cavas for showing image
        self.canvas = tk.Canvas(self.frame, height=512, width=512)
        self.canvas.grid(row=1, column=1, rowspan = 30, sticky=tk.W+tk.N)  # rowspan param to make sure listbox look nice
        tkimg = ImageTk.PhotoImage(Image.open('placeholder.png'))
        self.canvas.create_image(0, 0, anchor='nw', image=tkimg)
        self.canvas.bind("<Button-1>", self.mouseClick)
        self.canvas.bind("<Motion>", self.mouseMove)

        # list box1
        self.lb_shownames = tk.Label(self.frame, text = 'Video Class Names:')
        self.lb_shownames.grid(row = 1, column = 2,  sticky = tk.W+tk.N+tk.E)
        self.listbox = tk.Listbox(self.frame, width = 36, height = 7, font = 'Helvetica -12')
        self.listbox.grid(row = 2, column = 2, sticky = tk.N)
        self.select_btn = tk.Button(self.frame, text='Select', command=self.processVideo)
        self.select_btn.grid(row = 3, column = 2, sticky = tk.W+tk.E+tk.N)

        # list box2
        self.blank_space = tk.Label(self.frame)
        self.blank_space.grid(row=4, column=2, sticky= tk.W+tk.E+tk.N)
        self.lb_shownames2 = tk.Label(self.frame, text = 'Video Clip Names:')
        self.lb_shownames2.grid(row = 5, column = 2,  sticky = tk.W+tk.N+tk.E)
        self.listbox2 = tk.Listbox(self.frame, width = 36, height = 7, font = 'Helvetica -12')
        self.listbox2.grid(row = 6, column = 2, sticky = tk.N)
        self.select_btn2 = tk.Button(self.frame, text='Select', command=self.selectVideo)
        self.select_btn2.grid(row = 7, column = 2, sticky = tk.W+tk.E+tk.N)

        # video navigation
        self.blank_space2 = tk.Label(self.frame)
        self.blank_space2.grid(row=8, column=2, sticky= tk.W+tk.E+tk.N)
        self.video_navigation_label = tk.Label(self.frame, text="Video Navigation:")
        self.video_navigation_label.grid(row=9, column=2, sticky= tk.W+tk.E+tk.N)
        self.next_video_button = tk.Button(self.frame, text='Next Video', command=self.nextVideo)
        self.next_video_button.grid(row=10, column=2, sticky=tk.W+tk.E+tk.N)
        self.prev_video_button = tk.Button(self.frame, text='Prev Video', command=self.prevVideo)
        self.prev_video_button.grid(row=11, column=2, sticky=tk.W+tk.E+tk.N)

        # coordinate 
        self.blank_space2 = tk.Label(self.frame)
        self.blank_space2.grid(row=12, column=2, sticky= tk.W+tk.E+tk.N)
        self.coordinate_info = tk.Label(self.frame, text="Coordiante:")
        self.coordinate_info.grid(row=13, column=2, sticky= tk.W+tk.N)
        self.coordinate = tk.Label(self.frame, text="")
        self.coordinate.grid(row=14, column=2, sticky= tk.W+tk.N)

        # disp labels on the bottom
        self.disp1 = tk.Label(self.frame, text='[Video Information]')
        self.disp1.grid(row=31, column=0, columnspan=2, sticky=tk.W)
        self.disp5 = tk.Label(self.frame, text='')
        self.disp5.grid(row=32, column=0, columnspan=2, sticky=tk.W)
        self.disp2 = tk.Label(self.frame, text='')
        self.disp2.grid(row=33, column=0, columnspan=2, sticky=tk.W)
        self.disp3 = tk.Label(self.frame, text='[Frame Information]')
        self.disp3.grid(row=34, column=0, columnspan=2, sticky=tk.W)
        self.disp4 = tk.Label(self.frame, text='')
        self.disp4.grid(row=35, column=0, columnspan=2, sticky=tk.W)
       
        # The img goto navigation
        self.frame2 = tk.Frame(self.frame)
        self.frame2.grid(row=31, column=2, sticky=tk.W)
        self.gotoLabel = tk.Label(self.frame2, text='Go to img No.')
        self.gotoLabel.pack(side='left')
        self.goto_entry = tk.Entry(self.frame2, width = 5)
        self.goto_entry.pack(side='left')
        self.gotoBtn = tk.Button(self.frame2, text='Go!', command=self.gotoImg)
        self.gotoBtn.pack(side='left')

        self.frame.columnconfigure(1, weight = 1)

        # initial params
        # mouse
        self.STATE = {}
        self.STATE['click'] = 0  # whether the mouse is clicked
        self.STATE['x'], self.STATE['y'] = 0, 0

        # dirs
        self.source_dir = None
        self.target_dir = None
        self.current_video_name_dir = None  # current video type name being created in target_dir
        self.current_video_dir = None  # dir of current video being processed 
        self.tmp_dir = None  # tmp dir, for storing imgs from one video clip
        self.path_bigout = None  # the outside dir name
        self.total_frames = 0  # frames_num of each video

        # video & image related
        self.video_names = None  # video names, such as ['run', 'walk']
        self.img_cur = 0  # the index of the video frames in tmp dir (img)
        self.vid_cur = 0
        self.select_cur = 0  # selected video class name
        self.img_list = None
        self.video_list = None  # video list in each video class
        self.img_now = None  # the img being processed (PATH)
        self.box_image = None # the part of img cropped by the box
        self.fps = None
        self.select_vid_class_name = None  # selected video class name in listbox1
        self.box_label_count = 0  # count of box label
        self.box_label_list = []  # a list of names of the box labels, elements: self.img_cur
        self.box_label_set = None

        # label box
        self.hl = None
        self.vl = None
        self.box = None  # the box, a rectangle
        self.box_cor = []  # the cordinates of the box: [x1, y1, x2, y2]
        
    # functions
    def loadDir(self):
        self.source_dir = self.dir_entry.get()
        
        if not self.source_dir:
            messagebox.showerror(title='Error', message='Please input the source video path!')
        else:
            if self.source_dir.endswith('/'):
                self.source_dir = self.source_dir[:-1]
            self.path_bigout = os.path.dirname(self.source_dir)
            self.target_dir = os.path.join(self.path_bigout, 'target')
            self.tmp_dir = os.path.join(self.path_bigout, 'tmp')
            if not os.path.exists(self.target_dir):
                os.mkdir(self.target_dir)

            # insert video names in listbox
            self.video_names = self.listdir(self.source_dir)
            self.listbox.delete(0, len(self.video_names))
            for i in self.video_names:
                self.listbox.insert('end', i)

    def gotoImg(self):
        self.img_cur = int(self.goto_entry.get())
        self.start_label_action()

    def mouseClick(self, event):
        if self.STATE['click'] == 0:
            self.STATE['x'], self.STATE['y'] = event.x, event.y
        else:
            x1, x2 = min(self.STATE['x'], event.x), max(self.STATE['x'], event.x)
            y1, y2 = min(self.STATE['y'], event.y), max(self.STATE['y'], event.y)

            self.box_cor = [x1, y1, x2, y2]
            self.box_img = (cv2.imread(os.path.join(self.tmp_dir, self.img_list[self.img_cur])))[y1:y2, x1:x2]
            cv2.imwrite(self.current_video_dir + '/{}.png'.format(self.img_cur), self.box_img)
            self.box_label_list.append(self.img_cur)
            self.box_label_set = set(self.box_label_list)
            self.box_label_count = len(self.box_label_set) 
            self.disp4.config(text = "Progress: %4d/%4d    Label Frame Count: %3d" % (self.img_cur, len(self.img_list)-1, self.box_label_count))
        self.STATE['click'] = 1 - self.STATE['click']

    def mouseMove(self, event):
        global tkimg
        if tkimg:
            if self.hl:
                self.canvas.delete(self.hl)
            self.hl = self.canvas.create_line(0, event.y, tkimg.width(), event.y, width = 1)
            if self.vl:
                self.canvas.delete(self.vl)
            self.vl = self.canvas.create_line(event.x, 0, event.x, tkimg.height(), width = 1)
        if 1 == self.STATE['click']:
            if self.box:
                self.canvas.delete(self.box)
            self.box = self.canvas.create_rectangle(self.STATE['x'], self.STATE['y'], event.x, event.y, width = 1, outline = 'red')
        self.coordinate.config(text = 'x: %d, y: %d' %(event.x, event.y))

    def cancelbox(self, event):
        if self.STATE['click'] == 1:
            if self.box:
                self.canvas.delete(self.box)
                self.STATE['click'] = 0

    def nextImage(self, event):
        if self.img_cur < self.total_frames - 1:
            self.img_cur += 1
            self.start_label_action()
            self.disp4.config(text = "Progress: %4d/%4d    Label Frame Count: %3d" % (self.img_cur, len(self.img_list)-1, self.box_label_count))

    def prevImage(self, event):
        if self.img_cur >= 1:
            self.img_cur -= 1
            self.start_label_action()
            self.disp4.config(text = "Progress: %4d/%4d    Label Frame Count: %3d" % (self.img_cur, len(self.img_list)-1, self.box_label_count))
    
    def nextVideo(self):
        if self.vid_cur < len(self.video_list) - 1:
            self.img_cur = 0
            self.vid_cur += 1
            self.store_frames_to_tmp(self.select_cur, self.vid_cur)
            self.start_label_action()
            self.disp2.config(text = "FPS: %3d    Progress: %4d/%4d" \
                            %(self.fps, self.vid_cur, len(self.video_list)-1)) 
    
    def prevVideo(self):
        if self.vid_cur >= 1:
            self.img_cur = 0
            self.vid_cur -= 1
            self.store_frames_to_tmp(self.select_cur, self.vid_cur)
            self.start_label_action()
            self.disp2.config(text = "FPS: %3d    Progress: %4d/%4d" \
                            %(self.fps, self.vid_cur, len(self.video_list)-1)) 

    def listdir(self, dir):
        content = os.listdir(dir)
        if '.DS_Store' in content:
            content.remove('.DS_Store')
        return content

    def processVideo(self):

        self.select_cur = int(self.listbox.curselection()[0])  # select idx
        self.select_vid_class_name = self.video_names[self.select_cur]   # selected name

        # # create video class-name dir in target_dir
        # self.current_video_name_dir = os.path.join(self.target_dir, sel_name)
        # if not os.path.exists(self.current_video_name_dir):
        #     os.mkdir(self.current_video_name_dir)
        
        # # create dirs for video frame storage
        # # remove previous listbox2
        if self.video_list:
            self.listbox2.delete(0, len(self.video_list))
        self.video_list = self.listdir(os.path.join(self.source_dir, self.select_vid_class_name))
        for i in self.video_list:
                self.listbox2.insert('end', i) 

        self.disp5.config(text = "Video Classname: %s    Video Filename: %s " \
                            %(self.select_vid_class_name, self.video_list[self.vid_cur]))
    
    def selectVideo(self):
        self.img_cur = 0
        self.box_label_list = []

        # create video class-name dir in target_dir
        self.current_video_name_dir = os.path.join(self.target_dir, self.select_vid_class_name)
        if not os.path.exists(self.current_video_name_dir):
            os.mkdir(self.current_video_name_dir)
        
        self.current_video_dir = self.current_video_name_dir + '/{}'.format(self.vid_cur)
        if not os.path.exists(self.current_video_dir):
            os.mkdir(self.current_video_dir)

        self.vid_cur = int(self.listbox2.curselection()[0])
        self.store_frames_to_tmp(self.select_cur, self.vid_cur)
        self.start_label_action()
        self.disp2.config(text = "FPS: %3d    Progress: %4d/%4d" \
                            %(self.fps, self.vid_cur, len(self.video_list)-1)) 
        self.disp4.config(text = "Progress: %4d/%4d    Label Frame Count: %3d" % (self.img_cur, len(self.img_list)-1, self.box_label_count))
    
    def store_frames_to_tmp(self, select_cur, vid_cur):
        # delete and create tmp dir
        os.system('rm -rf ' + self.tmp_dir)
        os.mkdir(self.tmp_dir)

        video_path = os.path.join(self.source_dir, self.video_names[select_cur], self.video_list[vid_cur])
        cap = cv2.VideoCapture(video_path)
        self.total_frames = int(cap.get(7))
        self.fps = int(cap.get(5))
        for i in range(self.total_frames):
            ret, frame = cap.read()
            frame = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_AREA)
            cv2.imwrite(self.tmp_dir + '/{}.png'.format(i), frame)
        cap.release()

        self.img_list = os.listdir(self.tmp_dir)
        self.img_list.sort(key=lambda x: int(x[:-4]))

    def start_label_action(self):
        global tkimg
        self.img_now = os.path.join(self.tmp_dir, self.img_list[self.img_cur])

        tkimg = ImageTk.PhotoImage(Image.open(self.img_now))
        self.canvas.create_image(0, 0, anchor='nw', image=tkimg)
          
if __name__ == '__main__':
    window = tk.Tk()
    window.geometry('810x700')
    tool = VideoLabelTool(window)
    window.mainloop()
