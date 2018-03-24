#coding: utf-8
import os
import cv2
try:
	import xml.etree.cElementTree as ET
except:
	import xml.etree.ElementTree as ET
import numpy as np
def get_data(input_path):
	'''
	input:
		input_path: VOC 所在文件夹
	return:
		all_imgs: list, 元素为 annotation_data (dict). 其中,
				  annotation_data: {'filepath': xxx (图片路径),
				  					'width': xxx (图片宽),
									'height': xxx (图片高),
									'bboxes': list, 元素为 dict:{'class': xxx (目标类别名字),
																'x1', 'x2', 'y1', 'y2', 'difficult': 布尔值},
									'imageset': 'trainval' or 'test'}
		class_count: dict. 目标个数字典
		class_mapping: dict. 目标编号字典
	'''
	all_imgs = []

	# key: 目标名称
	# value: 目标个数
	classes_count = {}

	# key: 目标名称
	# value: 按先后出现顺序给目标的编号
	class_mapping = {}

	visualise = False

	data_paths = [os.path.join(input_path,s) for s in ['VOC2007', 'VOC2012']]
	
	print('Parsing annotation files')

	for data_path in data_paths:

		annot_path = os.path.join(data_path, 'Annotations')
		imgs_path = os.path.join(data_path, 'JPEGImages')
		imgsets_path_trainval = os.path.join(data_path, 'ImageSets','Main','trainval.txt')
		imgsets_path_test = os.path.join(data_path, 'ImageSets','Main','test.txt')

		trainval_files = []
		test_files = []
		try:
			with open(imgsets_path_trainval) as f:
				for line in f:
					trainval_files.append(line.strip() + '.jpg')
		except Exception as e:
			print(e)

		try:
			with open(imgsets_path_test) as f:
				for line in f:
					test_files.append(line.strip() + '.jpg')
		except Exception as e:
			if data_path[-7:] == 'VOC2012':
				# this is expected, most pascal voc distibutions dont have the test.txt file
				pass
			else:
				print(e)
		
		annots = [os.path.join(annot_path, s) for s in os.listdir(annot_path)]
		idx = 0
		for annot in annots:
			try:
				idx += 1

				et = ET.parse(annot)
				element = et.getroot()

				# element_objs存储了(多个)目标对象定位框的
				element_objs = element.findall('object')
				# 图片名称
				element_filename = element.find('filename').text
				element_width = int(element.find('size').find('width').text)
				element_height = int(element.find('size').find('height').text)

				if len(element_objs) > 0:
					annotation_data = {'filepath': os.path.join(imgs_path, element_filename), 'width': element_width,
									   'height': element_height, 'bboxes': []}

					if element_filename in trainval_files:
						annotation_data['imageset'] = 'trainval'
					elif element_filename in test_files:
						annotation_data['imageset'] = 'test'
					else:
						annotation_data['imageset'] = 'trainval'

				for element_obj in element_objs:
					class_name = element_obj.find('name').text
					if class_name not in classes_count:
						classes_count[class_name] = 1
					else:
						classes_count[class_name] += 1

					if class_name not in class_mapping:
						class_mapping[class_name] = len(class_mapping)

					obj_bbox = element_obj.find('bndbox')
					x1 = int(round(float(obj_bbox.find('xmin').text)))
					y1 = int(round(float(obj_bbox.find('ymin').text)))
					x2 = int(round(float(obj_bbox.find('xmax').text)))
					y2 = int(round(float(obj_bbox.find('ymax').text)))
					difficulty = int(element_obj.find('difficult').text) == 1
					annotation_data['bboxes'].append(
						{'class': class_name, 'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'difficult': difficulty})
				all_imgs.append(annotation_data)

				if visualise:
					img = cv2.imread(annotation_data['filepath'])
					for bbox in annotation_data['bboxes']:
						cv2.rectangle(img, (bbox['x1'], bbox['y1']), (bbox[
									  'x2'], bbox['y2']), (0, 0, 255))
					cv2.imshow('img', img)
					cv2.waitKey(0)

			except Exception as e:
				print(e)
				continue
	return all_imgs, classes_count, class_mapping
