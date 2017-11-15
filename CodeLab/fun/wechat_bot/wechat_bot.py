# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import itchat
from itchat.content import *
import pyautogui as pg
import os
from glob import glob
import cv2
import time
import json
import commands

os.chdir('/Users/chenyang/Desktop/wechat_bot')
auto_reply = 0
auto_reply_word = u'您好，我现在有事不在，稍后回来恢复您'
passwd = '1245819728'

def login():
	itchat.auto_login(hotReload=True)
	itchat.dump_login_status(fileDir='/Users/chenyang/Desktop/wechat_bot/log')
	
# iTunes control
itunes_play = '''
exec osascript <<EOF
tell application "iTunes"
	playpause
end tell'''

itunes_next = '''
exec osascript <<EOF
tell application "iTunes"
	next track
end tell'''

itunes_previous = '''
exec osascript <<EOF
tell application "iTunes"
	previous track
end tell'''

itunes_replay= '''
exec osascript <<EOF
tell application "iTunes"
	previous track
	next track
	play
end tell'''

itunes_playselected='''
exec osascript <<EOF
tell application "iTunes"
	set results to (every track whose name contains "{}")
	repeat with tune in results
		play tune
	end repeat
end tell
EOF'''

iTunes_control = {u'播放': itunes_play, u'暂停': itunes_play, 
u'上一首': itunes_previous, u'下一首': itunes_next, u'重播': itunes_replay}

volume_control='''
exec osascript <<EOF
set volume output volume {} --100%'''

# pg.screenshot 在息屏下截图会失败，一般熄屏就是锁屏
# 但是截图失败再接着调用 pg.screenshot 就会报错，只好自己实现
def screenshot_imgname():
	pg.hotkey('shift', 'command', '3')
	img_list = glob('/Users/chenyang/Desktop/*png')
	print img_list
	if len(img_list) == 1:
		screenshot_img = img_list[0]
	else:
		img_list.sort(key=lambda img: os.stat(img).st_mtime)
		screenshot_img = img_list[-1]	
	return screenshot_img


@itchat.msg_register([TEXT,PICTURE])
def text_reply(msg):
	global auto_reply, commands
	# 给自己发消息
	if msg['FromUserName'] == msg['ToUserName']:
		if msg.text in [u'截图', u'截屏']:
			msg.user.send(u'正在执行操作:截图')
			try:
				pg.screenshot('wechat_screenshot.png')
				msg.user.send_image('wechat_screenshot.png')
				# 锁屏时 screenshot 函数失效，截图不存在，因此下一行
				# 删除文件时报 IO 错误
				os.remove('wechat_screenshot.png')
			# 锁屏状态
			except IOError:
				itchat.send(u'似乎是锁屏了，正在自动解锁并截图')
				pg.typewrite(passwd)
				pg.press('enter')
				# 这里用 pg.screenshot('chat_screenshot.png') 报错
				msg.user.send_image(screenshot_imgname())
				os.remove(screenshot_imgname())
		elif msg.text == u'拍照':
			cap = cv2.VideoCapture(0)
			time.sleep(1)
			_, pic = cap.read()
			cv2.imwrite('monitor.png', pic)
			cap.release()
			msg.user.send(u'System: 正在执行操作--拍照')
			msg.user.send_image('monitor.png')
			os.remove('monitor.png')
		elif msg.text.startswith(u'说 '):
			os.system('say ' + msg.text[2:].encode('utf-8'))
		elif msg.text in [u'播放', u'暂停', u'上一首', u'下一首', u'重播'] or msg.text.startswith(u'播放'):
			if len(msg.text) > 2 and msg.text.startswith(u'播放'):
				os.system(itunes_playselected.format(msg.text[2:].encode('utf-8')))
			else:
				os.system(iTunes_control[msg.text])
		elif msg.text.startswith(u'音量') or msg.text == u'静音':
			if msg.text == u'静音':
				os.system(volume_control.format(0))
			else:
				os.system(volume_control.format(msg.text[2:].encode('utf-8')))
		elif msg.text in [u'退出', u'logout']:
			itchat.send(u'您的小管家已经下线，再见~')
			itchat.logout()
		elif msg.text == u'锁屏':
			pg.hotkey('shift', 'command', '2')
			msg.user.send(u'已锁屏')
		elif msg.text == u'解锁':
			pg.typewrite(passwd)
			pg.press('enter')
			msg.user.send(u'已解锁')
		elif u'自动回复' in msg.text:
			if msg.text.startswith(u'开启'):
				auto_reply = 1
				return auto_reply_word
			elif msg.text.startswith(u'关闭'):
				auto_reply = 0
				return u'自动回复已关闭'
			elif msg.text == u'自动回复状态':
				return u'自动回复已开启' if auto_reply else u'自动回复已关闭'
			elif msg.text.startswith(u'设置自动回复 '):
				auto_reply_word = msg.text[7:]
				return u'自动回复信息已设置为:' + msg.text[7:]
			else:
				return u'''指令错误，可选指令为:
【开启/关闭自动回复】: 开启/关闭自动回复系统
【自动回复状态】: 查看自动回复是否开启
【设置自动回复 XXX】: 设置自动回复内容为XXX'''
		elif msg.text == u'指令列表':
			commands = u'''【截图】: 桌面截图发过来
【拍照】: 远程调用摄像头拍照
【说 XXX】: 电脑远程朗读出 XXX 内容
【send XXX】: XXX 发送到电脑剪贴板
【播放 XXX】: 电脑 iTunes 播放歌曲 XXX (关键词匹配)
【播放/暂停/上一首/下一首/重播】: 控制 iTunes 播放
【音量+数字(百分比)/静音】: 调节系统音量
【开启/关闭自动回复】: 开启/关闭自动回复系统
【自动回复状态】: 查看自动回复是否开启
【设置自动回复 XXX】: 设置自动回复内容为XXX
【退出】: 退出远端微信
【锁屏】: 电脑锁屏
【解锁】: 电脑解锁
【电量】: 获取电脑剩余电量'''
			itchat.send(commands)
		elif msg.text in [u'电量', u'剩余电量', u'电脑电量']:
			batt_remain = commands.getoutput('pmset -g batt | egrep "([0-9]+\%).*" -o --colour=auto | cut -f1 -d";"')
#			print type(batt_remain)
			itchat.send(batt_remain)
		elif msg.text.startswith(u'send '):
			send_text = msg.text[5:]
			send_order = u'echo "{}" | pbcopy'.format(send_text).encode('utf-8') 
			os.system(send_order)
			itchat.send(u'send 后文字已经发送到剪贴板')
		else:
			target_username = itchat.search_friends(nickName='NEAL')[0].UserName
			itchat.send(u'抱歉，没有这条指令！输入【指令列表】获取可用指令', toUserName=target_username)
	# 别人发消息给我
	elif msg['FromUserName'] == msg.user[u'UserName']:
		if auto_reply: 
			msg.user.send(u'你好，我现在有事不在，一会回来回复')

@itchat.msg_register([TEXT, NOTE, SHARING, PICTURE, RECORDING, ATTACHMENT], isGroupChat=True)
def group_text_reply(msg):
	
	time_now = time.strftime('%m/%d %A %H:%M',time.localtime(time.time())) # 时间
	chatroom_id = msg['FromUserName']  # 群ID
	chatroom_name = msg.User['NickName'] # 群名
	sender_name = msg['ActualNickName']  # 发消息人的名字(群昵称，其次再是昵称)
	
	# 红包优先级第一
	if msg.type == 'Note' and u'红包' in msg.Text:
		itchat.send(u'%s\n群[%s]\n有人发红包啦!\n快去抢啊!' % (time_now, chatroom_name))

		sys_out = u'%s发红包啦!' % chatroom_name
		for i in range(3):
			os.system('say %s' % sys_out.encode('utf-8'))

		# 抢长林的红包，提醒发到指定的群
		if chatroom_name == u'长林咖啡书屋':
			# itchat.send(u'%s\n群[%s]\n[%s]\n发红包啦!' % (time_now, chatroom_name, sender_name), toUserName=huyun.UserName)
			itchat.send(u'%s\n群[%s]\n有人发红包啦!\n快去抢啊!' % (time_now, chatroom_name), toUserName=hongbao_target['UserName'])
		
	# 过滤已经屏蔽群的指定消息
	if chatroom_name in target_chatgroup:
		# 文件优先
		if msg.type == 'Attachment':
			itchat.send(u'%s\n[%s] 在 [%s] 发送了文件!' % (time_now, sender_name, chatroom_name))
		# 只接收指定人或者带有指定关键词的消息
		if sender_name in target_chatgroup[chatroom_name]['vip'] or sum([keyword in msg.text for keyword in target_chatgroup[chatroom_name]['keyword']]) > 0:
			if msg.type == 'Text':
				itchat.send(u'%s\n[%s] 在 [%s] 说道:\n%s' % (time_now, sender_name, chatroom_name, msg.text))
			elif msg.type == 'Sharing':
				itchat.send(u'%s\n[%s] 在 [%s] 分享了:\n"%s\n%s"' % (time_now, sender_name, chatroom_name, msg.text, msg['Url']))
			elif msg.type == 'Recording':
				itchat.send(u'%s\n[%s] 在 [%s] 发送了语音:' % (time_now, sender_name, chatroom_name))
				itchat.send(msg.fileName)
			elif msg.type == 'Picture':
				msg.download(msg.fileName)
				itchat.send(u'%s\n[%s] 在 [%s] 发送了图片:' % (time_now, sender_name, chatroom_name))
				itchat.send_image(msg.fileName)
				os.remove(msg.fileName)
#	print msg.type
	
login()

itchat.send(u'您的小管家已经上线~')


huyun = itchat.search_friends(name=u'胡昀')[0]
hongbao_target = itchat.search_chatrooms(name=u'red packet')[0]

# itchat.send(u'官方提示: 红包提醒机器人已经上线~', toUserName=hongbao_target['UserName'])

# 先调用一下 pg，免得第一个指令出现调用失败
pg.press('fn')

# target_chatgroup = {
# 	u'北航三系17级硕士':
# 		{'vip': [u'程楠楠', u'305-郭晓明'],
# 		 'keyword': [u'通知', u'需知', u'考试', u'开学', u'3集合', u'点名', u'证件', u'报道', u'入住', u'钥匙', u'班会']},
# 	u'66666':
# 		{'vip': [u'123'],
# 		 'keyword': [u'测试']}
# }

f = open('target_chatgroup.txt', 'r').read().decode('utf-8')
target_chatgroup = f

itchat.run()