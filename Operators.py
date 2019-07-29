def check_left(bbox1, bbox2):
	# if bbox1[0] < bbox2[0] and bbox1[2] < bbox2[2]:
	# if bbox1[2] < bbox2[0]:
	centre_bbox1_x =  (bbox1[0] + bbox1[2])/2
	centre_bbox2_x =  (bbox2[0] + bbox2[2])/2
	if centre_bbox1_x < centre_bbox2_x and bbox1[2] < bbox2[0]:
		return True
	else:
		return False

def check_front(bbox1, bbox2):
	centre_bbox1_y = (bbox1[1] + bbox1[3])/2
	centre_bbox2_y = (bbox2[1] + bbox2[3])/2
	if centre_bbox1_y - centre_bbox2_y < 20: #and abs((bbox1[3] - bbox2[1])) < 60:
		return True
	else:
		return False

def check_right(bbox1, bbox2):
	# centre_bbox1_x =  (bbox1[0] + bbox1[2])/2
	# centre_bbox2_x =  (bbox2[0] + bbox2[2])/2
	if bbox1[0] < bbox2[0] and bbox1[2] < bbox2[2]:
		return False
	else:
		return True

operations = { 'left': check_left,'left2': check_left,'left3': check_left,'left4': check_left, 'right': check_right, 'front': check_front,'front2': check_front,'front3': check_front,'front4': check_front,'front5': check_front}

for i in range(5,101):
	operations[f'front{i}'] = check_front


