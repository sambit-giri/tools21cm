import numpy as np 
from skimage.morphology import ball
import tools21cm as t2c 

rad = 5
data_ball = ball(rad)

def test_fof():
	out = t2c.fof(data_ball)
	assert out[1].squeeze()==data_ball.sum()

def test_mfp():
	out  = t2c.mfp(data_ball, boxsize=data_ball.shape[0], iterations=100000)
	peak = out[0][out[1].argmax()] 
	assert peak<rad+1 or peak>rad

def test_spa():
	out = t2c.spa(data_ball, boxsize=data_ball.shape[0])
	peak = out[0][out[1].argmax()] 
	assert peak<3 or peak>2