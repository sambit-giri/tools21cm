import numpy as np 
from skimage.morphology import ball
import tools21cm as t2c 

rad = 5
data_ball = ball(rad)

def test_bubbles_from_fixed_threshold():
	out = t2c.bubbles_from_fixed_threshold(data_ball, 0.5, upper_lim=False)
	assert out.sum()==data_ball.sum()
	out = t2c.bubbles_from_fixed_threshold(data_ball, 5, upper_lim=False)
	assert out.sum()==0

def test_bubbles_from_slic():
	out = t2c.bubbles_from_slic(1-data_ball, n_segments=200)
	assert out[rad,rad,rad]==data_ball[rad,rad,rad]

def test_bubbles_from_kmeans():
	out = t2c.bubbles_from_kmeans(data_ball, n_clusters=2, upper_lim=False)
	assert out[rad,rad,rad]==data_ball[rad,rad,rad]
	out = t2c.bubbles_from_kmeans(data_ball, n_clusters=3, upper_lim=False)
	assert out[rad,rad,rad]==data_ball[rad,rad,rad]