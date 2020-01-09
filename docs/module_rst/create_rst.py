from glob import glob
import os

all_modules = glob('../../tools21cm/*.py')

for ff in all_modules:
	filename = ff.split('/')[-1].split('.py')[0]+'.rst'
	if not os.path.isfile(filename):
		with open(filename, "w") as f:
    			f.write("""
%s
%s
.. automodule:: tools21cm.%s
    :members:

"""%(' '.join(filename.split('.rst')[0].split('_')),''.join(['-' for i in range(len(filename.split('.rst')[0]))]),filename.split('.rst')[0]))

