#!/share/apps/opt/python/2.7.9/bin/python2
#$ -S /share/apps/opt/python/2.7.9/python2
#$ -V
#$ -cwd
#$ -j y
#$ -M aty@ucsd.edu
#$ -o ./output
#$ -e ./error
#$ -q batch.q

import os

# Variables
M = [200]
Ninit = 10

SGE_TASK_ID = int(os.getenv("SGE_TASK_ID", 0))

i_M = (int(SGE_TASK_ID - 1) / int(Ninit)) % int(len(M)))
initID = int(SGE_TASK_ID - 1) % Ninit + 1

print("M = %d" % (M[i_M],))
print("SGE_TASK_ID = %d" % (SGE_TASK_ID,))

print(os.system("uname -n"))
os.system("python2 hmc_boom.py %d %d" % (initID, M[i_M]))