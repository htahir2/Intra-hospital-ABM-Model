"""

    @author: hannantahir

"""

# run.py
from intra_hospital.model import *  # omit this in jupyter notebooks
from intra_hospital import params
import time

start = time.time()

intra_model = IntraHospModel()
for i in range(params.max_iter):
    intra_model.step()

end = time.time()
elapsed = end - start
print('Total simulation time is = ', elapsed/60, ' Minutes')