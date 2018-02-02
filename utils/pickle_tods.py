"""This script get tods from sql database and save it in pickle so it
is faster to study"""

import moby2
from moby2.instruments import actpol
import cPickle as pickle

db = actpol.TODDatabase()

print "[INFO] Retrieving data from SQL database"
ids = db.select_tods()
print "[INFO] Successfully retrieved data from SQL database"
print "[INFO] Total number of TODs: %d" % len(ids)
print "[INFO] Dumping into pickle file"
with open("tods.pickle","wb") as f:
    pickle.dump(ids, f, 2)

print "[INFO] Done!"





