import pickle
import parameters as ps

filename = '3B_DFT_pars.pkl'

with open(filename, 'wb') as f:
    pickle.dump(ps.dic_params_H, f)
    print("Saved to file")
