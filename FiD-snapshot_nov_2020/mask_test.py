import numpy as np
context_mask=np.full((4,9,9), False)
record=[[1,3,5,8],[2,4,7,9],[3,6,7,8],[4,6,7,9]]

for i,sample in enumerate(record):
    for j,x in enumerate(sample):
        # import pdb
        # pdb.set_trace()
        if j ==0:
            context_mask[i][0:x,:]=True
            context_mask[i][:,0:x]=True
        
        else:
            pre = sample[j-1]
            post = sample[j]
            context_mask[i][pre:post,pre:post]=True

import pdb
pdb.set_trace()