# bayesian
Notes of my work on Bayesian stats and machine learning. Meaning, a brief summary of my work during Master's. 

All [http://jupyter.org/ Jupyter notebooks] are set to work with python3.5, and require numpy, scipy and matplotlib. Some cells in notebooks may require additional libraries installed but you should be fine if you just skip those.

### Bayesian_Change_Point_Model
The up-to-date documentation of my main focus in my grad years. Contains the model definition, generative model, inference and the source code. Currently, I am working on parameter learning. So, this part is incomplete.

### EM_Algorithm
Notes are copied-pasted from EM section in Bayesian_Change_Point_Model notebook. I mentioned the Jensen's inequality, briefly explained the EM algorithm, and showed bound is strict. Planning to add a toy example, like the on in Barber's book.

### Online_EM_Notes
I took [https://github.com/atcemgil/notes/blob/master/HiddenMarkovModel.ipynb Ali Taylan Cemgil's HMM notes] and added the code for online parameter learning. Currently, trying to tune the learning rate and train with mini batches. Not very promosing results. 

### Visualization
Almost empty notebook. Will add functions to visualize 2D/3D plots, histograms, etc.