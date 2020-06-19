# ClientAware-FL

Here three Federated Learning systems are implemented, used to define four different experiments using:

- MNIST to define iid local datasets (```MNIST``` folder)
- MNIST to define non-iid local datasets (```MNIST``` folder)
- CelebA to define non-iid local datasets (```CelebA``` folder)
- Shakespeare to define non-iid local datasets (to check - ```Shakespeare``` folder)

Each system can be used with different criteria to weight local updates defined by clients. Moreover, all the possible permutation of such criteria can be applied. 
The final weight is defined using a score function based on priority.

To perform an experiment, run the ```main.py``` file inside one of the tre folders cited above. The possible flags are:

- ```--iid``` to set true or false (only for MNIST)
- ```--lr``` to define the learning rate value
- ```--permutation``` or ```-p``` to set the permutation index for aggregation (considering the three criteria defined inside the code)
- ```--gpu``` or ```-g``` to select the GPU to use
- ```--char_dim``` or ```-c``` to define the embedding dimension of a single char (only for Shakespeare)

NB: you must provide CelebA and Shakespeare datasets inside the correspondent folders, respecting the constraints inside the code

Into the ```Evaluation``` folder there are the scripts used to compute results using data obtained running experiments. 

The script inside the ```evaluation.py``` file returns a CSV file that contains a table with the communication rounds needed to gain a certain target accuracy for a certain percentage of clients inside the federation. The possible flags are:

- ```--path``` or ```-p``` to specify the directory path in which are placed experiments
- ```--target-accuracy``` or ```-ta``` to specify the target accuracy to consider for stats generation. Real number in range [0,1]
- ```--percentage-list``` or ```-pl``` to specify the list of client percentages that exceed the target accuracy choosen. E.g. 50 60 80 90
- ```--dataset``` or ```-ds``` to specify the reference dataset (currently only MNIST and CelebA)

The script inside the ```plots.py``` file defines a plot of the aggregate accuracy value (specified in data obtained running experiments) over communication rounds. Parameters must be specified inside the code.

Credits: Tommaso Di Noia, Antonio Ferrara, Domenico Siciliani
