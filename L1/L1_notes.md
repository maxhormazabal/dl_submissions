- Mac Instalation:

https://caffeinedev.medium.com/how-to-install-tensorflow-on-m1-mac-8e9b91d93706

- Problems with maximum output size:

https://stackoverflow.com/questions/71025992/change-limit-max-output-file-vscode

- Problems in apple silicon (M1) installing tensor-flow new versions
https://developer.apple.com/forums/thread/721619

The number of parameters in the second line (401920) is the result of = (784 (input size) * 512 (neurons in the dense layer)) + 512 (number of biases, one for each neuron), that means the number of conections between layers. The third line is 512*10 + 20 for the same reason.

**Loss Function: ** Function to minimize, especifically cross-entropy means ** PENDIENTE **

**ITERATION:** Each time you modify the weigths.
**EPOCH:** Each time we use all of the training data.
**BATCH SIZE:** Amount of values to take into account in each iteration inside of an epoch.

**Questions**
¿Cuál es el significado del loss y accuracy de una época? Es el último valor o el promedio de todas??
¿Qué pasa si la división del total de los datos con batch size no es un entero? EL último set tiene una menor cantidad de valores? o se rellena con más valores???