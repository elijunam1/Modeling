# Modeling

Gaussian linear models are often insufficient in practical applications, where noise can be heavytailed.
In this problem, we consider a linear model of the form yi = a xi + b + ei. The (ei) are
independent noise from a distribution that depends on x as well as on global parameters; however,
the noise distribution has conditional mean zero given x. The goal is to derive a good estimator for
the parameters a and b based on a sample of observed (x; y) pairs.
1.1 Instructions:
1. Load the data, which is provided as (x; y) pairs in CSV format. Each file contains a data set
generated with different values of a and b. The noise distribution, conditional on x, is the
same for all data sets.
2. Formulate a model for the data-generating process.
3. Based on your model, formulate a loss function for all parameters: a, b, and any additional
parameters needed for your model.
4. Solve a suitable optimization problem, corresponding to your chosen loss function, to obtain
point estimates for the model parameters.
5. Formulate and carry out an assessment of the quality of your parameter estimates.
6. Try additional models if necessary, repeating steps 2 􀀀 5.
