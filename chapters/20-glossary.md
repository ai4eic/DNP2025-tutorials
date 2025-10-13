# Glossary

## Machine Learning Terms

**Activation Function**  
Non-linear function applied to neuron outputs (e.g., ReLU, sigmoid, tanh) to introduce non-linearity in neural networks.

**Adam Optimizer**  
Adaptive learning rate optimization algorithm combining advantages of AdaGrad and RMSProp.

**Adversarial Training**  
Training approach where two models compete (e.g., generator vs discriminator in GANs).

**Attention Mechanism**  
Technique allowing models to focus on specific parts of input data, weighing their importance.

**Backpropagation**  
Algorithm for computing gradients of loss function with respect to network parameters using chain rule.

**Batch Normalization**  
Normalization technique applied to layer inputs to stabilize and accelerate training.

**Batch Size**  
Number of training examples processed together in one forward/backward pass.

**Binary Cross-Entropy**  
Loss function for binary classification: L = -[y log(ŷ) + (1-y) log(1-ŷ)]

**CNN (Convolutional Neural Network)**  
Neural network architecture using convolution operations, effective for image processing.

**Conditional Generation**  
Generating outputs based on specific input conditions (e.g., particle type, energy).

**Confusion Matrix**  
Table showing true positives, false positives, true negatives, and false negatives.

**Convolution**  
Mathematical operation sliding a filter over input to extract features.

**Discriminator**  
Model in GAN that tries to distinguish real from generated samples.

**Dropout**  
Regularization technique randomly deactivating neurons during training to prevent overfitting.

**Early Stopping**  
Training termination when validation performance stops improving.

**Embedding**  
Dense vector representation of discrete variables (e.g., particle types).

**Encoder-Decoder**  
Architecture with encoder compressing input and decoder reconstructing output.

**Epoch**  
One complete pass through entire training dataset.

**Feature Map**  
Output of applying convolutional filters to input, representing learned features.

**Fine-tuning**  
Adjusting pre-trained model parameters on new task-specific data.

**GAN (Generative Adversarial Network)**  
Framework with generator and discriminator competing to improve generation quality.

**Generator**  
Model in GAN that creates synthetic samples from random noise.

**Gradient Descent**  
Optimization algorithm iteratively updating parameters in direction of negative gradient.

**Hyperparameter**  
Configuration external to model (e.g., learning rate, batch size) set before training.

**KL Divergence**  
Measure of difference between two probability distributions.

**Latent Space**  
Lower-dimensional representation encoding essential features of data.

**Learning Rate**  
Step size in gradient descent controlling parameter update magnitude.

**Loss Function**  
Measure of model error used to guide optimization.

**Max Pooling**  
Downsampling operation taking maximum value in each pooling window.

**Mode Collapse**  
GAN failure where generator produces limited variety of samples.

**Normalization**  
Scaling data to standard range (e.g., [0,1] or mean=0, std=1).

**Overfitting**  
Model learns training data too well, performing poorly on unseen data.

**Pooling**  
Downsampling operation reducing spatial dimensions of feature maps.

**Precision**  
Fraction of positive predictions that are correct: TP/(TP+FP)

**Recall (Sensitivity)**  
Fraction of actual positives correctly identified: TP/(TP+FN)

**Regularization**  
Techniques preventing overfitting (e.g., L2 penalty, dropout, data augmentation).

**ResNet (Residual Network)**  
Architecture using skip connections enabling training of very deep networks.

**ROC Curve**  
Plot of true positive rate vs false positive rate at various classification thresholds.

**Sigmoid Function**  
Activation function σ(x) = 1/(1+e^(-x)) mapping input to (0,1).

**Softmax**  
Function converting vector of values to probability distribution.

**Stochastic Gradient Descent (SGD)**  
Optimization using random mini-batches instead of full dataset.

**Stride**  
Step size of filter movement in convolution or pooling.

**Transfer Learning**  
Using knowledge from model trained on one task for different but related task.

**Underfitting**  
Model too simple to capture underlying patterns in data.

**VAE (Variational Autoencoder)**  
Generative model learning latent representations through probabilistic encoding.

**Validation Set**  
Data subset used to tune hyperparameters and prevent overfitting.

**Vanishing Gradient**  
Problem where gradients become extremely small, preventing effective training of deep networks.

## Physics Terms

**BCAL (Barrel Calorimeter)**  
Cylindrical calorimeter surrounding central detector region in GlueX.

**Bremsstrahlung**  
Electromagnetic radiation emitted when charged particle is decelerated.

**Calorimeter**  
Detector measuring particle energy through absorption and energy deposition.

**CDC (Central Drift Chamber)**  
Tracking detector measuring charged particle trajectories near target.

**Cherenkov Radiation**  
Electromagnetic radiation emitted when particle travels faster than light in medium.

**Cluster**  
Group of adjacent detector cells with energy deposition forming coherent pattern.

**dE/dx**  
Rate of energy loss per unit distance, used for particle identification.

**Electromagnetic Shower**  
Cascade of particles produced when high-energy photon or electron interacts in material.

**Energy Resolution**  
Detector's ability to measure energy accurately, typically σ/E = a/√E ⊕ b.

**FCAL (Forward Calorimeter)**  
Lead-glass calorimeter measuring forward-going particles in GlueX.

**FDC (Forward Drift Chamber)**  
Tracking detector for forward-going charged particles.

**GlueX**  
Experiment at Jefferson Lab studying exotic mesons using photon beam.

**Gluon**  
Force carrier of strong interaction binding quarks in hadrons.

**Hadronic Shower**  
Particle cascade initiated by strongly-interacting particle in material.

**HDGeant**  
GlueX's Geant4-based detector simulation software.

**Hybrid Meson**  
Exotic meson containing valence quark-antiquark pair plus gluonic excitation.

**Invariant Mass**  
Mass calculated from energy and momentum: M² = E² - p²c².

**Jefferson Lab (JLab)**  
Thomas Jefferson National Accelerator Facility in Newport News, Virginia.

**Kinematics**  
Study of particle motion described by energy, momentum, and angles.

**Molière Radius**  
Characteristic transverse size of electromagnetic shower.

**Monte Carlo**  
Computational method using random sampling for simulation.

**Multiplicity**  
Number of particles of specific type in event.

**Pair Production**  
Creation of electron-positron pair from photon: γ → e⁺e⁻.

**PDG (Particle Data Group)**  
International collaboration compiling particle physics data.

**PDG Code**  
Numerical identifier for particle types in standard scheme.

**Photomultiplier Tube (PMT)**  
Device converting light into electrical signal through photoemission.

**Photoproduction**  
Particle production via photon-nucleus/photon-nucleon interaction.

**PID (Particle Identification)**  
Process of determining particle type from detector signals.

**Pion (π)**  
Lightest meson, composed of quark-antiquark pair. Three types: π⁺, π⁻, π⁰.

**Position Resolution**  
Accuracy of determining particle impact position in detector.

**Radiation Length**  
Mean distance over which high-energy electron loses all but 1/e of its energy by bremsstrahlung.

**Reconstruction**  
Process of determining particle properties from detector signals.

**Shower Shape**  
Spatial distribution of energy deposition in calorimeter shower.

**Splitoff**  
Secondary electromagnetic shower appearing separated from primary shower.

**Start Counter**  
Detector providing event timing reference in GlueX.

**TOF (Time-of-Flight)**  
Detector measuring particle transit time for mass/velocity determination.

**Tracking**  
Reconstruction of charged particle trajectories from position measurements.

**Trigger**  
System deciding which events to record based on detector signals.

## Computing Terms

**Array**  
Multi-dimensional data structure holding elements of same type.

**Conda**  
Package and environment management system for Python and other languages.

**CUDA**  
NVIDIA's parallel computing platform for GPU acceleration.

**Data Augmentation**  
Techniques artificially increasing dataset size through transformations.

**Dataloader**  
Iterator providing batches of data for training.

**Dataset**  
Collection of data used for training, validation, or testing.

**FID (Fréchet Inception Distance)**  
Metric measuring quality of generated images.

**GPU (Graphics Processing Unit)**  
Specialized processor accelerating parallel computations.

**HDF5**  
Hierarchical Data Format for storing large numerical datasets.

**Inception Score**  
Metric evaluating quality and diversity of generated images.

**Jupyter Notebook**  
Interactive computing environment for data analysis and visualization.

**NumPy**  
Fundamental Python package for numerical computing with arrays.

**Pandas**  
Python library for data manipulation and analysis using DataFrames.

**PyTorch**  
Open-source deep learning framework developed by Meta/Facebook.

**ROOT**  
Data analysis framework widely used in particle physics.

**TensorFlow**  
Open-source deep learning framework developed by Google.

**Tensor**  
Multi-dimensional array generalizing scalars, vectors, and matrices.

**Uproot**  
Python library for reading ROOT files without ROOT installation.

**Virtual Environment**  
Isolated Python environment with specific package versions.

## Statistical Terms

**AUC (Area Under Curve)**  
Area under ROC curve, measuring classifier's ability to distinguish classes.

**Bias**  
Systematic error in estimator or model predictions.

**Correlation**  
Statistical relationship between two variables.

**Cross-Validation**  
Technique assessing model performance by partitioning data into training/validation sets.

**Distribution**  
Probability function describing likelihood of different outcomes.

**Error Bar**  
Graphical representation of uncertainty in measurement.

**F1 Score**  
Harmonic mean of precision and recall: 2(precision × recall)/(precision + recall).

**False Positive**  
Incorrect positive classification (Type I error).

**False Negative**  
Incorrect negative classification (Type II error).

**Gaussian (Normal) Distribution**  
Bell-shaped probability distribution characterized by mean and standard deviation.

**p-value**  
Probability of obtaining result at least as extreme as observed, assuming null hypothesis.

**Significance**  
Statistical measure of result's reliability, often expressed in standard deviations (σ).

**Standard Deviation**  
Measure of spread in data: σ = √(Σ(x-μ)²/N).

**True Positive**  
Correct positive classification.

**True Negative**  
Correct negative classification.

**Variance**  
Measure of data spread: σ².

## Acronyms

**AI** - Artificial Intelligence  
**ANN** - Artificial Neural Network  
**BCAL** - Barrel Calorimeter  
**CDC** - Central Drift Chamber  
**CNN** - Convolutional Neural Network  
**CPU** - Central Processing Unit  
**CV** - Cross-Validation  
**DCGAN** - Deep Convolutional GAN  
**DNP** - Division of Nuclear Physics  
**DNN** - Deep Neural Network  
**FCAL** - Forward Calorimeter  
**FDC** - Forward Drift Chamber  
**FID** - Fréchet Inception Distance  
**GAN** - Generative Adversarial Network  
**GPU** - Graphics Processing Unit  
**HDF5** - Hierarchical Data Format version 5  
**HEP** - High Energy Physics  
**IS** - Inception Score  
**JLab** - Jefferson Lab  
**KL** - Kullback-Leibler (divergence)  
**ML** - Machine Learning  
**MLP** - Multi-Layer Perceptron  
**NN** - Neural Network  
**PDG** - Particle Data Group  
**PID** - Particle Identification  
**PMT** - Photomultiplier Tube  
**RAM** - Random Access Memory  
**ReLU** - Rectified Linear Unit  
**RMS** - Root Mean Square  
**ROC** - Receiver Operating Characteristic  
**SGD** - Stochastic Gradient Descent  
**TOF** - Time-of-Flight  
**VAE** - Variational Autoencoder  
**VRAM** - Video RAM (GPU memory)

---

For more detailed explanations, refer to the relevant chapters or the [References](19-references.md) section.
