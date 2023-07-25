# Cybersecurity-for-satellite-and-drone
Currently widely used in military and civilian applications, drones are endangered by various cyber-attacks [1]. For successful unmanned aerial vehicles (UAV) operations, it is imperative to implement cybersecurity mechanisms. An intrusion detection system (IDS) that detects and identifies a cyber-attack is an essential component of cybersecurity [2]. 
A detailed review of literature relevant to IDS can be found in Appendix ‎8.1. Significant part of recent work in these areas has been based on Machine Learning (ML) techniques (see ‎8.1.1).
Among others, there is an IDS proposal for the Internet of Vehicles (IoV) that uses a combination of convolutional neural networks (CNNs) [3] with transfer learning (TL) [4], ensemble learning [5], and hyperparameter optimization (HPO) [6]. CNN is usually used for image-related ML, and we found the application of this technique to non-image data intriguing. 
We have defined two research questions:

A.	Verify how the IDS approach presented in [6] compares to the ones proposed for drones. 

B.	Test whether CNN-based IDS can be used on board of a drone.

To address question ‎A, we have researched the available datasets, adopted them to CNN models, and ran the standard training and testing procedures. In the process we have encountered an issue with the original research, that we needed to overcome.
To address question ‎B, we have implemented a simulator running on Raspberry Pi 4.  The processor has been selected to be “light” enough to be used on board a drone, and powerful enough to perform CNN transformations. The simulator ran test records through trained models in a single shot mode. We used it to measure:

•	CPU usage

•	RAM consumption

•	Timing per record

# References

[1] 	The Wall Street Journal, "Ukrainian Analysis Identifies Western Supply CHain Behind Iran's DXrones," The Wall Street Journal, 22 November 2022. [Online]. Available: https://www.wsj.com/articles/ukrainian-analysis-identifies-western-supply-chain-behind-irans-drones-11668575332.

[2] 	K. &. V. A. Singh, "Threat Modeling for Multi-UAVs Adhoc Networks," in IEEE Region 10 Conference (TENCON), Penang, 2017. 

[3] 	M.V. Valueva, N.N. Nagornov, P.A. Lyakhov, G.V. Valuev and N.I. Chervyakov, "Application of the residue number system to reduce hardware costs of the convolutional neural network implementation," Mathematics and Computers in Simulation, vol. 177, pp. 232-243, 2020. 

[4] 	M. M. Leonardo, T. J. Carvalho, E. Rezende, R. Zucchi and F. A. Faria, "Deep Feature-Based Classifiers for Fruit Fly Identification (Diptera: Tephritidae)," in 31st SIBGRAPI Conference on Graphics, Patterns and Images (SIBGRAPI), 2018. 

[5] 	D. Opitz and R. Maclin, "Popular Ensemble Methods: An Empirical Study," Journal of Artificial Intelligence Research, vol. 11, pp. pp. 169-198, 1999. 

[6] 	L. Yang and A. Shami, "A Transfer Learning and Optimized CNN Based Intrusion Detection System for Internet of Vehicles," in ICC 2022 - IEEE International Conference on Communications, Ithaka, 2022. 


