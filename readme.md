# CSCE 421 Final Project by Rengang Yang and Zenxiaoran Kang

For obvious reasons we are not allowed to include the raw data or the scripts
to download the data in this repository. The scripts were not written by us and
the developer has their own [form to fill out](https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EKAgQbeDVhmXagg/viewform) to obtain access to
the download script. The raw data creation process, before even generating training
and test data, also takes considerable time and space. IIRC it took around 30 hours to
extract all the frames and 2.5TB of storage.<br/>
You can find the project paper drafts/tex in the `paper/` folder.<br/>
The final PDF is in the root folder as `Final_paper.pdf`.<br/>

As an alterantive, I have included all the pickled sets for training and testing, which
take up considerably less space, but still total to more than 1GB. All trained SVMs
are also pickled.<br/>


## Python script descriptions.
create_dataset.py and Data Processing.ipynb were both ONLY USED for creating the datasets.
They are in now way necessary since I have premade the datasets.
* Binary_SVM.ipynb generated the binary classifiers that are pertinent to each generator.
* Multiclass_Classifier.ipynb was used to generate the multiclass classifier as documented in our paper.
* Ensemble_Classifier.ipynb was used to bring the entire project together, and test the accuracy of the final ensemble classifier.


## Pickled Data generated for binary classifiers
Each pkl file is data generated, looking at the file names, its in this format.<br/>
`<train/test>_<generator>_<numfake>_<epsilon>.pkl`<br/>
* The `<train/test>` indicates whether the set is intended for training or testing.
* The `<generator>` indicates which deepfake generated it. At the moment, we have three generators, DeepFake, Face2Face, and Faceswap.
* The `<numfake>` indicates how many fake images are in the dataset, which are in a 1:1 ratio to real images.
* The `<epsilon>` indicates shift in magnitude when preprocessing.
Thus, an example data file name would be:<br/>
`train_deepfake_10k_500.pkl`, which is a trainingset for deepfakes with 10k deepfake images and 10k real images, preprocessed with a rbf channel width of 500.<br/>

## Pickled Binary SVMS
Each binary svm that we decide to include is pickled, with the names of the variables included in the file names
