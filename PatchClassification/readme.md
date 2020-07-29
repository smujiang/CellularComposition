## cell classification
1. Draw ROIs, and then process ROIs
2. Annotate cells
3. Process annotations, export patches and masks
4. Export measurements
5. Train a model to differentiate cells
6. Validation

## patch classification
1. Draw ROIs (10 ROIs each case), and then process ROIs
2. detect cells in ROIs, export measurements  (preparePatchClassification.groovy)
3. use the trained cell classifier to classify cells (../CellClassification/hand_crafted/cell_prediction.py)
4. import classification result (cell labels) into QuPath (load_updated_measurements.groovy)
5. export patches and masks
6. with cell classification results, prepare data for training and testing 
   (aggregate cellular features as patch descriptors ...)
7. train patch classification model
8. Validation
    i. ROC curve
    ii. Confusion matrix
    iii. Feature importance
    vi. Visualize misclassified patches (import into QuPath: load_patch_locations.groovy)









  