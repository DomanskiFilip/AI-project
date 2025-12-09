package features;

//modes for MLP and SVM used to specify which feature set to use
public enum FeatureMode {
    RAW_ONLY,
    CENTROID_ONLY,
    RAW_CENTROID,
    RAW_KMEANS,
    RAW_GA,
    ALL
}