package algorithms;

// A class to normalize features by removing the mean and scaling to unit variance
public class FeatureNormalizer {
    double[] mean;
    double[] std;
    int featureSize;

    FeatureNormalizer(int size) {
        this.featureSize = size;
        this.mean = new double[size];
        this.std = new double[size];
    }

    void fitAndTransform(double[][] matrix) {
        if (matrix.length == 0) {
            return;
        }

        for (int featureIndex = 0; featureIndex < featureSize; featureIndex++) {
            double sum = 0, sumSquared = 0;
            for (double[] row : matrix) {
                sum += row[featureIndex];
                sumSquared += row[featureIndex] * row[featureIndex];
            }
            mean[featureIndex] = sum / matrix.length;
            double variance = (sumSquared / matrix.length) - (mean[featureIndex] * mean[featureIndex]);
            std[featureIndex] = Math.max(Math.sqrt(Math.max(variance, 0)), 1e-9);
        }

        for (double[] row : matrix) {
            for (int featureIndex = 0; featureIndex < featureSize; featureIndex++) {
                row[featureIndex] = (row[featureIndex] - mean[featureIndex]) / std[featureIndex];
            }
        }
    }

    double[] normalize(double[] vector) {
        double[] output = new double[featureSize];
        for (int featureIndex = 0; featureIndex < featureSize; featureIndex++)
            output[featureIndex] = (vector[featureIndex] - mean[featureIndex]) / std[featureIndex];
        return output;
    }
}