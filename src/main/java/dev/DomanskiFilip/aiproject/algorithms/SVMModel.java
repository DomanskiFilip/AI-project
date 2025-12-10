package algorithms;

// A class to hold the SVM model parameters and methods
public class SVMModel {
    double[][] weights;
    double[] bias;
    double[][] weightSums;
    double[] biasSums;
    long steps = 0;
    final FeatureNormalizer normalizer;
    int positiveClassLabel = -1;
    int negativeClassLabel = -1;

    SVMModel(int numClasses, int numFeatures, FeatureNormalizer norm) {
        this.weights = new double[numClasses][numFeatures];
        this.bias = new double[numClasses];
        this.weightSums = new double[numClasses][numFeatures];
        this.biasSums = new double[numClasses];
        this.normalizer = norm;
    }

    PredictionResult computeScores(double[] features) {
        double[] scores = new double[weights.length];
        int predicted = 0;
        double maxScore = Double.NEGATIVE_INFINITY;
        for (int classIndex = 0; classIndex < weights.length; classIndex++) {
            scores[classIndex] = pointInFeatureSpace(classIndex, features) + bias[classIndex];
            if (scores[classIndex] > maxScore) {
                maxScore = scores[classIndex];
                predicted = classIndex;
            }
        }
        return new PredictionResult(predicted, scores);
    }

    int predictOneVsRest(double[] features) {
        return computeScores(features).predictedClass;
    }

    double pointInFeatureSpace(int classIdx, double[] features) {
        double sum = 0;
        for (int featureIndex = 0; featureIndex < features.length; featureIndex++) {
            sum += weights[classIdx][featureIndex] * features[featureIndex];
        }
        return sum;
    }

    void update(double[] features, int classIdx, double direction) {
        for (int featureIndex = 0; featureIndex < features.length; featureIndex++) {
            weights[classIdx][featureIndex] += 0.02 * direction * features[featureIndex];
        }
        bias[classIdx] += 0.02 * direction;
    }

    void accumulateAverages() {
        steps++;
        for (int classIndex = 0; classIndex < weights.length; classIndex++) {
            for (int featureIndex = 0; featureIndex < weights[0].length; featureIndex++) {
                weightSums[classIndex][featureIndex] += weights[classIndex][featureIndex];
            }
            biasSums[classIndex] += bias[classIndex];
        }
    }

    void finalizeWeights() {
        if (steps == 0) {
            return;
        }
        for (int classIndex = 0; classIndex < weights.length; classIndex++) {
            for (int featureIndex = 0; featureIndex < weights[0].length; featureIndex++) {
                weights[classIndex][featureIndex] = weightSums[classIndex][featureIndex] / steps;
            }
            bias[classIndex] = biasSums[classIndex] / steps;
        }
    }
}
