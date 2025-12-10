package algorithms;

import constants.Constants;
import features.FeatureExtractor;
import java.util.ArrayList;
import java.util.List;

public class MahalanobisDistance implements Algorithm {
    private static final int CLASSES = 10;

    @Override
    public Object predict(List<Integer> sample, List<List<Integer>> trainingSet) {
        int featureCount = Constants.BITMAP_SIZE;

        double[][] centroids = FeatureExtractor.calculateCentroids(trainingSet);
        int[] classCounts = new int[CLASSES];
        
        List<List<List<Integer>>> classSamples = new ArrayList<>();
        for (int i = 0; i < CLASSES; i++) {
            classSamples.add(new ArrayList<>());
        }
        
        for (List<Integer> row : trainingSet) {
            int label = row.get(Constants.BITMAP_SIZE);
            classCounts[label]++;
            classSamples.get(label).add(row);
        }

        double[][][] inverseCovariances = new double[CLASSES][][];
        for (int digit = 0; digit < CLASSES; digit++) {
            if (classCounts[digit] > featureCount) {
                double[][] covariance = computeClassCovarianceMatrix(
                    classSamples.get(digit), 
                    featureCount, 
                    centroids[digit]
                );
                inverseCovariances[digit] = MatrixUtils.invertMatrix(covariance);
            } else {
                inverseCovariances[digit] = null;
            }
        }

        double[] sampleVector = new double[featureCount];
        for (int featureIndex = 0; featureIndex < featureCount; featureIndex++) {
            sampleVector[featureIndex] = sample.get(featureIndex);
        }

        double bestDistance = Double.MAX_VALUE;
        int bestClass = -1;
        for (int digit = 0; digit < CLASSES; digit++) {
            if (classCounts[digit] == 0) {
                continue;
            }

            double[] diff = new double[featureCount];
            for (int featureIndex = 0; featureIndex < featureCount; featureIndex++) {
                diff[featureIndex] = sampleVector[featureIndex] - centroids[digit][featureIndex];
            }

            double distance;
            if (inverseCovariances[digit] != null) {
                distance = computeMahalanobisDistance(diff, inverseCovariances[digit]);
            } else {
                distance = computeEuclideanDistance(diff);
            }
            
            if (distance < bestDistance) {
                bestDistance = distance;
                bestClass = digit;
            }
        }
        return Integer.valueOf(bestClass >= 0 ? bestClass : 0);
    }

    private static double[][] computeClassCovarianceMatrix(
            List<List<Integer>> classSamples, 
            int featureCount, 
            double[] classCentroid) {
        
        double[][] covariance = new double[featureCount][featureCount];
        if (classSamples.size() <= 1) {
            for (int i = 0; i < featureCount; i++) {
                covariance[i][i] = 1.0;
            }
            return covariance;
        }

        for (List<Integer> row : classSamples) {
            for (int rowFeatureIndex = 0; rowFeatureIndex < featureCount; rowFeatureIndex++) {
                double diffRow = row.get(rowFeatureIndex) - classCentroid[rowFeatureIndex];
                for (int colFeatureIndex = 0; colFeatureIndex < featureCount; colFeatureIndex++) {
                    double diffCol = row.get(colFeatureIndex) - classCentroid[colFeatureIndex];
                    covariance[rowFeatureIndex][colFeatureIndex] += diffRow * diffCol;
                }
            }
        }

        double denominator = classSamples.size() - 1.0;
        for (int rowIndex = 0; rowIndex < featureCount; rowIndex++) {
            for (int colIndex = 0; colIndex < featureCount; colIndex++) {
                covariance[rowIndex][colIndex] /= denominator;
            }
            covariance[rowIndex][rowIndex] += 0.1;
        }
        return covariance;
    }

    private static double computeEuclideanDistance(double[] diff) {
        double sumSquares = 0;
        for (double d : diff) {
            sumSquares += d * d;
        }
        return Math.sqrt(sumSquares);
    }

    private static double computeMahalanobisDistance(double[] diff, double[][] inverseCovariance) {
        double[] intermediate = new double[diff.length];
        for (int rowIndex = 0; rowIndex < diff.length; rowIndex++) {
            double sum = 0;
            for (int colIndex = 0; colIndex < diff.length; colIndex++) {
                sum += inverseCovariance[rowIndex][colIndex] * diff[colIndex];
            }
            intermediate[rowIndex] = sum;
        }
        
        double distance = 0;
        for (int index = 0; index < diff.length; index++) {
            distance += diff[index] * intermediate[index];
        }
        return Math.sqrt(Math.max(0, distance));
    }
}
