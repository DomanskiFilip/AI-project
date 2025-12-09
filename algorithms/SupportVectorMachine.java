package algorithms;

import constants.Constants;
import features.FeatureExtractor;
import features.FeatureMode;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class SupportVectorMachine implements Algorithm {

    private static final int CLASSES = 10;
    private static final int MAX_EPOCHS = 10;
    private static final double LEARNING_RATE = 0.02;
    private static final double MARGIN = 0.002;
    private static final long RANDOM_SEED = 42;

    private final FeatureMode mode;
    private boolean trained = false;

    private double[][] centroidCache;
    private double[][] kmeansCentroidCache;
    private double[] gaWeightsCache;

    private SVMModel oneVsRestModel;
    private List<SVMModel> oneVsOneModels;

    public SupportVectorMachine(FeatureMode mode) {
        this.mode = mode;
    }

    @Override
    public Object predict(List<Integer> sample, List<List<Integer>> trainingSet) {
        if (!trained) {
            train(trainingSet);
            trained = true;
        }

        double[] rawFeatures = projectToFeatureSpace(sample);
        double[] features = oneVsRestModel.normalizer.normalize(rawFeatures);

        int oneVsRestPrediction = oneVsRestModel.predictOneVsRest(features);
        int oneVsOnePrediction = classifyOneVsOne(features);

        return new int[] { oneVsRestPrediction, oneVsOnePrediction };
    }

    public boolean isTrained() {
        return trained;
    }

    private void train(List<List<Integer>> trainingSet) {
        boolean needCentroids = (mode == FeatureMode.ALL || mode == FeatureMode.CENTROID_ONLY || mode == FeatureMode.RAW_CENTROID);
        boolean needKMeans = (mode == FeatureMode.ALL || mode == FeatureMode.RAW_KMEANS);
        boolean needGA = (mode == FeatureMode.ALL || mode == FeatureMode.RAW_GA);

        if (needCentroids) {
            centroidCache = FeatureExtractor.calculateCentroids(trainingSet);
        }
        if (needKMeans) {
            kmeansCentroidCache = FeatureExtractor.computeKMeansCentroids(trainingSet, Constants.KMEANS_CLUSTERS);
        }
        if (needGA) {
            gaWeightsCache = FeatureExtractor.evolveGeneticWeights(trainingSet);
        }

        int numSamples = trainingSet.size();
        double[] firstSample = projectToFeatureSpace(trainingSet.get(0));
        int featureSize = firstSample.length;

        double[][] featureMatrix = new double[numSamples][featureSize];
        int[] labels = new int[numSamples];

        for (int sampleIndex = 0; sampleIndex < numSamples; sampleIndex++) {
            List<Integer> row = trainingSet.get(sampleIndex);
            featureMatrix[sampleIndex] = projectToFeatureSpace(row);
            labels[sampleIndex] = row.get(Constants.BITMAP_SIZE);
        }

        FeatureNormalizer normalizer = new FeatureNormalizer(featureSize);
        normalizer.fitAndTransform(featureMatrix);

        this.oneVsRestModel = trainPerceptronModel(featureMatrix, labels, normalizer);

        this.oneVsOneModels = new ArrayList<>();
        for (int classA = 0; classA < CLASSES; classA++) {
            for (int classB = classA + 1; classB < CLASSES; classB++) {
                List<Integer> pairIndices = new ArrayList<>();
                for (int sampleIndex = 0; sampleIndex < numSamples; sampleIndex++) {
                    if (labels[sampleIndex] == classA || labels[sampleIndex] == classB) {
                        pairIndices.add(sampleIndex);
                    }
                }

                if (!pairIndices.isEmpty()) {
                    SVMModel pairModel = trainLinearPerceptron(featureMatrix, labels, pairIndices, classA, classB);
                    oneVsOneModels.add(pairModel);
                }
            }
        }
    }

    private double[] projectToFeatureSpace(List<Integer> sample) {
        double[] rawPixels = FeatureExtractor.buildRawPixelsVector(sample);

        switch (mode) {
            case CENTROID_ONLY:
                return FeatureExtractor.buildCentroidDistanceVector(sample, centroidCache);
            case RAW_CENTROID:
                return FeatureExtractor.concatVectors(rawPixels, FeatureExtractor.buildCentroidDistanceVector(sample, centroidCache));
            case RAW_KMEANS:
                return FeatureExtractor.concatVectors(rawPixels, FeatureExtractor.buildKMeansDistanceVector(sample, kmeansCentroidCache));
            case RAW_GA:
                return FeatureExtractor.concatVectors(rawPixels, FeatureExtractor.buildGAWeightedVector(sample, gaWeightsCache));
            case ALL:
            default:
                return FeatureExtractor.buildCombinedFeatureVector(sample, centroidCache, kmeansCentroidCache, gaWeightsCache);
        }
    }

    private int classifyOneVsOne(double[] features) {
        int[] votes = new int[CLASSES];
        for (SVMModel model : oneVsOneModels) {
            double score = model.pointInFeatureSpace(0, features) + model.bias[0];
            if (score >= 0) {
                votes[model.positiveClassLabel]++;
            } else {
                votes[model.negativeClassLabel]++;
            }
        }
        int bestClass = 0;
        int maxVotes = -1;
        for (int classIndex = 0; classIndex < CLASSES; classIndex++) {
            if (votes[classIndex] > maxVotes) {
                maxVotes = votes[classIndex];
                bestClass = classIndex;
            }
        }
        return bestClass;
    }

    private SVMModel trainPerceptronModel(double[][] features, int[] labels, FeatureNormalizer normalizer) {
        int featureSize = features[0].length;
        SVMModel model = new SVMModel(CLASSES, featureSize, normalizer);
        int numSamples = features.length;
        List<Integer> indices = new ArrayList<>();
        for (int sampleIndex = 0; sampleIndex < numSamples; sampleIndex++) {
            indices.add(sampleIndex);
        }
        Random rand = new Random(RANDOM_SEED);

        for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
            Collections.shuffle(indices, rand);
            int mistakes = 0;
            for (int idx : indices) {
                double[] sample = features[idx];
                int target = labels[idx];
                PredictionResult res = model.computeScores(sample);

                int bestRival = -1;
                double bestRivalScore = Double.NEGATIVE_INFINITY;
                for (int classIndex = 0; classIndex < CLASSES; classIndex++) {
                    if (classIndex == target) {
                        continue;
                    }
                    if (res.scores[classIndex] > bestRivalScore) {
                        bestRivalScore = res.scores[classIndex];
                        bestRival = classIndex;
                    }
                }

                boolean violation = (res.scores[target] - bestRivalScore) <= MARGIN;
                if (violation) {
                    mistakes++;
                    model.update(sample, target, 1.0);
                    if (bestRival >= 0) {
                        model.update(sample, bestRival, -1.0);
                    }
                }
                model.accumulateAverages();
            }
            if (mistakes == 0) {
                break;
            }
        }
        model.finalizeWeights();
        return model;
    }

    private SVMModel trainLinearPerceptron(double[][] features, int[] labels, List<Integer> indices, int classA, int classB) {
        int featureSize = features[0].length;
        SVMModel model = new SVMModel(1, featureSize, null);
        model.positiveClassLabel = classA;
        model.negativeClassLabel = classB;
        Random rand = new Random(RANDOM_SEED + classA * 100 + classB);

        for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
            Collections.shuffle(indices, rand);
            boolean updated = false;
            for (int idx : indices) {
                int targetLabel = (labels[idx] == classA) ? 1 : -1;
                double score = model.pointInFeatureSpace(0, features[idx]) + model.bias[0];

                if (targetLabel * score <= MARGIN) {
                    for (int featureIndex = 0; featureIndex < featureSize; featureIndex++) {
                        model.weights[0][featureIndex] += LEARNING_RATE * targetLabel * features[idx][featureIndex];
                    }
                    model.bias[0] += LEARNING_RATE * targetLabel;
                    updated = true;
                }
                model.accumulateAverages();
            }
            if (!updated) {
                break;
            }
        }
        model.finalizeWeights();
        return model;
    }
}