package algorithms;

import constants.Constants;
import features.FeatureExtractor;
import features.FeatureMode;
import java.util.List;
import java.util.Random;

public class MultiLayerPerceptron implements Algorithm {
    
    private static final int PERCEPTRONS = 300;
    private static final int EPOCHS = 50;
    private static final double LEARNING_RATE = 0.1;
    private static final long RANDOM_SEED = 42;
    private static final int CLASSES = 10;

    private final FeatureMode mode;
    
    private double[][] weightsInputHidden;
    private double[] biasHidden;
    private double[][] weightsHiddenOutput;
    private double[] biasOutput;
    private boolean trained = false;
    
    private double[][] centroidCache;
    private double[][] kmeansCentroidCache;
    private double[] gaWeightsCache;
    private int inputSize;

    private double[] featureMean;
    private double[] featureStd;

    public MultiLayerPerceptron() {
        this(FeatureMode.RAW_ONLY);
    }
    
    public MultiLayerPerceptron(FeatureMode mode) {
        this.mode = mode;
    }

    @Override
    public Object predict(List<Integer> sample, List<List<Integer>> trainingSet) {
        if (!trained) {
            train(trainingSet);
        }
        
        double[] inputs = buildFeatureVector(sample);

        if (mode != FeatureMode.RAW_ONLY){
            inputs = normalizeFeatures(inputs);
        }
        
        double[] outputs = forward(inputs);
        
        int bestIndex = 0;
        double bestValue = outputs[0];
        for (int outputIndex = 1; outputIndex < outputs.length; outputIndex++) {
            if (outputs[outputIndex] > bestValue) {
                bestValue = outputs[outputIndex];
                bestIndex = outputIndex;
            }
        }
        return Integer.valueOf(bestIndex);
    }

    private void train(List<List<Integer>> trainingSet) {
        computeFeatureCaches(trainingSet);
        
        inputSize = buildFeatureVector(trainingSet.get(0)).length;
        
        if (mode != FeatureMode.RAW_ONLY){
            computeNormalizationStats(trainingSet);
        }
        
        initializeWeights();

        for (int epoch = 0; epoch < EPOCHS; epoch++) {
            for (List<Integer> row : trainingSet) {
                double[] inputs = buildFeatureVector(row);
                if (mode != FeatureMode.RAW_ONLY) {
                    inputs = normalizeFeatures(inputs);
                }
                int targetClass = row.get(Constants.BITMAP_SIZE);
                double[] hidden = new double[PERCEPTRONS];
                double[] outputs = new double[CLASSES];

                for (int hiddenIndex = 0; hiddenIndex < PERCEPTRONS; hiddenIndex++) {
                    double sum = biasHidden[hiddenIndex];
                    for (int inputIndex = 0; inputIndex < inputSize; inputIndex++) {
                        sum += weightsInputHidden[hiddenIndex][inputIndex] * inputs[inputIndex];
                    }
                    hidden[hiddenIndex] = sigmoid(sum);
                }

                for (int outputIndex = 0; outputIndex < CLASSES; outputIndex++) {
                    double sum = biasOutput[outputIndex];
                    for (int hiddenIndex = 0; hiddenIndex < PERCEPTRONS; hiddenIndex++) {
                        sum += weightsHiddenOutput[outputIndex][hiddenIndex] * hidden[hiddenIndex];
                    }
                    outputs[outputIndex] = sigmoid(sum);
                }

                double[] target = new double[CLASSES];
                target[targetClass] = 1.0;

                double[] outputDeltas = new double[CLASSES];
                for (int outputIndex = 0; outputIndex < CLASSES; outputIndex++) {
                    double error = target[outputIndex] - outputs[outputIndex];
                    outputDeltas[outputIndex] = error * sigmoidDerivative(outputs[outputIndex]);
                }

                double[] hiddenDeltas = new double[PERCEPTRONS];
                for (int hiddenIndex = 0; hiddenIndex < PERCEPTRONS; hiddenIndex++) {
                    double error = 0;
                    for (int outputIndex = 0; outputIndex < CLASSES; outputIndex++) {
                        error += outputDeltas[outputIndex] * weightsHiddenOutput[outputIndex][hiddenIndex];
                    }
                    hiddenDeltas[hiddenIndex] = error * sigmoidDerivative(hidden[hiddenIndex]);
                }

                for (int hiddenIndex = 0; hiddenIndex < PERCEPTRONS; hiddenIndex++) {
                    for (int inputIndex = 0; inputIndex < inputSize; inputIndex++) {
                        weightsInputHidden[hiddenIndex][inputIndex] += LEARNING_RATE * hiddenDeltas[hiddenIndex] * inputs[inputIndex];
                    }
                    biasHidden[hiddenIndex] += LEARNING_RATE * hiddenDeltas[hiddenIndex];
                }

                for (int outputIndex = 0; outputIndex < CLASSES; outputIndex++) {
                    for (int hiddenIndex = 0; hiddenIndex < PERCEPTRONS; hiddenIndex++) {
                        weightsHiddenOutput[outputIndex][hiddenIndex] += LEARNING_RATE * outputDeltas[outputIndex] * hidden[hiddenIndex];
                    }
                    biasOutput[outputIndex] += LEARNING_RATE * outputDeltas[outputIndex];
                }
            }
        }

        trained = true;
    }

    private void computeFeatureCaches(List<List<Integer>> trainingSet) {
        boolean needCentroids = (mode == FeatureMode.ALL || mode == FeatureMode.CENTROID_ONLY || 
                                mode == FeatureMode.RAW_CENTROID);
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
    }

    private double[] buildFeatureVector(List<Integer> sample) {
        double[] rawPixels = FeatureExtractor.buildRawPixelsVector(sample);

        switch (mode) {
            case RAW_ONLY:
                return rawPixels;
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

    private void computeNormalizationStats(List<List<Integer>> trainingSet) {
        featureMean = new double[inputSize];
        featureStd = new double[inputSize];
        
        for (List<Integer> row : trainingSet) {
            double[] features = buildFeatureVector(row);
            for (int featureIndex = 0; featureIndex < inputSize; featureIndex++) {
                featureMean[featureIndex] += features[featureIndex];
            }
        }
        for (int featureIndex = 0; featureIndex < inputSize; featureIndex++) {
            featureMean[featureIndex] /= trainingSet.size();
        }
        
        for (List<Integer> row : trainingSet) {
            double[] features = buildFeatureVector(row);
            for (int featureIndex = 0; featureIndex < inputSize; featureIndex++) {
                double diff = features[featureIndex] - featureMean[featureIndex];
                featureStd[featureIndex] += diff * diff;
            }
        }
        for (int featureIndex = 0; featureIndex < inputSize; featureIndex++) {
            featureStd[featureIndex] = Math.sqrt(featureStd[featureIndex] / trainingSet.size());
            if (featureStd[featureIndex] < 1e-9) {
                featureStd[featureIndex] = 1.0;
            }
        }
    }

    private double[] normalizeFeatures(double[] features) {
        double[] normalized = new double[features.length];
        for (int featureIndex = 0; featureIndex < features.length; featureIndex++) {
            normalized[featureIndex] = (features[featureIndex] - featureMean[featureIndex]) / featureStd[featureIndex];
        }
        return normalized;
    }

    private void initializeWeights() {
        weightsInputHidden = new double[PERCEPTRONS][inputSize];
        biasHidden = new double[PERCEPTRONS];
        weightsHiddenOutput = new double[CLASSES][PERCEPTRONS];
        biasOutput = new double[CLASSES];
        Random random = new Random(RANDOM_SEED);

        double range = 0.1;
        for (int hiddenIndex = 0; hiddenIndex < PERCEPTRONS; hiddenIndex++) {
            biasHidden[hiddenIndex] = (random.nextDouble() * 2 - 1) * range;
            for (int inputIndex = 0; inputIndex < inputSize; inputIndex++) {
                weightsInputHidden[hiddenIndex][inputIndex] = (random.nextDouble() * 2 - 1) * range;
            }
        }
        for (int outputIndex = 0; outputIndex < CLASSES; outputIndex++) {
            biasOutput[outputIndex] = (random.nextDouble() * 2 - 1) * range;
            for (int hiddenIndex = 0; hiddenIndex < PERCEPTRONS; hiddenIndex++) {
                weightsHiddenOutput[outputIndex][hiddenIndex] = (random.nextDouble() * 2 - 1) * range;
            }
        }
    }

    private double[] forward(double[] inputs) {
        double[] hidden = new double[PERCEPTRONS];
        for (int hiddenIndex = 0; hiddenIndex < PERCEPTRONS; hiddenIndex++) {
            double sum = biasHidden[hiddenIndex];
            for (int inputIndex = 0; inputIndex < inputSize; inputIndex++) {
                sum += weightsInputHidden[hiddenIndex][inputIndex] * inputs[inputIndex];
            }
            hidden[hiddenIndex] = sigmoid(sum);
        }

        double[] outputs = new double[CLASSES];
        for (int outputIndex = 0; outputIndex < CLASSES; outputIndex++) {
            double sum = biasOutput[outputIndex];
            for (int hiddenIndex = 0; hiddenIndex < PERCEPTRONS; hiddenIndex++) {
                sum += weightsHiddenOutput[outputIndex][hiddenIndex] * hidden[hiddenIndex];
            }
            outputs[outputIndex] = sigmoid(sum);
        }
        return outputs;
    }

    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    private double sigmoidDerivative(double activatedValue) {
        return activatedValue * (1.0 - activatedValue);
    }

    public String getParameters() {
        return "Mode: " + mode + ", Epochs: " + EPOCHS + ", Learning rate: " + LEARNING_RATE + ", Perceptrons: " + PERCEPTRONS +  ", Random seed: " + RANDOM_SEED;
    }
}
