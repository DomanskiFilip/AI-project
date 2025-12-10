package features;

import constants.Constants;
import java.util.List;
import java.util.Random;

// A class to extract and use kernel functions and other calculations on features from the dataset for various algorithms
public class FeatureExtractor {

    // Calculate centroids for each class in the training set
    public static double[][] calculateCentroids(List<List<Integer>> trainingSet) {
        double[][] centroids = new double[10][Constants.BITMAP_SIZE];
        double[][] sumPerClass = new double[10][Constants.BITMAP_SIZE];
        int[] countPerClass = new int[10];

        for (List<Integer> row : trainingSet) {
            int digitClass = row.get(Constants.BITMAP_SIZE);
            countPerClass[digitClass]++;
            for (int featureIndex = 0; featureIndex < Constants.BITMAP_SIZE; featureIndex++) {
                sumPerClass[digitClass][featureIndex] += row.get(featureIndex);
            }
        }

        for (int digit = 0; digit < 10; digit++) {
            if (countPerClass[digit] > 0) {
                for (int featureIndex = 0; featureIndex < Constants.BITMAP_SIZE; featureIndex++) {
                    centroids[digit][featureIndex] = sumPerClass[digit][featureIndex] / countPerClass[digit];
                }
            }
        }
        return centroids;
    }

    // K-Means clustering to compute centroids for the dataset
    public static double[][] computeKMeansCentroids(List<List<Integer>> trainingSet, int clusters) {
        if (trainingSet == null || trainingSet.isEmpty()) {
            return new double[clusters][Constants.BITMAP_SIZE];
        }
        Random random = new Random(42);
        double[][] centroids = new double[clusters][Constants.BITMAP_SIZE];

        List<Integer> firstSample = trainingSet.get(random.nextInt(trainingSet.size()));
        for (int featureIndex = 0; featureIndex < Constants.BITMAP_SIZE; featureIndex++) {
            centroids[0][featureIndex] = firstSample.get(featureIndex);
        }

        for (int clusterIndex = 1; clusterIndex < clusters; clusterIndex++) {
            double[] distances = new double[trainingSet.size()];
            double totalDistance = 0;
            for (int sampleIndex = 0; sampleIndex < trainingSet.size(); sampleIndex++) {
                List<Integer> sample = trainingSet.get(sampleIndex);
                double minDistance = Double.MAX_VALUE;
                for (int existingCluster = 0; existingCluster < clusterIndex; existingCluster++) {
                    double distance = 0;
                    for (int featureIndex = 0; featureIndex < Constants.BITMAP_SIZE; featureIndex++) {
                        double diff = sample.get(featureIndex) - centroids[existingCluster][featureIndex];
                        distance += diff * diff;
                    }
                    if (distance < minDistance) {
                        minDistance = distance;
                    }
                }
                distances[sampleIndex] = minDistance;
                totalDistance += minDistance;
            }
            double randomValue = random.nextDouble() * totalDistance;
            double cumulativeDistance = 0;
            for (int sampleIndex = 0; sampleIndex < distances.length; sampleIndex++) {
                cumulativeDistance += distances[sampleIndex];
                if (cumulativeDistance >= randomValue) {
                    List<Integer> pickedSample = trainingSet.get(sampleIndex);
                    for (int featureIndex = 0; featureIndex < Constants.BITMAP_SIZE; featureIndex++) {
                        centroids[clusterIndex][featureIndex] = pickedSample.get(featureIndex);
                    }
                    break;
                }
            }
        }

        for (int iteration = 0; iteration < Constants.KMEANS_MAX_ITERATIONS; iteration++) {
            double[][] sums = new double[clusters][Constants.BITMAP_SIZE];
            int[] counts = new int[clusters];

            for (List<Integer> sample : trainingSet) {
                int bestCluster = 0;
                double bestDistance = Double.MAX_VALUE;
                for (int clusterIndex = 0; clusterIndex < clusters; clusterIndex++) {
                    double distance = 0;
                    for (int featureIndex = 0; featureIndex < Constants.BITMAP_SIZE; featureIndex++) {
                        double diff = sample.get(featureIndex) - centroids[clusterIndex][featureIndex];
                        distance += diff * diff;
                    }
                    if (distance < bestDistance) {
                        bestDistance = distance;
                        bestCluster = clusterIndex;
                    }
                }
                counts[bestCluster]++;
                for (int featureIndex = 0; featureIndex < Constants.BITMAP_SIZE; featureIndex++)
                    sums[bestCluster][featureIndex] += sample.get(featureIndex);
            }

            boolean centroidsMoved = false;
            for (int clusterIndex = 0; clusterIndex < clusters; clusterIndex++) {
                if (counts[clusterIndex] > 0) {
                    for (int featureIndex = 0; featureIndex < Constants.BITMAP_SIZE; featureIndex++) {
                        double newValue = sums[clusterIndex][featureIndex] / counts[clusterIndex];
                        if (Math.abs(newValue - centroids[clusterIndex][featureIndex]) > 1e-6) {
                            centroidsMoved = true;
                        }
                        centroids[clusterIndex][featureIndex] = newValue;
                    }
                }
            }
            if (!centroidsMoved) {
                break;
            }
        }
        return centroids;
    }

    // Compute distances from a sample to K-Means centroids
    public static double[] computeKMeansDistances(List<Integer> sample, double[][] kmeansCentroids) {
        if (kmeansCentroids == null) {
            return new double[0];
        }
        int clusters = kmeansCentroids.length;
        double[] distances = new double[clusters];
        for (int clusterIndex = 0; clusterIndex < clusters; clusterIndex++) {
            double sum = 0;
            for (int featureIndex = 0; featureIndex < Constants.BITMAP_SIZE; featureIndex++) {
                double diff = sample.get(featureIndex) - kmeansCentroids[clusterIndex][featureIndex];
                sum += diff * diff;
            }
            distances[clusterIndex] = sum;
        }
        return distances;
    }

    // Evolve genetic algorithm weights for feature selection
    public static double[] evolveGeneticWeights(List<List<Integer>> trainingSet) {
        Random randomGenerator = new Random(42);
        List<double[]> population = new java.util.ArrayList<>();
        for (int individualIndex = 0; individualIndex < Constants.GA_POPULATION; individualIndex++) {
            double[] individual = new double[Constants.BITMAP_SIZE];
            for (int featureIndex = 0; featureIndex < Constants.BITMAP_SIZE; featureIndex++)
                individual[featureIndex] = randomGenerator.nextDouble();
            population.add(individual);
        }

        double fitnessThreshold = 0.99;
        int maxGenerations = Constants.GA_GENERATIONS * 5;
        double bestScore = -1;
        double[] bestIndividual = null;
        for (int generation = 0; generation < maxGenerations; generation++) {
            double[] fitnessScores = new double[population.size()];
            int bestIndex = 0;
            for (int individualIndex = 0; individualIndex < population.size(); individualIndex++) {
                fitnessScores[individualIndex] = evaluateWeightFitness(population.get(individualIndex), trainingSet);
                if (fitnessScores[individualIndex] > bestScore) {
                    bestScore = fitnessScores[individualIndex];
                    bestIndividual = population.get(individualIndex);
                    bestIndex = individualIndex;
                }
            }
            if (bestScore >= fitnessThreshold) {
                break;
            }

            List<double[]> nextGeneration = new java.util.ArrayList<>();
            nextGeneration.add(bestIndividual.clone());
            while (nextGeneration.size() < population.size()) {
                int parent1Index = randomGenerator.nextInt(population.size());
                int parent2Index = randomGenerator.nextInt(population.size());
                if (parent1Index == bestIndex) {
                    parent1Index = (parent1Index + 1) % population.size();
                }
                if (parent2Index == bestIndex) {
                    parent2Index = (parent2Index + 1) % population.size();
                }
                double[] parent1 = population.get(parent1Index);
                double[] parent2 = population.get(parent2Index);
                double[] childIndividual = new double[Constants.BITMAP_SIZE];
                for (int featureIndex = 0; featureIndex < Constants.BITMAP_SIZE; featureIndex++) {
                    childIndividual[featureIndex] = randomGenerator.nextDouble() < 0.5 ? parent1[featureIndex] : parent2[featureIndex];
                }
                    
                for (int featureIndex = 0; featureIndex < Constants.BITMAP_SIZE; featureIndex++) {
                    if (randomGenerator.nextDouble() < Constants.GA_MUTATION) {
                         childIndividual[featureIndex] += randomGenerator.nextGaussian() * 0.1;
                    }
                }
                   
                nextGeneration.add(childIndividual);
            }
            population = nextGeneration;
        }
        return bestIndividual;
    }

    // Evaluate fitness of weights based on classification accuracy
    private static double evaluateWeightFitness(double[] weights, List<List<Integer>> trainingSet) {
        double[][] centroids = calculateCentroids(trainingSet);
        int correctMatches = 0;
        for (List<Integer> sample : trainingSet) {
            int bestClass = 0;
            double bestDistance = Double.MAX_VALUE;
            for (int classIndex = 0; classIndex < centroids.length; classIndex++) {
                double distance = 0;
                for (int featureIndex = 0; featureIndex < Constants.BITMAP_SIZE; featureIndex++) {
                    double diff = (sample.get(featureIndex) * weights[featureIndex]) - centroids[classIndex][featureIndex];
                    distance += diff * diff;
                }
                if (distance < bestDistance) {
                    bestDistance = distance;
                    bestClass = classIndex;
                }
            }
            if (bestClass == sample.get(Constants.BITMAP_SIZE)) {
                correctMatches++;
            }
        }
        return correctMatches / (double) Math.max(1, trainingSet.size());
    }

    // Build raw pixel vector from sample
    public static double[] buildRawPixelsVector(List<Integer> sample) {
        double[] vector = new double[Constants.BITMAP_SIZE];
        for (int pixelIndex = 0; pixelIndex < Constants.BITMAP_SIZE; pixelIndex++) {
             vector[pixelIndex] = sample.get(pixelIndex);
        }
        return vector;
    }

    // Build centroid distance vector from sample
    public static double[] buildCentroidDistanceVector(List<Integer> sample, double[][] centroids) {
        int numCentroids = (centroids != null) ? centroids.length : 0;
        double[] vector = new double[numCentroids];
        for (int centroidIndex = 0; centroidIndex < numCentroids; centroidIndex++) {
            double sum = 0;
            for (int featureIndex = 0; featureIndex < Constants.BITMAP_SIZE; featureIndex++) {
                double diff = sample.get(featureIndex) - centroids[centroidIndex][featureIndex];
                sum += diff * diff;
            }
            vector[centroidIndex] = Math.sqrt(sum);
        }
        return vector;
    }

    // Build K-Means distance vector from sample
    public static double[] buildKMeansDistanceVector(List<Integer> sample, double[][] kmeansCentroids) {
        if (kmeansCentroids == null) {
            return new double[0];
        }
        int clusters = kmeansCentroids.length;
        double[] vector = new double[clusters];
        for (int clusterIndex = 0; clusterIndex < clusters; clusterIndex++) {
            double sum = 0;
            for (int featureIndex = 0; featureIndex < Constants.BITMAP_SIZE; featureIndex++) {
                double diff = sample.get(featureIndex) - kmeansCentroids[clusterIndex][featureIndex];
                sum += diff * diff;
            }
            vector[clusterIndex] = Math.sqrt(sum);
        }
        return vector;
    }

    // Build GA weighted vector from sample
    public static double[] buildGAWeightedVector(List<Integer> sample, double[] gaWeights) {
        double[] vector = new double[Constants.BITMAP_SIZE];
        for (int pixelIndex = 0; pixelIndex < Constants.BITMAP_SIZE; pixelIndex++) {
            vector[pixelIndex] = sample.get(pixelIndex) * (gaWeights != null ? gaWeights[pixelIndex] : 1.0);
        }
        return vector;
    }

    // Concatenate multiple vectors into one
    public static double[] concatVectors(double[]... parts) {
        int totalLength = 0;
        for (double[] part : parts) {
            if (part != null) {
                totalLength += part.length;
            }
        }
        double[] output = new double[totalLength];
        int position = 0;
        for (double[] part : parts) {
            if (part == null) {
                continue;
            }
            System.arraycopy(part, 0, output, position, part.length);
            position += part.length;
        }
        return output;
    }

    // Build combined feature vector from sample
    public static double[] buildCombinedFeatureVector(List<Integer> sample, double[][] centroidCache, double[][] kmeansCache, double[] gaWeightsCache) {
        double[] rawPixels = buildRawPixelsVector(sample);
        double[] centroidDistances = buildCentroidDistanceVector(sample, centroidCache);
        double[] kmeansDistances = buildKMeansDistanceVector(sample, kmeansCache);
        double[] gaWeighted = buildGAWeightedVector(sample, gaWeightsCache);

        return concatVectors(rawPixels, centroidDistances, kmeansDistances, gaWeighted);
    }
}