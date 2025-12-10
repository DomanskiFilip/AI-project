package evaluation;

import algorithms.*;
import constants.Constants;
import features.FeatureMode;
import java.util.List;
import java.util.Map;
import java.util.LinkedHashMap;

// Evaluator class to assess algorithm performance and display results
public class Evaluator {

    public static EvaluationResult evaluateAlgorithmWithResults(List<List<Integer>> dataSetA, List<List<Integer>> dataSetB, Algorithm algorithm, String label) {
        long startTime = System.nanoTime();

        if (algorithm instanceof SupportVectorMachine svm) {
            if (!svm.isTrained()) {
                svm.predict(dataSetA.get(0), dataSetA);
            }
        }

        if (algorithm instanceof MultiLayerPerceptron) {
            System.out.println("\n--- " + label + " parameters used in calculation ---");
            System.out.println(((MultiLayerPerceptron) algorithm).getParameters());
        }

        int correctMatches = 0;
        int correctOneVsRest = 0;
        int correctOneVsOne = 0;
        boolean isSplitResult = false;

        for (List<Integer> sample : dataSetB) {
            int actualDigit = sample.get(Constants.BITMAP_SIZE);
            Object result = algorithm.predict(sample, dataSetA);

            if (result instanceof int[]) {
                isSplitResult = true;
                int[] predictions = (int[]) result;
                if (predictions[0] == actualDigit) {
                    correctOneVsRest++;
                }
                if (predictions[1] == actualDigit) {
                    correctOneVsOne++;
                }
            } else if (result instanceof Integer) {
                if ((Integer) result == actualDigit) {
                    correctMatches++;
                }
            }
        }

        long endTime = System.nanoTime();
        double duration = (endTime - startTime) / 1_000_000_000.0;
        int total = dataSetB.size();

        System.out.println("\n--- " + label + " Success Rate ---");
        if (isSplitResult) {
            System.out.printf("   One-vs-Rest Correct: %d / %d%n", correctOneVsRest, total);
            System.out.printf("   One-vs-Rest Success Rate: %.5f%%%n", (correctOneVsRest / (double) total) * 100);
            System.out.printf("   One-vs-One Correct: %d / %d%n", correctOneVsOne, total);
            System.out.printf("   One-vs-One Success Rate: %.5f%%%n", (correctOneVsOne / (double) total) * 100);
            return new EvaluationResult(label, correctOneVsRest, correctOneVsOne, total, duration);
        } else {
            System.out.printf("   Correct Matches: %d / %d%n", correctMatches, total);
            System.out.printf("   Success Rate: %.5f%%%n", (correctMatches / (double) total) * 100);
            System.out.println("   Evaluation Time: " + duration + " seconds");
            System.out.println("\n");
            return new EvaluationResult(label, correctMatches, total, duration);
        }
    }

    public static void evaluateAlgorithm(List<List<Integer>> dataSetA, List<List<Integer>> dataSetB, Algorithm algorithm, String label) {
        evaluateAlgorithmWithResults(dataSetA, dataSetB, algorithm, label);
    }

    public static void runAllInOrder(List<List<Integer>> dataSetA, List<List<Integer>> dataSetB) {
        Map<String, EvaluationResult> resultsAonB = new LinkedHashMap<>();
        Map<String, EvaluationResult> resultsBonA = new LinkedHashMap<>();
        
        System.out.println("\n========================================");
        System.out.println("Running All Algorithms in Sequence");
        System.out.println("Trained on dataset A tested on dataset B");
        System.out.println("========================================\n");
        
        resultsAonB.put("Euclidean Distance", evaluateAlgorithmWithResults(dataSetA, dataSetB, new EuclideanDistance(), "Euclidean Distance"));
        resultsAonB.put("Distance From Centroid", evaluateAlgorithmWithResults(dataSetA, dataSetB, new DistanceFromCentroid(), "Distance From Centroid"));
        resultsAonB.put("K Nearest Neighbour", evaluateAlgorithmWithResults(dataSetA, dataSetB, new KNearestNeighbour(), "K Nearest Neighbour"));
        resultsAonB.put("Mahalanobis Distance", evaluateAlgorithmWithResults(dataSetA, dataSetB, new MahalanobisDistance(), "Mahalanobis Distance"));
        
        AllAtOnce allsetA = new AllAtOnce();
        allsetA.predict(dataSetA.get(0), dataSetA);
        
        System.out.println("\n--- Multi-Layer Perceptron Variants ---");
        resultsAonB.put("MLP [Raw Only]", evaluateAlgorithmWithResults(dataSetA, dataSetB, allsetA.getMlpRawOnly(), "MLP [Raw Only]"));
        resultsAonB.put("MLP [Centroid Only]", evaluateAlgorithmWithResults(dataSetA, dataSetB, allsetA.getMlpCentroidOnly(), "MLP [Centroid Only]"));
        resultsAonB.put("MLP [Raw + Centroid]", evaluateAlgorithmWithResults(dataSetA, dataSetB, allsetA.getMlpRawCentroid(), "MLP [Raw + Centroid]"));
        resultsAonB.put("MLP [Raw + KMeans]", evaluateAlgorithmWithResults(dataSetA, dataSetB, allsetA.getMlpRawKMeans(), "MLP [Raw + KMeans]"));
        resultsAonB.put("MLP [Raw + GA]", evaluateAlgorithmWithResults(dataSetA, dataSetB, allsetA.getMlpRawGA(), "MLP [Raw + GA]"));
        resultsAonB.put("MLP [All Features]", evaluateAlgorithmWithResults(dataSetA, dataSetB, allsetA.getMlpAll(), "MLP [All Features]"));
        
        System.out.println("\n--- Support Vector Machine Variants ---");
        resultsAonB.put("SVM [Centroid Only]", evaluateAlgorithmWithResults(dataSetA, dataSetB, allsetA.getSvmCentroidOnly(), "SVM [Centroid Only]"));
        resultsAonB.put("SVM [Raw + Centroid]", evaluateAlgorithmWithResults(dataSetA, dataSetB, allsetA.getSvmRawCentroid(), "SVM [Raw + Centroid]"));
        resultsAonB.put("SVM [Raw + KMeans]", evaluateAlgorithmWithResults(dataSetA, dataSetB, allsetA.getSvmRawKMeans(), "SVM [Raw + KMeans]"));
        resultsAonB.put("SVM [Raw + GA]", evaluateAlgorithmWithResults(dataSetA, dataSetB, allsetA.getSvmRawGA(), "SVM [Raw + GA]"));
        resultsAonB.put("SVM [All Features]", evaluateAlgorithmWithResults(dataSetA, dataSetB, allsetA.getSvmAll(), "SVM [All Features]"));
        
        System.out.println("\n--- All Algorithms at once (pick most voted class) ---");
        resultsAonB.put("All at Once", evaluateAlgorithmWithResults(dataSetA, dataSetB, allsetA, "All at Once"));
        
        System.out.println("\n========================================");
        System.out.println("Running All Algorithms in Sequence");
        System.out.println("Trained on dataset B tested on dataset A");
        System.out.println("========================================\n");
        
        resultsBonA.put("Euclidean Distance", evaluateAlgorithmWithResults(dataSetB, dataSetA, new EuclideanDistance(), "Euclidean Distance"));
        resultsBonA.put("Distance From Centroid", evaluateAlgorithmWithResults(dataSetB, dataSetA, new DistanceFromCentroid(), "Distance From Centroid"));
        resultsBonA.put("K Nearest Neighbour", evaluateAlgorithmWithResults(dataSetB, dataSetA, new KNearestNeighbour(), "K Nearest Neighbour"));
        resultsBonA.put("Mahalanobis Distance", evaluateAlgorithmWithResults(dataSetB, dataSetA, new MahalanobisDistance(), "Mahalanobis Distance"));
        
        AllAtOnce allsetB = new AllAtOnce();
        allsetB.predict(dataSetB.get(0), dataSetB);
        
        System.out.println("\n--- Multi-Layer Perceptron Variants ---");
        resultsBonA.put("MLP [Raw Only]", evaluateAlgorithmWithResults(dataSetB, dataSetA, allsetB.getMlpRawOnly(), "MLP [Raw Only]"));
        resultsBonA.put("MLP [Centroid Only]", evaluateAlgorithmWithResults(dataSetB, dataSetA, allsetB.getMlpCentroidOnly(), "MLP [Centroid Only]"));
        resultsBonA.put("MLP [Raw + Centroid]", evaluateAlgorithmWithResults(dataSetB, dataSetA, allsetB.getMlpRawCentroid(), "MLP [Raw + Centroid]"));
        resultsBonA.put("MLP [Raw + KMeans]", evaluateAlgorithmWithResults(dataSetB, dataSetA, allsetB.getMlpRawKMeans(), "MLP [Raw + KMeans]"));
        resultsBonA.put("MLP [Raw + GA]", evaluateAlgorithmWithResults(dataSetB, dataSetA, allsetB.getMlpRawGA(), "MLP [Raw + GA]"));
        resultsBonA.put("MLP [All Features]", evaluateAlgorithmWithResults(dataSetB, dataSetA, allsetB.getMlpAll(), "MLP [All Features]"));
        
        System.out.println("\n--- Support Vector Machine Variants ---");
        resultsBonA.put("SVM [Centroid Only]", evaluateAlgorithmWithResults(dataSetB, dataSetA, allsetB.getSvmCentroidOnly(), "SVM [Centroid Only]"));
        resultsBonA.put("SVM [Raw + Centroid]", evaluateAlgorithmWithResults(dataSetB, dataSetA, allsetB.getSvmRawCentroid(), "SVM [Raw + Centroid]"));
        resultsBonA.put("SVM [Raw + KMeans]", evaluateAlgorithmWithResults(dataSetB, dataSetA, allsetB.getSvmRawKMeans(), "SVM [Raw + KMeans]"));
        resultsBonA.put("SVM [Raw + GA]", evaluateAlgorithmWithResults(dataSetB, dataSetA, allsetB.getSvmRawGA(), "SVM [Raw + GA]"));
        resultsBonA.put("SVM [All Features]", evaluateAlgorithmWithResults(dataSetB, dataSetA, allsetB.getSvmAll(), "SVM [All Features]"));
        
        System.out.println("\n--- All Algorithms at once (pick most voted class) ---");
        resultsBonA.put("All at Once", evaluateAlgorithmWithResults(dataSetB, dataSetA, allsetB, "All at Once"));
        
        System.out.println("\n========================================");
        System.out.println("Averages between both runs:");
        System.out.println("========================================\n");
        
        for (String algorithmName : resultsAonB.keySet()) {
            EvaluationResult resultAonB = resultsAonB.get(algorithmName);
            EvaluationResult resultBonA = resultsBonA.get(algorithmName);
            
            System.out.println("--- " + algorithmName + " ---");
            
            if (resultAonB.isSplitResult && resultBonA.isSplitResult) {
                double avgOneVsRest = (resultAonB.successRateOneVsRest + resultBonA.successRateOneVsRest) / 2.0;
                double avgOneVsOne = (resultAonB.successRateOneVsOne + resultBonA.successRateOneVsOne) / 2.0;
                double avgTime = (resultAonB.evaluationTime + resultBonA.evaluationTime) / 2.0;
                
                System.out.printf("   Average One-vs-Rest Success Rate: %.5f%%%n", avgOneVsRest);
                System.out.printf("   Average One-vs-One Success Rate: %.5f%%%n", avgOneVsOne);
                System.out.printf("   Average Evaluation Time: %.5f seconds%n", avgTime);
            } else {
                double avgSuccessRate = (resultAonB.successRate + resultBonA.successRate) / 2.0;
                double avgTime = (resultAonB.evaluationTime + resultBonA.evaluationTime) / 2.0;
                
                System.out.printf("   A>B Success Rate: %.5f%%%n", resultAonB.successRate);
                System.out.printf("   B>A Success Rate: %.5f%%%n", resultBonA.successRate);
                System.out.printf("   Average Success Rate: %.5f%%%n", avgSuccessRate);
                System.out.printf("   Average Evaluation Time: %.5f seconds%n", avgTime);
            }
            System.out.println();
        }
        
        System.out.println("========================================");
        System.out.println("Evaluation Complete!");
        System.out.println("========================================\n");
    }
}