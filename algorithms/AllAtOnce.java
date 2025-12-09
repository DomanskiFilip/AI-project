package algorithms;

import features.FeatureMode;
import java.util.List;

// An ensemble algorithm that combines multiple algorithms' predictions and picks the most voted digit as prediction
public class AllAtOnce implements Algorithm {
    
    private SupportVectorMachine svmCentroidOnly;
    private SupportVectorMachine svmAll;
    private SupportVectorMachine svmRawCentroid;
    private SupportVectorMachine svmRawKMeans;
    private SupportVectorMachine svmRawGA;
    
    private MultiLayerPerceptron mlpRawOnly;
    private MultiLayerPerceptron mlpCentroidOnly;
    private MultiLayerPerceptron mlpAll;
    private MultiLayerPerceptron mlpRawCentroid;
    private MultiLayerPerceptron mlpRawKMeans;
    private MultiLayerPerceptron mlpRawGA;
    
    private boolean trained = false;
    
    @Override
    public Object predict(List<Integer> sample, List<List<Integer>> trainingSet) {
        if (!trained) {
            System.out.println("Training SVM and MLP variants (this will take a moment)...");
            trainAllSVMs(trainingSet);
            trainAllMLPs(trainingSet);
            trained = true;
            System.out.println("variants training complete!");
        }
        
        Algorithm[] algorithms = {
            new EuclideanDistance(),
            new DistanceFromCentroid(),
            new KNearestNeighbour(),
            new MahalanobisDistance(),
            svmCentroidOnly,
            svmAll,
            svmRawCentroid,
            svmRawKMeans,
            svmRawGA,
            mlpRawOnly,
            mlpCentroidOnly,
            mlpAll,
            mlpRawCentroid,
            mlpRawKMeans,
            mlpRawGA
        };

        int[] votes = new int[10];

        for (Algorithm algorithm : algorithms) {
            Object result = algorithm.predict(sample, trainingSet);
            
            if (result instanceof int[]) {
                int[] intResult = (int[]) result;
                for (int prediction : intResult) {
                    if (prediction >= 0 && prediction < 10) {
                        votes[prediction]++;
                    }
                }
            } 
            else if (result instanceof Integer) {
                int prediction = (Integer) result;
                if (prediction >= 0 && prediction < 10) {
                    votes[prediction]++;
                }
            }
        }

        int maxVotes = 0;
        int bestDigit = 0;
        
        for (int digit = 0; digit < votes.length; digit++) {
            if (votes[digit] > maxVotes) {
                maxVotes = votes[digit];
                bestDigit = digit;
            }
        }

        return bestDigit;
    }
    
    private void trainAllSVMs(List<List<Integer>> trainingSet) {
        System.out.println("  Training SVM variants...");
        
        svmCentroidOnly = new SupportVectorMachine(FeatureMode.CENTROID_ONLY);
        svmCentroidOnly.predict(trainingSet.get(0), trainingSet);
        
        svmAll = new SupportVectorMachine(FeatureMode.ALL);
        svmAll.predict(trainingSet.get(0), trainingSet);
        
        svmRawCentroid = new SupportVectorMachine(FeatureMode.RAW_CENTROID);
        svmRawCentroid.predict(trainingSet.get(0), trainingSet);
        
        svmRawKMeans = new SupportVectorMachine(FeatureMode.RAW_KMEANS);
        svmRawKMeans.predict(trainingSet.get(0), trainingSet);
        
        svmRawGA = new SupportVectorMachine(FeatureMode.RAW_GA);
        svmRawGA.predict(trainingSet.get(0), trainingSet);
        
        System.out.println("  SVM training complete!");
    }

    private void trainAllMLPs(List<List<Integer>> trainingSet) {
        System.out.println("  Training MLP variants...");
        
        mlpRawOnly = new MultiLayerPerceptron(FeatureMode.RAW_ONLY);
        mlpRawOnly.predict(trainingSet.get(0), trainingSet);
        
        mlpCentroidOnly = new MultiLayerPerceptron(FeatureMode.CENTROID_ONLY);
        mlpCentroidOnly.predict(trainingSet.get(0), trainingSet);
        
        mlpAll = new MultiLayerPerceptron(FeatureMode.ALL);
        mlpAll.predict(trainingSet.get(0), trainingSet);
        
        mlpRawCentroid = new MultiLayerPerceptron(FeatureMode.RAW_CENTROID);
        mlpRawCentroid.predict(trainingSet.get(0), trainingSet);
        
        mlpRawKMeans = new MultiLayerPerceptron(FeatureMode.RAW_KMEANS);
        mlpRawKMeans.predict(trainingSet.get(0), trainingSet);
        
        mlpRawGA = new MultiLayerPerceptron(FeatureMode.RAW_GA);
        mlpRawGA.predict(trainingSet.get(0), trainingSet);
        
        System.out.println("  MLP training complete!");
    }
    
    public SupportVectorMachine getSvmCentroidOnly() { return svmCentroidOnly; }
    public SupportVectorMachine getSvmAll() { return svmAll; }
    public SupportVectorMachine getSvmRawCentroid() { return svmRawCentroid; }
    public SupportVectorMachine getSvmRawKMeans() { return svmRawKMeans; }
    public SupportVectorMachine getSvmRawGA() { return svmRawGA; }
    
    public MultiLayerPerceptron getMlpRawOnly() { return mlpRawOnly; }
    public MultiLayerPerceptron getMlpCentroidOnly() { return mlpCentroidOnly; }
    public MultiLayerPerceptron getMlpAll() { return mlpAll; }
    public MultiLayerPerceptron getMlpRawCentroid() { return mlpRawCentroid; }
    public MultiLayerPerceptron getMlpRawKMeans() { return mlpRawKMeans; }
    public MultiLayerPerceptron getMlpRawGA() { return mlpRawGA; }
}