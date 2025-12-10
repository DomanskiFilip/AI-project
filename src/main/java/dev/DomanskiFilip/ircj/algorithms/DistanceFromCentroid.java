package algorithms;

import constants.Constants;
import features.FeatureExtractor;
import java.util.List;

public class DistanceFromCentroid implements Algorithm {
    @Override
    public Object predict(List<Integer> sample, List<List<Integer>> trainingSet) {
        double[][] centroids = FeatureExtractor.calculateCentroids(trainingSet);
        double minDistance = Double.MAX_VALUE;
        int closestClass = -1;

        for (int digit = 0; digit < 10; digit++) {
            double sum = 0;
            for (int pixelIndex = 0; pixelIndex < Constants.BITMAP_SIZE; pixelIndex++) {
                double diff = sample.get(pixelIndex) - centroids[digit][pixelIndex];
                sum += diff * diff;
            }
            double distance = Math.sqrt(sum);
            if (distance < minDistance) {
                minDistance = distance;
                closestClass = digit;
            }
        }
        return Integer.valueOf(closestClass);
    }
}
