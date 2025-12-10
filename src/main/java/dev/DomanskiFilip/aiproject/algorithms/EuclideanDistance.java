package algorithms;

import constants.Constants;
import java.util.List;

public class EuclideanDistance implements Algorithm {

    @Override
    public Object predict(List<Integer> sample, List<List<Integer>> trainingSet) {
        double minDistance = Double.MAX_VALUE;
        List<Integer> closest = null;
        for (List<Integer> candidate : trainingSet) {
            double sum = 0;
            for (int pixelIndex = 0; pixelIndex < Constants.BITMAP_SIZE; pixelIndex++) {
                double distance = sample.get(pixelIndex) - candidate.get(pixelIndex);
                sum += distance * distance;
            }
            if (sum < minDistance) {
                minDistance = sum;
                closest = candidate;
            }
        }
        return closest != null ? Integer.valueOf(closest.get(Constants.BITMAP_SIZE)) : Integer.valueOf(-1);
    }
}