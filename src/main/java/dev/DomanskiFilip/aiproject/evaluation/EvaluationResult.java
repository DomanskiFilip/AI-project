package evaluation;

// Class to hold evaluation results for calculation, display and use in algorithms like "All AT Once"
public class EvaluationResult {
    public String algorithmName;
    public int correctMatches;
    public int total;
    public double successRate;
    public double evaluationTime;
    
    public int correctOneVsRest;
    public int correctOneVsOne;
    public double successRateOneVsRest;
    public double successRateOneVsOne;
    public boolean isSplitResult;
    
    public EvaluationResult(String name, int correct, int total, double time) {
        this.algorithmName = name;
        this.correctMatches = correct;
        this.total = total;
        this.successRate = (correct / (double) total) * 100;
        this.evaluationTime = time;
        this.isSplitResult = false;
    }
    
    public EvaluationResult(String name, int correctOvR, int correctOvO, int total, double time) {
        this.algorithmName = name;
        this.correctOneVsRest = correctOvR;
        this.correctOneVsOne = correctOvO;
        this.total = total;
        this.successRateOneVsRest = (correctOvR / (double) total) * 100;
        this.successRateOneVsOne = (correctOvO / (double) total) * 100;
        this.evaluationTime = time;
        this.isSplitResult = true;
    }
}
