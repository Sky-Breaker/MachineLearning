namespace NeuralNetwork
{
    public class SigmoidFunction : ActivationFunction
    {
        public double ValueAt(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }

        public double DerivativeAt(double x)
        {
            double eToX = Math.Exp(x);
            return eToX / Math.Pow(eToX + 1, 2);
        }

    }
}
