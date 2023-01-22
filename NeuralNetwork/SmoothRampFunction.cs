namespace NeuralNetwork
{
    public class SmoothRampFunction : ActivationFunction
    {
        public double ValueAt(double x)
        {
            return Math.Log(Math.Exp(x) + 1);
        }

        public double DerivativeAt(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }
    }
}
