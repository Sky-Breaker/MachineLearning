namespace NeuralNetwork
{
    public interface ActivationFunction
    {
        public double ValueAt(double x);
        public double DerivativeAt(double x);
    }
}
