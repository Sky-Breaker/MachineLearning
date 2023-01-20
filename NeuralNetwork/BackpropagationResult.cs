namespace NeuralNetwork
{
    public class BackpropagationResult
    {
        public NetworkGradient Gradient;
        public double Error;

        public BackpropagationResult(NetworkGradient gradient, double error)
        {
            Gradient = gradient;
            Error = error;
        }
    }
}
