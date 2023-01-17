namespace NeuralNetwork
{
    public class BackpropagationResult
    {
        public NetworkGradient Gradient;
        public float Error;

        public BackpropagationResult(NetworkGradient gradient, float error)
        {
            Gradient = gradient;
            Error = error;
        }
    }
}
