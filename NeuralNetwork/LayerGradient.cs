namespace NeuralNetwork
{
    public class LayerGradient
    {
        public double[] BiasGradients
        {
            get; 
            set;
        }
    
        public double[,] WeightGradients
        {
            get;
            set;
        }
    
        public LayerGradient(double[] biasGradients, double[,] weightGradients)
        {
            BiasGradients = biasGradients;
            WeightGradients = weightGradients;
        }
    }
}
