namespace NeuralNetwork
{
    public class LayerGradient
    {
        public float[] BiasGradients
        {
            get; 
            set;
        }
    
        public float[,] WeightGradients
        {
            get;
            set;
        }
    
        public LayerGradient(float[] biasGradients, float[,] weightGradients)
        {
            BiasGradients = biasGradients;
            WeightGradients = weightGradients;
        }
    }
}
