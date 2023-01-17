namespace NeuralNetwork
{
    public class NetworkGradient
    {
        public LayerGradient[] LayerGradients
        {
            get;
            set;
        }

        public NetworkGradient(LayerGradient[] layerGradients)
        {
            LayerGradients = layerGradients;
        }
    }
}
