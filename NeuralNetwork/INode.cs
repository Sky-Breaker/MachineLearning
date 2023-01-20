namespace NeuralNetwork
{
    public interface INode
    {
        double[] weights { get; set; }

        double CalculateValue(double[] inputs);
        double CalculateValue(Node[] inputs);
    }
}