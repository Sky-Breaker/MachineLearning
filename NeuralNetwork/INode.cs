namespace NeuralNetwork
{
    public interface INode
    {
        float[] weights { get; set; }

        float CalculateValue(float[] inputs);
        float CalculateValue(Node[] inputs);
    }
}