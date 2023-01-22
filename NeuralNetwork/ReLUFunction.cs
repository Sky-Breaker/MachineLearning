namespace NeuralNetwork
{
    public class ReLUFunction : ActivationFunction
    {
        public double ValueAt(double x)
        {
            if (x < 0)
            {
                return 0;
            } 
            else
            {
                return x;
            }
        }

        public double DerivativeAt(double x)
        {
            return ValueAt(x);
        }

    }
}
