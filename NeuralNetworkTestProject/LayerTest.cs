using NeuralNetwork;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkTestProject
{
    [TestClass]
    public class LayerTest
    {
        /*
        [TestMethod]
        public void CalculateLayerShouldReturnZero()
        {
            var nodeMock = new Mock<INode>();
            nodeMock.Setup(n => n.CalculateValue(It.IsAny<double[]>())).Returns(0);
            IList<INode> nodes = new List<INode> { new Node(2) }; 

            var l = new Layer(nodes);
            

            //Does this return a value or void?
            //double layerValue =  l.CalculateLayer();

            //Assert?
        }
        */

        [TestMethod]
        public void CalculateLayerShouldGetNodeValues()
        {
            // Arrange
            TestNode[] testNodes = new TestNode[4];
            int nOfInputs = 6;

            double value1 = 0.3f;
            testNodes[0] = new TestNode(value1, nOfInputs);
            double value2 = -0.6f;
            testNodes[1] = new TestNode(value2, nOfInputs);
            double value3 = 1.9f;
            testNodes[2] = new TestNode(value3, nOfInputs);
            double value4 = -1.4f;
            testNodes[3] = new TestNode(value4, nOfInputs);

            Layer testLayer = new Layer(testNodes, null);

            double[] expectedValues = new double[] { value1, value2, value3, value4 };

            // Act
            double[] nullArray = null;
            double[] calculateLayerResults = testLayer.CalculateLayer(nullArray);

            // Assert
            Assert.AreEqual(expectedValues, calculateLayerResults);
        }

        /*
        [TestMethod]
        public void CalculateAllLayersShouldRecursivelyGetOutputNodeValues()
        {
            TestNode[] testNodes = new TestNode[4];
            Layer firstLayer = new Layer();
        }
        */

        private class TestNode : Node
        {
            private double testReturnValue;
            public TestNode(double value, int nOfInputs) : base(nOfInputs)
            {
                testReturnValue = value;
            }

            public new double CalculateValue(double[] unusedInputs)
            {
                return testReturnValue;
            }
        }
    }
}
