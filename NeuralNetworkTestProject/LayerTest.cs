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
            nodeMock.Setup(n => n.CalculateValue(It.IsAny<float[]>())).Returns(0);
            IList<INode> nodes = new List<INode> { new Node(2) }; 

            var l = new Layer(nodes);
            

            //Does this return a value or void?
            //float layerValue =  l.CalculateLayer();

            //Assert?
        }
        */

        [TestMethod]
        public void CalculateLayerShouldGetNodeValues()
        {
            // Arrange
            TestNode[] testNodes = new TestNode[4];
            int nOfInputs = 6;

            float value1 = 0.3f;
            testNodes[0] = new TestNode(value1, nOfInputs);
            float value2 = -0.6f;
            testNodes[1] = new TestNode(value2, nOfInputs);
            float value3 = 1.9f;
            testNodes[2] = new TestNode(value3, nOfInputs);
            float value4 = -1.4f;
            testNodes[3] = new TestNode(value4, nOfInputs);

            Layer testLayer = new Layer(testNodes, null);

            float[] expectedValues = new float[] { value1, value2, value3, value4 };

            // Act
            float[] nullArray = null;
            float[] calculateLayerResults = testLayer.CalculateLayer(nullArray);

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

        private class TestNode : Neuron
        {
            private float testReturnValue;
            public TestNode(float value, int nOfInputs) : base(nOfInputs)
            {
                testReturnValue = value;
            }

            public new float CalculateValue(float[] unusedInputs)
            {
                return testReturnValue;
            }
        }
    }
}
