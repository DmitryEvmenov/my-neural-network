using MyNeuralNetwork.Layers;

namespace MyNeuralNetwork.Models
{
    class Network
    {
        public readonly InputLayer InputLayer = new InputLayer();
        public HiddenLayer HiddenLayer = new HiddenLayer(4, 2);
        public OutputLayer OutputLayer = new OutputLayer(2, 4);

        public double[] Fact = new double[2];
       
    }
}
