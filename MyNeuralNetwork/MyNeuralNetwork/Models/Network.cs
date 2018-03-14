using MyNeuralNetwork.Layers;

namespace MyNeuralNetwork.Models
{
    class Network
    {
        public readonly InputLayer InputLayer = new InputLayer();
        public HiddenLayer HiddenLayer = new HiddenLayer(4, 2);
        public OutputLayer OutputLayer = new OutputLayer(2, 4);

        public Network()
        {
            FactResult = new double[InputLayer.Trainset[0].Item2.OpCount];
        }

        public double[] FactResult;

        public const double AllowedErrorRate = 0.001d;
    }
}
