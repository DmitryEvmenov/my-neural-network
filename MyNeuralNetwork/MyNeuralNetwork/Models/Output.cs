
namespace MyNeuralNetwork.Models
{
    class Output
    {
        public double Xor { get; set; }
        public double Xand { get; set; }

        public Output(double xor, double xand)
        {
            Xor = xor;
            Xand = xand;
        }
    }
}
