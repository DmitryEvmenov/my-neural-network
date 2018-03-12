
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

        public double[] ToDoubles() => new[] { Xor, Xand };

        public int OpCount = 2;

        public double this[int i] => i == 0
            ? Xor
            : Xand;
    }
}
