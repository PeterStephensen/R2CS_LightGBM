using System.Globalization;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace OnnxInference;

public static class Program
{
    public static void Main(string[] args)
    {
        string modelPath = args.Length > 0 ? args[0] : FindFile("model/lightgbm_model.onnx");
        string dataPath  = args.Length > 1 ? args[1] : FindFile("data/test_features.csv");

        var (features, numSamples, numFeatures) = LoadCsv(dataPath);
        Console.WriteLine($"Indlæst {numSamples} samples med {numFeatures} features");

        using var session = new InferenceSession(modelPath);

        Console.WriteLine("\nModel inputs:");
        foreach (var kvp in session.InputMetadata)
            Console.WriteLine($"  {kvp.Key}: {kvp.Value.ElementType}, dims=[{string.Join(",", kvp.Value.Dimensions)}]");

        Console.WriteLine("Model outputs:");
        foreach (var kvp in session.OutputMetadata)
            Console.WriteLine($"  {kvp.Key}: {kvp.Value.ElementType}");

        string inputName = session.InputMetadata.Keys.First();
        var inputTensor = new DenseTensor<float>(features, [numSamples, numFeatures]);
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
        };

        using var results = session.Run(inputs);

        var probOutput = results.Last();
        var probTensor = probOutput.AsTensor<float>();

        Console.WriteLine($"\nFørste 10 predictions:");
        Console.WriteLine($"{"Sample",-8} {"P(class=0)",-14} {"P(class=1)",-14}");
        Console.WriteLine(new string('-', 36));

        for (int i = 0; i < Math.Min(10, numSamples); i++)
            Console.WriteLine($"{i,-8} {probTensor[i, 0],-14:F8} {probTensor[i, 1],-14:F8}");

        string rPredPath = Path.Combine(Path.GetDirectoryName(dataPath)!, "test_predictions_r.csv");
        if (File.Exists(rPredPath))
        {
            double[] rPreds = LoadPredictions(rPredPath);
            double maxDiff = 0;
            for (int i = 0; i < numSamples; i++)
                maxDiff = Math.Max(maxDiff, Math.Abs(rPreds[i] - probTensor[i, 1]));

            Console.WriteLine($"\nMax afvigelse vs. R-predictions: {maxDiff:E6}");
        }

        string pyPredPath = Path.Combine(Path.GetDirectoryName(dataPath)!, "test_predictions_python.csv");
        if (File.Exists(pyPredPath))
        {
            double[] pyPreds = LoadPredictions(pyPredPath);
            double maxDiff = 0;
            for (int i = 0; i < numSamples; i++)
                maxDiff = Math.Max(maxDiff, Math.Abs(pyPreds[i] - probTensor[i, 1]));

            Console.WriteLine($"Max afvigelse vs. Python-predictions: {maxDiff:E6}");
        }

        Console.WriteLine("\nFærdig.");
    }

    /// <summary>
    /// Søger efter filen relativt til exe-placering og op til 4 niveauer op i mappetræet.
    /// </summary>
    private static string FindFile(string relativePath)
    {
        string? dir = AppContext.BaseDirectory;
        for (int i = 0; i < 6; i++)
        {
            string candidate = Path.Combine(dir!, relativePath);
            if (File.Exists(candidate))
                return candidate;
            dir = Path.GetDirectoryName(dir);
            if (dir is null) break;
        }

        if (File.Exists(relativePath))
            return Path.GetFullPath(relativePath);

        throw new FileNotFoundException(
            $"Kan ikke finde '{relativePath}'. Kør programmet fra workspace-roden eller angiv stien som argument.");
    }

    private static (float[] data, int rows, int cols) LoadCsv(string path)
    {
        string[] lines = File.ReadAllLines(path).Skip(1).Where(l => l.Length > 0).ToArray();
        int rows = lines.Length;
        string[] firstCols = lines[0].Split(',');
        int cols = firstCols.Length;
        float[] data = new float[rows * cols];

        for (int i = 0; i < rows; i++)
        {
            string[] values = lines[i].Split(',');
            for (int j = 0; j < cols; j++)
                data[i * cols + j] = float.Parse(values[j], CultureInfo.InvariantCulture);
        }

        return (data, rows, cols);
    }

    private static double[] LoadPredictions(string path)
    {
        return File.ReadAllLines(path)
            .Skip(1)
            .Where(l => l.Length > 0)
            .Select(line => double.Parse(line.Trim(), CultureInfo.InvariantCulture))
            .ToArray();
    }
}
