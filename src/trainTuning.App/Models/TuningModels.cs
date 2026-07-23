using System.Text.Json;
using System.Text.Json.Serialization;

namespace TrainTuning.App.Models;

public sealed class TuningRequest
{
    public string DatasetPath { get; set; } = "data/dataset.yaml";
    public string Task { get; set; } = "detect";
    public string ModelFamily { get; set; } = "yolov8";
    public string ModelVariant { get; set; } = "n";
    public int ImageCount { get; set; } = 2000;
    public int ClassCount { get; set; } = 10;
    public double GpuMemoryGb { get; set; } = 8;
    public int GpuCount { get; set; } = 1;
    public string ObjectSize { get; set; } = "mixed";
    public string TrainingGoal { get; set; } = "balanced";
    public string ClassBalance { get; set; } = "balanced";
    public string LabelQuality { get; set; } = "high";
    public string SceneType { get; set; } = "fixed_industrial";
    public bool DirectionSensitive { get; set; }
    public string InitialWeightsPath { get; set; } = string.Empty;
}

public sealed class Recommendation
{
    public string Model { get; set; } = string.Empty;
    public string Summary { get; set; } = string.Empty;
    public int Confidence { get; set; }
    public string Command { get; set; } = string.Empty;
    public List<RecommendedParameter> Parameters { get; set; } = [];
    public List<string> Warnings { get; set; } = [];
    public Dictionary<string, JsonElement> Config { get; set; } = [];
    public DatasetAuditResult? DatasetAudit { get; set; }
    public List<ExperimentCandidate> Experiments { get; set; } = [];
}

public sealed class RecommendedParameter
{
    public string Key { get; set; } = string.Empty;
    public string Label { get; set; } = string.Empty;
    public string Value { get; set; } = string.Empty;
    public string Category { get; set; } = string.Empty;
    public string Reason { get; set; } = string.Empty;
}

public sealed class WorkerRequest
{
    public string Id { get; set; } = Guid.NewGuid().ToString("N");
    public string Action { get; set; } = string.Empty;
    public object? Payload { get; set; }
}

public sealed class WorkerControl
{
    public string Action { get; set; } = "adjustment_response";
    public string ProposalId { get; set; } = string.Empty;
    public bool Accepted { get; set; }
}

public sealed class WorkerEvent
{
    public string Id { get; set; } = string.Empty;
    public string Event { get; set; } = string.Empty;
    public string? Message { get; set; }
    public double? Progress { get; set; }
    public int? Epoch { get; set; }
    public int? Epochs { get; set; }
    public Recommendation? Result { get; set; }
    public JsonElement? Data { get; set; }
    public TrainingAdjustment? Adjustment { get; set; }
}

public sealed class DatasetAuditResult
{
    public string Status { get; set; } = string.Empty;
    public string Summary { get; set; } = string.Empty;
    public List<string> Warnings { get; set; } = [];
}

public sealed class TrainingAdjustment
{
    public string ProposalId { get; set; } = string.Empty;
    public int Epoch { get; set; }
    public string Action { get; set; } = string.Empty;
    public string Title { get; set; } = string.Empty;
    public string Reason { get; set; } = string.Empty;
    public List<string> Evidence { get; set; } = [];
    public double? CurrentValue { get; set; }
    public double? ProposedValue { get; set; }
    public int Confidence { get; set; }
    public bool RequiresRestart { get; set; }
}

public sealed class TrainingMetricItem
{
    public string Name { get; init; } = string.Empty;
    public string Value { get; init; } = string.Empty;
}

public sealed class ExperimentCandidate
{
    public string Name { get; set; } = string.Empty;
    public string Change { get; set; } = string.Empty;
    public string Reason { get; set; } = string.Empty;
    public Dictionary<string, JsonElement> ConfigOverrides { get; set; } = [];
}

public static class JsonDefaults
{
    public static readonly JsonSerializerOptions Options = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        PropertyNameCaseInsensitive = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
    };
}
