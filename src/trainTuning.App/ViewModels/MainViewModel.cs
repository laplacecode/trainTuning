using System.Collections.ObjectModel;
using System.IO;
using System.Text.Json;
using System.Windows;
using Microsoft.Win32;
using TrainTuning.App.Infrastructure;
using TrainTuning.App.Models;
using TrainTuning.App.Services;

namespace TrainTuning.App.ViewModels;

public sealed class MainViewModel : ObservableObject, IDisposable
{
    private readonly PythonWorkerService _workerService = new();
    private CancellationTokenSource? _trainingCts;

    private string _datasetPath = "data/dataset.yaml";
    private string _selectedTask = "目标检测";
    private string _selectedModelFamily = "YOLOv8n（默认）";
    private string _imageCount = "2000";
    private string _classCount = "10";
    private string _gpuMemory = "8";
    private string _selectedObjectSize = "混合尺寸";
    private string _selectedGoal = "均衡";
    private string _selectedClassBalance = "基本均衡";
    private string _selectedLabelQuality = "较高";
    private string _statusText = "准备就绪";
    private string _modelName = "尚未生成";
    private string _summary = "填写数据集与硬件情况，生成第一组训练参数。";
    private string _commandText = string.Empty;
    private int _confidence;
    private double _trainingProgress;
    private bool _isBusy;
    private bool _hasRecommendation;

    public MainViewModel()
    {
        RecommendCommand = new AsyncRelayCommand(RecommendAsync, () => !IsBusy);
        StartTrainingCommand = new AsyncRelayCommand(StartTrainingAsync, () => !IsBusy && HasRecommendation);
        StopTrainingCommand = new RelayCommand(StopTraining, () => IsBusy);
        CopyCommandCommand = new RelayCommand(CopyCommand, () => !string.IsNullOrWhiteSpace(CommandText));
        BrowseDatasetCommand = new RelayCommand(BrowseDataset);
        AddLog("应用已启动，Python 训练引擎等待任务。");
    }

    public IReadOnlyList<string> Tasks { get; } = ["目标检测", "实例分割", "姿态估计", "图像分类", "旋转框 OBB"];
    public IReadOnlyList<string> ModelFamilies { get; } = ["YOLOv8n（默认）", "YOLO11（自动）", "YOLO26（自动）"];
    public IReadOnlyList<string> ObjectSizes { get; } = ["小目标为主", "混合尺寸", "大目标为主"];
    public IReadOnlyList<string> Goals { get; } = ["均衡", "精度优先", "速度优先"];
    public IReadOnlyList<string> ClassBalances { get; } = ["基本均衡", "轻度失衡", "严重失衡"];
    public IReadOnlyList<string> LabelQualities { get; } = ["较高", "一般", "较低"];

    public ObservableCollection<RecommendedParameter> Parameters { get; } = [];
    public ObservableCollection<string> Warnings { get; } = [];
    public ObservableCollection<string> Logs { get; } = [];

    public AsyncRelayCommand RecommendCommand { get; }
    public AsyncRelayCommand StartTrainingCommand { get; }
    public RelayCommand StopTrainingCommand { get; }
    public RelayCommand CopyCommandCommand { get; }
    public RelayCommand BrowseDatasetCommand { get; }

    public string DatasetPath { get => _datasetPath; set => SetProperty(ref _datasetPath, value); }
    public string SelectedTask { get => _selectedTask; set => SetProperty(ref _selectedTask, value); }
    public string SelectedModelFamily { get => _selectedModelFamily; set => SetProperty(ref _selectedModelFamily, value); }
    public string ImageCount { get => _imageCount; set => SetProperty(ref _imageCount, value); }
    public string ClassCount { get => _classCount; set => SetProperty(ref _classCount, value); }
    public string GpuMemory { get => _gpuMemory; set => SetProperty(ref _gpuMemory, value); }
    public string SelectedObjectSize { get => _selectedObjectSize; set => SetProperty(ref _selectedObjectSize, value); }
    public string SelectedGoal { get => _selectedGoal; set => SetProperty(ref _selectedGoal, value); }
    public string SelectedClassBalance { get => _selectedClassBalance; set => SetProperty(ref _selectedClassBalance, value); }
    public string SelectedLabelQuality { get => _selectedLabelQuality; set => SetProperty(ref _selectedLabelQuality, value); }
    public string StatusText { get => _statusText; private set => SetProperty(ref _statusText, value); }
    public string ModelName { get => _modelName; private set => SetProperty(ref _modelName, value); }
    public string Summary { get => _summary; private set => SetProperty(ref _summary, value); }
    public string CommandText { get => _commandText; private set => SetProperty(ref _commandText, value); }
    public int Confidence { get => _confidence; private set => SetProperty(ref _confidence, value); }
    public double TrainingProgress { get => _trainingProgress; private set => SetProperty(ref _trainingProgress, value); }

    public bool IsBusy
    {
        get => _isBusy;
        private set
        {
            if (SetProperty(ref _isBusy, value))
            {
                RefreshCommands();
            }
        }
    }

    public bool HasRecommendation
    {
        get => _hasRecommendation;
        private set
        {
            if (SetProperty(ref _hasRecommendation, value))
            {
                RefreshCommands();
            }
        }
    }

    private async Task RecommendAsync()
    {
        if (!TryCreateRequest(out var tuningRequest))
        {
            return;
        }

        IsBusy = true;
        StatusText = "正在分析条件";
        Parameters.Clear();
        Warnings.Clear();
        AddLog("正在请求参数推荐...");

        try
        {
            var request = new WorkerRequest
            {
                Action = "recommend",
                Payload = tuningRequest
            };

            await _workerService.ExecuteAsync(request, workerEvent =>
                Application.Current.Dispatcher.Invoke(() => HandleWorkerEvent(workerEvent)));
        }
        catch (Exception exception)
        {
            StatusText = "推荐失败";
            AddLog($"错误：{exception.Message}");
            MessageBox.Show(exception.Message, "推荐失败", MessageBoxButton.OK, MessageBoxImage.Error);
        }
        finally
        {
            IsBusy = false;
        }
    }

    private async Task StartTrainingAsync()
    {
        if (!TryCreateRequest(out var tuningRequest) || !HasRecommendation)
        {
            return;
        }

        if (!DatasetCanBeResolved(tuningRequest.DatasetPath))
        {
            MessageBox.Show(
                "当前数据集 YAML 不存在。请先点击“浏览…”选择有效配置，再启动真实训练。",
                "数据集不存在",
                MessageBoxButton.OK,
                MessageBoxImage.Warning);
            return;
        }

        IsBusy = true;
        TrainingProgress = 0;
        StatusText = "训练任务运行中";
        _trainingCts = new CancellationTokenSource();
        AddLog("启动验证训练。未安装 Ultralytics 时将运行模拟训练。");

        try
        {
            var config = Parameters.ToDictionary(
                item => item.Key,
                item => (object)item.Value);

            var request = new WorkerRequest
            {
                Action = "train",
                Payload = new
                {
                    request = tuningRequest,
                    config,
                    dry_run_if_unavailable = true
                }
            };

            await _workerService.ExecuteAsync(
                request,
                workerEvent => Application.Current.Dispatcher.Invoke(() => HandleWorkerEvent(workerEvent)),
                _trainingCts.Token);
        }
        catch (OperationCanceledException)
        {
            StatusText = "训练已停止";
            AddLog("训练任务已由用户停止。");
        }
        catch (Exception exception)
        {
            StatusText = "训练失败";
            AddLog($"错误：{exception.Message}");
        }
        finally
        {
            _trainingCts?.Dispose();
            _trainingCts = null;
            IsBusy = false;
        }
    }

    private void HandleWorkerEvent(WorkerEvent workerEvent)
    {
        switch (workerEvent.Event)
        {
            case "recommendation" when workerEvent.Result is not null:
                ApplyRecommendation(workerEvent.Result);
                break;
            case "progress":
                TrainingProgress = Math.Clamp(workerEvent.Progress ?? 0, 0, 100);
                StatusText = workerEvent.Message ?? $"训练进度 {TrainingProgress:0}%";
                if (workerEvent.Epoch is not null)
                {
                    AddLog($"Epoch {workerEvent.Epoch}/{workerEvent.Epochs} · {workerEvent.Message}");
                }
                break;
            case "completed":
                TrainingProgress = 100;
                StatusText = workerEvent.Message ?? "任务完成";
                AddLog(workerEvent.Message ?? "任务完成。");
                break;
            case "error":
                StatusText = "任务失败";
                AddLog($"训练引擎：{workerEvent.Message}");
                break;
            case "log":
            case "status":
                AddLog(workerEvent.Message ?? workerEvent.Event);
                break;
        }
    }

    private void ApplyRecommendation(Recommendation recommendation)
    {
        ModelName = recommendation.Model;
        Summary = recommendation.Summary;
        Confidence = recommendation.Confidence;
        CommandText = recommendation.Command;
        Parameters.Clear();
        foreach (var parameter in recommendation.Parameters)
        {
            Parameters.Add(parameter);
        }

        Warnings.Clear();
        foreach (var warning in recommendation.Warnings)
        {
            Warnings.Add(warning);
        }

        HasRecommendation = true;
        StatusText = "推荐方案已生成";
        AddLog($"完成：推荐 {recommendation.Model}，可信度 {recommendation.Confidence}%。");
    }

    private bool TryCreateRequest(out TuningRequest request)
    {
        request = new TuningRequest();
        if (!int.TryParse(ImageCount, out var imageCount) || imageCount <= 0 ||
            !int.TryParse(ClassCount, out var classCount) || classCount <= 0 ||
            !double.TryParse(GpuMemory, out var gpuMemory) || gpuMemory <= 0)
        {
            MessageBox.Show("图片数量、类别数量和显存必须是大于 0 的数字。", "输入有误",
                MessageBoxButton.OK, MessageBoxImage.Warning);
            return false;
        }

        request = new TuningRequest
        {
            DatasetPath = string.IsNullOrWhiteSpace(DatasetPath) ? "data/dataset.yaml" : DatasetPath.Trim(),
            Task = MapTask(SelectedTask),
            ModelFamily = MapModelFamily(SelectedModelFamily),
            ModelVariant = MapModelVariant(SelectedModelFamily),
            ImageCount = imageCount,
            ClassCount = classCount,
            GpuMemoryGb = gpuMemory,
            ObjectSize = MapObjectSize(SelectedObjectSize),
            TrainingGoal = MapGoal(SelectedGoal),
            ClassBalance = MapBalance(SelectedClassBalance),
            LabelQuality = MapQuality(SelectedLabelQuality)
        };
        return true;
    }

    private void StopTraining()
    {
        _trainingCts?.Cancel();
        _workerService.Stop();
    }

    private void CopyCommand()
    {
        Clipboard.SetText(CommandText);
        StatusText = "训练命令已复制";
    }

    private void BrowseDataset()
    {
        var dialog = new OpenFileDialog
        {
            Title = "选择 YOLO 数据集配置",
            Filter = "YAML 配置 (*.yaml;*.yml)|*.yaml;*.yml|所有文件 (*.*)|*.*",
            CheckFileExists = true
        };
        if (dialog.ShowDialog() == true)
        {
            DatasetPath = dialog.FileName;
        }
    }

    private void AddLog(string message)
    {
        Logs.Add($"[{DateTime.Now:HH:mm:ss}] {message}");
        while (Logs.Count > 200)
        {
            Logs.RemoveAt(0);
        }
    }

    private void RefreshCommands()
    {
        RecommendCommand.RaiseCanExecuteChanged();
        StartTrainingCommand.RaiseCanExecuteChanged();
        StopTrainingCommand.RaiseCanExecuteChanged();
        CopyCommandCommand.RaiseCanExecuteChanged();
    }

    private static string MapTask(string value) => value switch
    {
        "实例分割" => "segment",
        "姿态估计" => "pose",
        "图像分类" => "classify",
        "旋转框 OBB" => "obb",
        _ => "detect"
    };

    private static string MapObjectSize(string value) => value switch
    {
        "小目标为主" => "small",
        "大目标为主" => "large",
        _ => "mixed"
    };

    private static string MapGoal(string value) => value switch
    {
        "精度优先" => "accuracy",
        "速度优先" => "speed",
        _ => "balanced"
    };

    private static string MapBalance(string value) => value switch
    {
        "轻度失衡" => "mild",
        "严重失衡" => "severe",
        _ => "balanced"
    };

    private static string MapQuality(string value) => value switch
    {
        "一般" => "medium",
        "较低" => "low",
        _ => "high"
    };

    private static string MapModelFamily(string value)
    {
        if (value.StartsWith("YOLOv8", StringComparison.OrdinalIgnoreCase))
        {
            return "yolov8";
        }

        if (value.StartsWith("YOLO26", StringComparison.OrdinalIgnoreCase))
        {
            return "yolo26";
        }

        return "yolo11";
    }

    private static string MapModelVariant(string value) =>
        value.StartsWith("YOLOv8n", StringComparison.OrdinalIgnoreCase) ? "n" : "auto";

    private static bool DatasetCanBeResolved(string datasetPath)
    {
        if (File.Exists(datasetPath) ||
            File.Exists(Path.Combine(AppContext.BaseDirectory, datasetPath)))
        {
            return true;
        }

        // Ultralytics can resolve built-in dataset names such as coco8.yaml.
        return !datasetPath.Contains(Path.DirectorySeparatorChar) &&
               !datasetPath.Contains(Path.AltDirectorySeparatorChar);
    }

    public void Dispose()
    {
        _trainingCts?.Cancel();
        _trainingCts?.Dispose();
        _workerService.Dispose();
    }
}
