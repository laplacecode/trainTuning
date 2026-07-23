using System.Diagnostics;
using System.IO;
using System.Text;
using System.Text.Json;
using TrainTuning.App.Models;

namespace TrainTuning.App.Services;

public sealed class PythonWorkerService : IDisposable
{
    private readonly SemaphoreSlim _runLock = new(1, 1);
    private Process? _activeProcess;

    public bool IsRunning => _activeProcess is { HasExited: false };

    public async Task ExecuteAsync(
        WorkerRequest request,
        Action<WorkerEvent> onEvent,
        CancellationToken cancellationToken = default)
    {
        await _runLock.WaitAsync(cancellationToken);
        try
        {
            var workerPath = Path.Combine(AppContext.BaseDirectory, "engine", "worker.py");
            if (!File.Exists(workerPath))
            {
                throw new FileNotFoundException("找不到 Python Worker。", workerPath);
            }

            var python = ResolvePythonExecutable();
            var utf8WithoutBom = new UTF8Encoding(encoderShouldEmitUTF8Identifier: false);

            var startInfo = new ProcessStartInfo
            {
                FileName = python,
                Arguments = $"-u \"{workerPath}\"",
                WorkingDirectory = AppContext.BaseDirectory,
                UseShellExecute = false,
                RedirectStandardInput = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                StandardInputEncoding = utf8WithoutBom,
                StandardOutputEncoding = utf8WithoutBom,
                StandardErrorEncoding = utf8WithoutBom,
                CreateNoWindow = true
            };
            startInfo.Environment["PYTHONIOENCODING"] = "utf-8";
            startInfo.Environment["PYTHONUTF8"] = "1";

            _activeProcess = Process.Start(startInfo)
                ?? throw new InvalidOperationException("Python Worker 启动失败。");

            using var cancellationRegistration = cancellationToken.Register(Stop);

            var stderrTask = PumpErrorsAsync(_activeProcess, request.Id, onEvent);
            var json = JsonSerializer.Serialize(request, JsonDefaults.Options);
            await _activeProcess.StandardInput.WriteLineAsync(json);
            _activeProcess.StandardInput.Close();

            while (await _activeProcess.StandardOutput.ReadLineAsync(cancellationToken) is { } line)
            {
                if (string.IsNullOrWhiteSpace(line))
                {
                    continue;
                }

                try
                {
                    var workerEvent = JsonSerializer.Deserialize<WorkerEvent>(line, JsonDefaults.Options);
                    if (workerEvent is not null)
                    {
                        onEvent(workerEvent);
                    }
                }
                catch (JsonException)
                {
                    onEvent(new WorkerEvent
                    {
                        Id = request.Id,
                        Event = "log",
                        Message = line
                    });
                }
            }

            await _activeProcess.WaitForExitAsync(cancellationToken);
            await stderrTask;

            if (_activeProcess.ExitCode != 0 && !cancellationToken.IsCancellationRequested)
            {
                throw new InvalidOperationException($"Python Worker 异常退出，代码 {_activeProcess.ExitCode}。");
            }
        }
        finally
        {
            _activeProcess?.Dispose();
            _activeProcess = null;
            _runLock.Release();
        }
    }

    public void Stop()
    {
        try
        {
            if (_activeProcess is { HasExited: false })
            {
                _activeProcess.Kill(entireProcessTree: true);
            }
        }
        catch (InvalidOperationException)
        {
            // Process already exited.
        }
    }

    private static async Task PumpErrorsAsync(
        Process process,
        string requestId,
        Action<WorkerEvent> onEvent)
    {
        while (await process.StandardError.ReadLineAsync() is { } line)
        {
            if (!string.IsNullOrWhiteSpace(line))
            {
                onEvent(new WorkerEvent
                {
                    Id = requestId,
                    Event = "log",
                    Message = line
                });
            }
        }
    }

    private static string ResolvePythonExecutable()
    {
        var configured = Environment.GetEnvironmentVariable("TRAINTUNING_PYTHON");
        if (!string.IsNullOrWhiteSpace(configured))
        {
            return configured;
        }

        var bundledCandidates = new[]
        {
            Path.Combine(AppContext.BaseDirectory, "runtime", "python", "python.exe"),
            Path.Combine(AppContext.BaseDirectory, "python", "python.exe")
        };
        return bundledCandidates.FirstOrDefault(File.Exists) ?? "python";
    }

    public void Dispose()
    {
        Stop();
        _activeProcess?.Dispose();
        _runLock.Dispose();
    }
}
