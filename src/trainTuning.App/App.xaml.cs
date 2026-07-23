using System.IO;
using System.Windows;
using System.Windows.Threading;

namespace TrainTuning.App;

public partial class App : Application
{
    protected override void OnStartup(StartupEventArgs e)
    {
        base.OnStartup(e);
        DispatcherUnhandledException += OnDispatcherUnhandledException;

        try
        {
            var window = new MainWindow();
            MainWindow = window;
            window.Show();
        }
        catch (Exception exception)
        {
            WriteCrashLog(exception);
            MessageBox.Show(
                $"trainTuning 启动失败：{exception.Message}\n\n详细信息已写入 startup-error.log。",
                "启动失败",
                MessageBoxButton.OK,
                MessageBoxImage.Error);
            Shutdown(-1);
        }
    }

    private void OnDispatcherUnhandledException(object sender, DispatcherUnhandledExceptionEventArgs e)
    {
        WriteCrashLog(e.Exception);
        e.Handled = true;
        MessageBox.Show(
            $"发生未处理错误：{e.Exception.Message}",
            "trainTuning",
            MessageBoxButton.OK,
            MessageBoxImage.Error);
    }

    private static void WriteCrashLog(Exception exception)
    {
        var logPath = Path.Combine(AppContext.BaseDirectory, "startup-error.log");
        File.AppendAllText(
            logPath,
            $"[{DateTime.Now:yyyy-MM-dd HH:mm:ss}]\n{exception}\n\n");
    }
}
